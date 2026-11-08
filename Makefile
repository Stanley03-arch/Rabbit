CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11 -I./src
LDFLAGS ?= -lm

SRC = src/main.c src/memory.c src/model.c src/moe.c src/quant.c \
      src/tokenizer.c src/utils.c src/attention.c src/forward_v2.c
OBJ = $(SRC:.c=.o)

LIB_SRC = src/memory.c src/model.c src/moe.c src/quant.c \
          src/tokenizer.c src/utils.c src/attention.c src/forward_v2.c
LIB_OBJ = $(LIB_SRC:.c=.o)

TARGET = rabbit
TEST_TOOL = tools/create_test_model
TEST_BIN = tests/test_quant

# Optional quant for live-moe (0=f32 parity, 4=group-int4 disk-friendly)
LIVE_QUANT ?= 4

.PHONY: all clean debug test-model demo-model ref-model ref-v2 micro-hf mixtral-micro stream-mixtral stream-qwen stream-deepseek smoke bench-smoke test test-name-maps parity parity-v2 m2 exact-f32 live-moe check version

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_TOOL): tools/create_test_model.c
	$(CC) -O2 -Wall -Wextra -o $@ $<

$(TEST_BIN): tests/test_quant.c $(LIB_OBJ)
	$(CC) $(CFLAGS) tests/test_quant.c $(LIB_OBJ) -o $@ $(LDFLAGS)

version: $(TARGET)
	./$(TARGET) version

test: $(TEST_BIN)
	./$(TEST_BIN)

test-name-maps:
	python3 tests/test_name_maps.py

test-model: $(TEST_TOOL)
	@mkdir -p models
	./$(TEST_TOOL) models/tiny-test.rabbit

ref-model:
	@mkdir -p models
	python3 tools/create_ref_model.py --out models/ref-v1.rabbit --seed 1

ref-v2:
	@mkdir -p models
	python3 tools/create_ref_v2.py --out models/ref-v2.rabbit --seed 2

micro-hf:
	@mkdir -p models
	python3 tools/create_hf_micro_moe.py --out-dir models/hf_micro_moe --seed 3
	python3 tools/convert_hf_to_rabbit.py --model-dir models/hf_micro_moe \
		--out models/micro-hf-v2.rabbit --layout v2 --quant 8

# GQA + Mixtral tensor names (w1/w2/w3, block_sparse_moe)
mixtral-micro: $(TARGET)
	@mkdir -p models
	python3 tools/create_mixtral_micro.py --out-dir models/hf_mixtral_micro --seed 7
	python3 tools/convert_hf_to_rabbit.py --model-dir models/hf_mixtral_micro \
		--out models/mixtral-micro.rabbit --layout v2 --quant 4
	./$(TARGET) probe-v2 --model models/mixtral-micro.rabbit --prompt hi --tokens 8

# Streaming converter offline paths (needs pip install safetensors)
stream-mixtral: $(TARGET)
	python3 tests/test_stream_mixtral.py

stream-qwen: $(TARGET)
	python3 tests/test_stream_qwen_moe.py

stream-deepseek: $(TARGET)
	python3 tests/test_stream_deepseek.py

demo-model:
	@mkdir -p models
	python3 tools/convert_hf_to_rabbit.py --demo --out models/demo-moe.rabbit \
		--hidden-dim 128 --n-experts 16 --quant 4

parity: $(TARGET) ref-model
	python3 tests/test_parity_ref.py

parity-v2: $(TARGET) ref-v2
	python3 tests/test_parity_ref_v2.py

m2: $(TARGET) micro-hf
	python3 tests/test_m2_micro_hf.py

exact-f32: $(TARGET)
	python3 tests/test_exact_f32.py

smoke: $(TARGET) test-model
	@printf 'hello\nquit\n' | ./$(TARGET) chat --model models/tiny-test.rabbit \
		--tokens 16 --temp 0.5 >/tmp/rabbit-smoke.out 2>/tmp/rabbit-smoke.err || true
	@grep -q "Rabbit" /tmp/rabbit-smoke.out /tmp/rabbit-smoke.err
	@echo "smoke: ok"

bench-smoke: $(TARGET) test-model
	./$(TARGET) bench --model models/tiny-test.rabbit --tokens 8 --runs 2
	@echo "bench-smoke: ok"

check: test test-name-maps smoke bench-smoke parity parity-v2 m2 exact-f32 mixtral-micro stream-mixtral stream-qwen stream-deepseek
	@echo "check: all passed"

debug: CFLAGS += -g -O0 -DRABBIT_DEBUG
debug: clean $(TARGET)

clean:
	rm -f $(OBJ) $(TARGET) $(TEST_TOOL) $(TEST_BIN)

# Live Mixtral-style MoE (TinyBrainBot 360M).
# Requires ~3 GB free RAM for safetensors mmap + conversion (1.5 GB checkpoint).
# HF Xet/CDN sometimes yields 0-byte files via huggingface_hub; we validate and fall back to curl.
live-moe: $(TARGET)
	@mkdir -p models/tinybrainbot
	@python3 -c "import huggingface_hub, safetensors, numpy" 2>/dev/null || \
		(echo "pip install huggingface_hub safetensors numpy transformers" && exit 1)
	@echo "[live-moe] downloading TinyBrainBot 360M (expect ~1.5 GB model.safetensors)..."
	@python3 -c "from huggingface_hub import snapshot_download; snapshot_download('nkthebass/tinybrainbot-360m-a75m-v4.1-moe-base', local_dir='models/tinybrainbot', allow_patterns=['config.json','*.safetensors','tokenizer.json','tokenizer.model','tokenizer_config.json','model.safetensors.index.json','generation_config.json'])" || true
	@# Validate size; HF Xet often leaves a 0-byte placeholder without a token / proper client
	@SZ=$$(stat -c%s models/tinybrainbot/model.safetensors 2>/dev/null || echo 0); \
	if [ "$$SZ" -lt 100000000 ]; then \
		echo "[live-moe] model.safetensors is missing or too small ($$SZ bytes). Falling back to curl..."; \
		curl -L --fail --retry 3 -o models/tinybrainbot/model.safetensors \
			"https://huggingface.co/nkthebass/tinybrainbot-360m-a75m-v4.1-moe-base/resolve/main/model.safetensors" \
			|| (echo "Download failed. Try: HF_TOKEN=... make live-moe" && exit 1); \
	fi
	@SZ=$$(stat -c%s models/tinybrainbot/model.safetensors 2>/dev/null || echo 0); \
	echo "[live-moe] model.safetensors size=$$SZ bytes"; \
	if [ "$$SZ" -lt 1000000000 ]; then \
		echo "ERROR: model.safetensors still too small ($$SZ). Aborting."; exit 1; \
	fi
	@echo "[live-moe] converting (quant=$(LIVE_QUANT)). Needs ~3 GB free RAM for mmap..."
	python3 tools/convert_hf_stream.py --model-dir models/tinybrainbot \
		--out models/tinybrainbot-v2.rabbit --quant $(LIVE_QUANT) --name tinybrainbot
	python3 tools/export_hf_tokenizer.py --hf-dir models/tinybrainbot \
		--out models/tinybrainbot.tok || true
	./$(TARGET) probe-v2 --model models/tinybrainbot-v2.rabbit \
		--tokenizer models/tinybrainbot.tok --prompt Hello --tokens 16
	@echo "Chat: ./rabbit chat --model models/tinybrainbot-v2.rabbit --tokenizer models/tinybrainbot.tok"
