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

.PHONY: all clean debug test-model demo-model ref-model ref-v2 micro-hf mixtral-micro stream-mixtral stream-qwen stream-deepseek smoke bench-smoke warm-smoke test test-name-maps parity parity-v2 m2 exact-f32 live-moe live-olmoe check version

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
	./$(TARGET) bench --model models/tiny-test.rabbit --tokens 8 --runs 2 --json | head -1
	@echo "bench-smoke: ok"

warm-smoke: $(TARGET) ref-v2
	./$(TARGET) warm --model models/ref-v2.rabbit --tokens 8
	@echo "warm-smoke: ok"

check: test test-name-maps smoke bench-smoke parity parity-v2 m2 exact-f32 mixtral-micro stream-mixtral stream-qwen stream-deepseek
	@echo "check: all passed"

debug: CFLAGS += -g -O0 -DRABBIT_DEBUG
debug: clean $(TARGET)

clean:
	rm -f $(OBJ) $(TARGET) $(TEST_TOOL) $(TEST_BIN)

# Live Mixtral-style MoE (TinyBrainBot 360M).
live-moe: $(TARGET)
	@mkdir -p models/tinybrainbot
	@python3 -c "import huggingface_hub, safetensors, numpy" 2>/dev/null || \
		(echo "pip install huggingface_hub safetensors numpy transformers" && exit 1)
	@echo "[live-moe] downloading TinyBrainBot 360M (expect ~1.5 GB model.safetensors)..."
	@python3 -c "from huggingface_hub import snapshot_download; snapshot_download('nkthebass/tinybrainbot-360m-a75m-v4.1-moe-base', local_dir='models/tinybrainbot', allow_patterns=['config.json','*.safetensors','tokenizer.json','tokenizer.model','tokenizer_config.json','model.safetensors.index.json','generation_config.json'])" || true
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
	@echo "[live-moe] converting (quant=$(LIVE_QUANT)). Streaming reader: peak RAM ≈ largest tensor..."
	python3 tools/convert_hf_stream.py --model-dir models/tinybrainbot \
		--out models/tinybrainbot-v2.rabbit --quant $(LIVE_QUANT) --name tinybrainbot
	python3 tools/export_hf_tokenizer.py --hf-dir models/tinybrainbot \
		--out models/tinybrainbot.tok || true
	./$(TARGET) probe-v2 --model models/tinybrainbot-v2.rabbit \
		--tokenizer models/tinybrainbot.tok --prompt Hello --tokens 16
	@echo "Chat: ./rabbit chat --model models/tinybrainbot-v2.rabbit --tokenizer models/tinybrainbot.tok"

# Live ≥1B public MoE: allenai/OLMoE-1B-7B-0924 (7B total / ~1.3B active).
LIVE_OLMOE_REPO ?= allenai/OLMoE-1B-7B-0924
live-olmoe: $(TARGET)
	@mkdir -p models/olmoe
	@python3 -c "import huggingface_hub, numpy" 2>/dev/null || \
		(echo "pip install huggingface_hub numpy" && exit 1)
	@echo "[live-olmoe] downloading $(LIVE_OLMOE_REPO) (expect ~14 GB across 3 shards)..."
	@python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$(LIVE_OLMOE_REPO)', local_dir='models/olmoe', allow_patterns=['config.json','*.safetensors','tokenizer.json','tokenizer.model','tokenizer_config.json','model.safetensors.index.json','special_tokens_map.json','generation_config.json'])" || true
	@python3 -c "from pathlib import Path; p=Path('models/olmoe'); files=sorted(p.glob('model-*-of-*.safetensors')) or list(p.glob('*.safetensors')); total=sum(f.stat().st_size for f in files if f.is_file()); print(f'[live-olmoe] safetensors files={len(files)} total_bytes={total}'); import sys; sys.exit(0 if total > 10_000_000_000 else 1)" \
	|| ( echo "[live-olmoe] shards missing/small — trying curl for 3 shards..."; \
	     for i in 1 2 3; do \
	       f=model-0000$${i}-of-00003.safetensors; \
	       sz=$$(stat -c%s models/olmoe/$$f 2>/dev/null || echo 0); \
	       if [ "$$sz" -lt 100000000 ]; then \
	         curl -L --fail --retry 3 -o models/olmoe/$$f \
	           "https://huggingface.co/$(LIVE_OLMOE_REPO)/resolve/main/$$f" \
	           || (echo "Download failed for $$f. Try HF_TOKEN=... make live-olmoe" && exit 1); \
	       fi; \
	     done )
	@python3 -c "from pathlib import Path; p=Path('models/olmoe'); files=list(p.glob('model-*-of-*.safetensors')) or list(p.glob('*.safetensors')); total=sum(f.stat().st_size for f in files if f.is_file()); print(f'[live-olmoe] ready: {len(files)} files, {total/1e9:.2f} GB'); import sys; sys.exit(0 if total > 10_000_000_000 else 1)" \
	|| (echo "ERROR: OLMoE shards still incomplete. Aborting."; exit 1)
	@echo "[live-olmoe] converting (quant=$(LIVE_QUANT)). Streaming reader: peak RAM ≈ largest tensor..."
	python3 tools/convert_hf_stream.py --model-dir models/olmoe \
		--out models/olmoe-v2.rabbit --quant $(LIVE_QUANT) --name olmoe-1b-7b
	python3 tools/export_hf_tokenizer.py --hf-dir models/olmoe \
		--out models/olmoe.tok || true
	@echo "[live-olmoe] probe (RABBIT_NO_MMAP=1 recommended on FUSE/network FS)..."
	RABBIT_NO_MMAP=1 RABBIT_CACHE_SLOTS=64 ./$(TARGET) probe-v2 \
		--model models/olmoe-v2.rabbit --tokenizer models/olmoe.tok \
		--prompt "Hello" --tokens 16
	@echo "Chat: RABBIT_NO_MMAP=1 RABBIT_CACHE_SLOTS=64 ./rabbit chat --model models/olmoe-v2.rabbit --tokenizer models/olmoe.tok"
	@echo "Tips: copy .rabbit to local disk if FUSE is flaky; RABBIT_REF_V2_CTX=128 to shrink KV."
