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

.PHONY: all clean debug test-model demo-model ref-model ref-v2 micro-hf mixtral-micro smoke test parity parity-v2 m2 exact-f32 live-moe check

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_TOOL): tools/create_test_model.c
	$(CC) -O2 -Wall -Wextra -o $@ $<

$(TEST_BIN): tests/test_quant.c $(LIB_OBJ)
	$(CC) $(CFLAGS) tests/test_quant.c $(LIB_OBJ) -o $@ $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

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

# GQA + Mixtral tensor names (w1/w2/w3, block_sparse_moe) — offline stand-in for TinyBrainBot
mixtral-micro: $(TARGET)
	@mkdir -p models
	python3 tools/create_mixtral_micro.py --out-dir models/hf_mixtral_micro --seed 7
	python3 tools/convert_hf_to_rabbit.py --model-dir models/hf_mixtral_micro \
		--out models/mixtral-micro.rabbit --layout v2 --quant 4
	./$(TARGET) probe-v2 --model models/mixtral-micro.rabbit --prompt hi --tokens 8

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

check: test smoke parity parity-v2 m2 exact-f32 mixtral-micro
	@echo "check: all passed"

debug: CFLAGS += -g -O0 -DRABBIT_DEBUG
debug: clean $(TARGET)

clean:
	rm -f $(OBJ) $(TARGET) $(TEST_TOOL) $(TEST_BIN)

# Live Mixtral-style MoE (TinyBrainBot 360M).
# Requires: pip install huggingface_hub safetensors numpy
# Uses streaming converter so ~1GB RAM hosts can convert the 1.5GB f32 checkpoint.
live-moe: $(TARGET)
	@mkdir -p models/tinybrainbot
	python3 -c "from huggingface_hub import snapshot_download; snapshot_download('nkthebass/tinybrainbot-360m-a75m-v4.1-moe-base', local_dir='models/tinybrainbot', allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer.model','tokenizer_config.json'])"
	python3 tools/convert_hf_stream.py --model-dir models/tinybrainbot \
		--out models/tinybrainbot-v2.rabbit --quant 4
	python3 tools/export_hf_tokenizer.py --hf-dir models/tinybrainbot --out models/tinybrainbot.tok || true
	./$(TARGET) probe-v2 --model models/tinybrainbot-v2.rabbit --prompt Hello --tokens 16
	@echo "Chat: ./rabbit chat --model models/tinybrainbot-v2.rabbit --tokenizer models/tinybrainbot.tok"
