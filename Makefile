CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11 -I./src
LDFLAGS ?= -lm

SRC = src/main.c src/memory.c src/model.c src/moe.c src/quant.c \
      src/tokenizer.c src/utils.c src/attention.c
OBJ = $(SRC:.c=.o)

LIB_SRC = src/memory.c src/model.c src/moe.c src/quant.c \
          src/tokenizer.c src/utils.c src/attention.c
LIB_OBJ = $(LIB_SRC:.c=.o)

TARGET = rabbit
TEST_TOOL = tools/create_test_model
TEST_BIN = tests/test_quant

.PHONY: all clean debug test-model demo-model smoke test check

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

demo-model:
	@mkdir -p models
	python3 tools/convert_hf_to_rabbit.py --demo --out models/demo-moe.rabbit \
		--hidden-dim 128 --n-experts 16 --quant 4

smoke: $(TARGET) test-model
	@printf 'hello\nquit\n' | ./$(TARGET) chat --model models/tiny-test.rabbit \
		--tokens 16 --temp 0.5 >/tmp/rabbit-smoke.out 2>/tmp/rabbit-smoke.err || true
	@grep -q "Rabbit" /tmp/rabbit-smoke.out /tmp/rabbit-smoke.err
	@echo "smoke: ok"

check: test smoke
	@echo "check: all passed"

debug: CFLAGS += -g -O0 -DRABBIT_DEBUG
debug: clean $(TARGET)

clean:
	rm -f $(OBJ) $(TARGET) $(TEST_TOOL) $(TEST_BIN)
