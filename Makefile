CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11 -I./src
LDFLAGS = -lm -lpthread

SRC = src/main.c src/memory.c src/model.c src/moe.c src/quant.c src/tokenizer.c src/utils.c src/attention.c
OBJ = $(SRC:.c=.o)

TARGET = rabbit
TEST_TOOL = tools/create_test_model

.PHONY: all clean debug test-model demo-model

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_TOOL): tools/create_test_model.c
	$(CC) -O2 -o $@ $<

test-model: $(TEST_TOOL)
	@mkdir -p models
	./$(TEST_TOOL) models/tiny-test.rabbit

# Larger synthetic MoE with int4 experts
demo-model:
	@mkdir -p models
	python3 tools/convert_hf_to_rabbit.py --demo --out models/demo-moe.rabbit --hidden-dim 128 --n-experts 16 --quant 4

debug: CFLAGS += -g -O0 -DRABBIT_DEBUG
debug: clean $(TARGET)

clean:
	rm -f $(OBJ) $(TARGET) $(TEST_TOOL)
