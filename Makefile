CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11 -I./src
LDFLAGS = -lm -lpthread

SRC = src/main.c src/memory.c src/model.c src/moe.c src/quant.c src/tokenizer.c src/utils.c
OBJ = $(SRC:.c=.o)

TARGET = rabbit

.PHONY: all clean debug

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

debug: CFLAGS += -g -O0 -DRABBIT_DEBUG
debug: clean $(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)

# Placeholder for future model conversion
convert:
	@echo "Conversion tools coming soon"
