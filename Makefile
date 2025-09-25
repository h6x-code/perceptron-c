# --- Compiler / Flags --------------------------------------------------------

CC       := clang
CFLAGS   := -std=c11 -O2 -Wall -Wextra -pedantic -Isrc
LDFLAGS  := -lm -pthread

# Address/UB sanitizers (opt-in when debugging)
SAN_FLAGS := -fsanitize=address,undefined -fno-omit-frame-pointer

# --- Project layout ----------------------------------------------------------

SRC_DIR  := src
TEST_DIR := tests
BIN      := perceptron

SRC := $(wildcard $(SRC_DIR)/*.c) $(wildcard $(TEST_DIR)/*.c)

# --- Default build -----------------------------------------------------------

.PHONY: all
all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# --- Dev builds --------------------------------------------------------------

.PHONY: debug
debug: CFLAGS += -O0 -g $(SAN_FLAGS)
debug: clean $(BIN)

.PHONY: release
release: CFLAGS := -std=c11 -O3 -DNDEBUG -Wall -Wextra -pedantic
release: clean $(BIN)

# --- Utilities ---------------------------------------------------------------

.PHONY: run
run: all
	./$(BIN) --help

.PHONY: clean
clean:
	$(RM) $(BIN)
