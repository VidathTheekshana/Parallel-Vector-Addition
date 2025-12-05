
# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -O2

# Target executable
TARGET = vector_add_serial

# Source file
SRC = vector_add_serial.c

# Default target
all: $(TARGET)

# Compile the program
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)
	@echo "Compilation successful! Run with: ./$(TARGET)"

# Clean compiled files
clean:
	@rm -f $(TARGET)
	@echo "Cleaned up compiled files"

# Run the program
run: $(TARGET)
	./$(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  make          - Compile the program"
	@echo "  make all      - Compile the program"
	@echo "  make clean    - Remove compiled files"
	@echo "  make run      - Compile and run the program"
	@echo "  make help     - Show this help message"

.PHONY: all clean run help
	@echo "  make test     - Run quick test (invokes the program)"
