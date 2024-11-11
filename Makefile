# Python version and virtual environment settings
PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin

# OS detection for different path separators and commands
ifeq ($(OS),Windows_NT)
    VENV_PYTHON := $(VENV_BIN)/python.exe
    VENV_PIP := $(VENV_BIN)/pip.exe
    RM := rmdir /s /q
else
    VENV_PYTHON := $(VENV_BIN)/python
    VENV_PIP := $(VENV_BIN)/pip
    RM := rm -rf
endif

# Default target
.PHONY: all
all: install run

# Create virtual environment
$(VENV):
	$(PYTHON) -m venv $(VENV)

# Install dependencies
.PHONY: install
install: $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install flask numpy matplotlib scikit-learn scipy

# Create static directory
.PHONY: setup
setup:
	mkdir -p static

# Run the Flask application
.PHONY: run
run: setup
	$(VENV_PYTHON) app.py

# Clean up
.PHONY: clean
clean:
	$(RM) $(VENV)
	$(RM) static/*.png
	$(RM) __pycache__
	$(RM) .pytest_cache

# Help target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make install  - Create virtual environment and install dependencies"
	@echo "  make run     - Run the Flask application"
	@echo "  make clean   - Remove virtual environment and generated files"
	@echo "  make all     - Install dependencies and run the application"
	@echo "  make help    - Show this help message"