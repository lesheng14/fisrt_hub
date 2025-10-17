NVCC ?= nvcc
TARGET ?= build/scnn
SRC := src/scnn.cu

GENCODE ?= -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52
NVCCFLAGS ?= -O3 -std=c++17 $(GENCODE)

.PHONY: all build clean run data params

all: build $(TARGET)

build:
	mkdir -p build

$(TARGET): $(SRC) | build
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -rf build

run: $(TARGET) data params
	./$(TARGET) models/random

# Download FashionMNIST test set to data/FashionMNIST/raw
data:
	bash scripts/get_fashion_mnist.sh

# Generate random model parameter text files into models/random
params:
	python3 scripts/gen_random_params.py
