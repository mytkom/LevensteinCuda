CC := nvcc
CFLAGS := -std=c++11 -allow-unsupported-compiler
TARGET := levcuda
INCL := -Isrc
SRCS := main.cu src/utils.cpp src/kernel.cu src/cpu_calculator.cpp src/gpu_calculator.cu

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(INCL) $^ -o $(TARGET)

clean:
	rm -f $(TARGET) *_transformations.txt
