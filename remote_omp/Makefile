CC = gcc
CXX = g++
CFLAGS = -lm -O3 -march=native -mtune=native -ftree-vectorize
kmeans_omp: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = kmeans kmeans_omp


.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
