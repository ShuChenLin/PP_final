CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3 -march=native -mtune=native -ftree-vectorize
hw2a: CFLAGS += -pthread
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = kmeans kmeans_pthread


.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
