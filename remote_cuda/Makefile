NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler
CXXFLAGS := -fopenmp
LDFLAGS  := -lm
EXES     := final sequential

alls: $(EXES)

clean:
	rm -f $(EXES)

sequential: kmeans.cc
	g++ $(CXXFLAGS) $(LDFLAGS) -O3 -o $@ $?

final: kmeans_cuda.cu
	nvcc $(NVFLAGS) $(CXXFLAGS) $(LDFLAGS) -o $@ $?
