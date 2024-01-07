NVCXX = nvc++
NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler
NVCXXFLAGS = -O3 -std=c++17 -fast -acc -gpu=cc60 -Minfo=accel 
LDFLAGS  := -lm
EXES     := kmeans_cuda kmeans_acc
TARGETS  := $(EXES)

alls: $(EXES)

clean:
	rm -f $(EXES)

kmeans_cuda: kmeans_cuda.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

%: %.cpp
	$(NVCXX) $(NVCXXFLAGS) $(MNIST) -o $@ $<