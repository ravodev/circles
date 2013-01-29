NVFLAGS=-arch=compute_20 -code=sm_20 -O2
# list .c and .cu source files here
SRCFILES=ray.cu Image.cpp

all:	ray_cuda	

ray_cuda:  
	nvcc $(NVFLAGS) ray.cu Image.cpp -o ray_cuda $^

gprof:
	nvcc $(NVFLAGS) -pg ray.cu Image.cpp -o ray_cuda $^

clean: 
	rm -f *.o ray_cuda *.tga gmon.out
