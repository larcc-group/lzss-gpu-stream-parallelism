include ../../Makefile.inc

LD = $(NVCC)
LZSSPATH = ../src
LDFLAGS = -L. -L$(CUDAPATH) -L$(LZSSPATH) -llzss

tests: test_lzss_memory.o lzsslib
	$(LD) $< -I$(CUDAINCLUDE) $(LIBS) $(LDFLAGS) -o $@ -lcudart -lcrypto -lOpenCL

test_lzss_memory.o: test_lzss_memory.cu
	$(NVCC) $(CFLAGS) -D__CL_ENABLE_EXCEPTIONS -I../src -c $< -o $@ 

lzsslib:
	$(MAKE) -C $(LZSSPATH)
clean:
	rm -f *.o tests