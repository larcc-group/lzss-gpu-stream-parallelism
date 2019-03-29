############################################################################
# Makefile for lzss encode/decode library and app program
############################################################################

CUDAPATH = /usr/local/cuda/lib64 
CUDAINCLUDE = /usr/local/cuda/include
SPAR_PATH = $(HOME)/spar/
SPAR = $(SPAR_PATH)bin/spar
SPAR_LIB = $(SPAR_PATH)libraries
CC = g++
LD = nvcc
BLOCK_SIZE=256
NVCC = nvcc
NVCCFLAGS = -D__CL_ENABLE_EXCEPTIONS -Xcompiler="--std=c++11" -I$(CUDAINCLUDE) --std=c++11 -arch=sm_52 -DBLOCK_SIZE=$(BLOCK_SIZE)
CFLAGS = -D__CL_ENABLE_EXCEPTIONS -std=c++11 -g -I. -I$(CUDAINCLUDE) -DBATCH_SIZE=4194304 -DBLOCK_SIZE=$(BLOCK_SIZE) -c # -DPRINTOFFSETS -DSEQUENTIALORIGINAL

LDFLAGS =  -o
ifndef DEBUG
NVCCFLAGS := -O3 $(NVCCFLAGS)
CFLAGS := -O3 $(CFLAGS)
LDFLAGS := -O3 $(LDFLAGS)
else
NVCCFLAGS := -O0 $(NVCCFLAGS)
CFLAGS := -O0 $(CFLAGS)
LDFLAGS := -O0 $(LDFLAGS)

endif
#-Wall -Wextra -pedantic -ansi 


ifdef USE_CUDA
CFLAGS += -DUSE_CUDA
endif 
# ifndef DEBUG
# CFLAGS +=  -O3# $(CFLAGS)
# endif
# libraries
LIBS = -L. -L$(CUDAPATH) -llzss -loptlist

# Treat NT and non-NT windows the same
ifeq ($(OS),Windows_NT)
	OS = Windows
endif

ifeq ($(OS),Windows)
	ifeq ($(OSTYPE), cygwin)
		EXE = .exe
		DEL = rm
	else
		EXE = .exe
		DEL = del
	endif
else	#assume Linux/Unix
	EXE =
	DEL = rm -f
endif

FMOBJ = brute.o

MATCHERS = matcher_sequential.o matcher_cuda.o matcher_openacc.o matcher_opencl.o
LZOBJS = $(FMOBJ) lzss_gpu.o lzss.o $(MATCHERS) gpu_util.o  bitmemory.o lzss_spar.o

all:		app$(EXE) liblzss.a liboptlist.a

app$(EXE):	app.o liblzss.a liboptlist.a
		$(LD) $< -I$(CUDAINCLUDE) $(LIBS) $(LDFLAGS) $@ -lcudart -lOpenCL

app.o:	app.cpp lzss.h optlist.h
		$(CC) $(CFLAGS) $< -fpermissive

liblzss.a:	$(LZOBJS) bitfile.o 
		ar crv liblzss.a $(LZOBJS) bitfile.o
		ranlib liblzss.a

lzss.o:	lzss.cpp lzlocal.h bitfile.h file_stream.h
		$(CC) $(CFLAGS) $<

lzss_gpu.o: lzss_gpu.cpp lzss.h file_stream.h
		$(CC) -I$(CUDAINCLUDE) $(CFLAGS) $<
		

lzss_spar.o: lzss_spar.cpp lzss.h file_stream.h
		$(SPAR) -I$(CUDAINCLUDE) $(CFLAGS) -spar_ordered -spar_file $< -o $@
		#$(CC) -I$(CUDAINCLUDE) $(CFLAGS) $< -o $@
		
brute.o:	brute.cpp lzlocal.h
		$(CC) $(CFLAGS) $<

bitmemory.o:	bitmemory.cpp bitmemory.h
		$(CC) $(CFLAGS) $< -o $@

liboptlist.a:	optlist.o
		ar crv liboptlist.a optlist.o
		ranlib liboptlist.a

optlist.o:	optlist.cpp optlist.h
		$(CC) $(CFLAGS) $<


#matchers
matcher_sequential.o: matchers/matcher_sequential.cpp matchers/matcher_base.h
		$(CC) $(CFLAGS) $< -o $@

matcher_cuda.o: matchers/matcher_cuda.cu matchers/matcher_base.h
		$(NVCC) -default-stream per-thread $(NVCCFLAGS)  -c $< -o $@ -I.

matcher_opencl.o: matchers/matcher_opencl.cpp matchers/matcher_base.h
		$(NVCC) $(NVCCFLAGS) -c $< -o $@ -I. 

gpu_util.o: gpu_util.cpp gpu_util.h
		$(NVCC) $(NVCCFLAGS) -c $< -o $@ -I. 

matcher_openacc.o: matchers/matcher_openacc.cpp matchers/matcher_base.h
		$(CC) $(CFLAGS) $< -o $@ 

clean:
		$(DEL) *.o
		$(DEL) *.a
		$(DEL) app$(EXE)
