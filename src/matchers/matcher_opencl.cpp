#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <chrono>
#include <fstream>
#include "lzss.h"

#include "lzlocal.h"
#include "matcher_base.h"
#include "gpu_util.h"

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>


cl::Kernel FindMatchKernelCache;
cl::Kernel FindMatchWithoutBufferKernelCache;
bool openClKernelInitialized = false;
int MatcherOpenCl::Init(){
    MatcherBase::Init();
    if(openClKernelInitialized){
        context = getOpenClDefaultContext();    

        FindMatchKernel = FindMatchKernelCache;
        FindMatchWithoutBufferKernel = FindMatchWithoutBufferKernelCache;
        return 0;
    }
    cl::Program program;
    try{

        context = getOpenClDefaultContext();    
        // Read the program source
        std::ifstream sourceFile("matcher_kernel_opencl.cl");
        std::string sourceCode( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));

        // Make program from the source code
        program = cl::Program(context, source);

        // Build the program for the devices
        program.build(getOpenClDevices());


        // Make kernel
        FindMatchKernel = cl::Kernel (program, "FindMatchBatchKernel");
        FindMatchWithoutBufferKernel = cl::Kernel (program, "FindMatchBatchKernelWithoutBuffer");
        openClKernelInitialized  = true;
        FindMatchKernelCache = FindMatchKernel;
        FindMatchWithoutBufferKernelCache = FindMatchWithoutBufferKernel;
    }
    catch(cl::Error err) {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        logOpenClBuildError(program, err); 
    }

}

int MatcherOpenCl::FindMatchBatch(char* buffer, int bufferSize, int* matches_length, int* matches_offset, int* matchSize, bool isLast,int currentMatchCount , int currentBatch) {
	
    int bufferSizeAdjusted = bufferSize - MAX_CODED;
    auto queue = getOpenClDefaultCommandQueue(currentBatch);
	if (isLast) {
		bufferSizeAdjusted += MAX_CODED;
    }
    int matchCount = bufferSizeAdjusted - WINDOW_SIZE;
	*matchSize = matchCount;
    
    int sizeToLaunch = matchCount;
    int blocks = sizeToLaunch / BLOCK_SIZE + ( sizeToLaunch % BLOCK_SIZE > 0?1 : 0);
    int threads = BLOCK_SIZE; 


     try{

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        cl::Buffer buffer_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(char) * bufferSize);
        cl::Buffer buffer_matches_length = cl::Buffer(context,  CL_MEM_READ_ONLY, sizeof(int) * matchCount);
        cl::Buffer buffer_matches_offset = cl::Buffer(context,  CL_MEM_READ_ONLY, sizeof(int) * matchCount);

        queue.enqueueWriteBuffer( buffer_buffer, CL_FALSE, 0, sizeof(char) * bufferSize, buffer );
        queue.finish();
        
        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        timeSpentOnMemoryHostToDevice += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        begin = std::chrono::steady_clock::now();
        // Set the kernel arguments
        FindMatchKernel.setArg( 0, buffer_buffer );
        FindMatchKernel.setArg( 1, bufferSize );
        FindMatchKernel.setArg( 2, buffer_matches_length);
        FindMatchKernel.setArg( 3, buffer_matches_offset);
        FindMatchKernel.setArg( 4, bufferSizeAdjusted );
        FindMatchKernel.setArg( 5, currentMatchCount );
        FindMatchKernel.setArg( 6, isLast?1:0 );


        int size = sizeToLaunch;
        if(sizeToLaunch % BLOCK_SIZE != 0){
            size = sizeToLaunch + BLOCK_SIZE - (sizeToLaunch % (BLOCK_SIZE) );
        }
        // Execute the kernel
        cl::NDRange global( size  );
        cl::NDRange local( BLOCK_SIZE  );
        
        #ifdef DEBUG
            std::cout 
                << "Size launch "
                << global[0]  << "."
                << local[0] << "." 
                << BLOCK_SIZE << "." 
                << sizeToLaunch << "." 
                << std::endl;
        #endif
        queue.enqueueNDRangeKernel( FindMatchKernel, cl::NullRange, global, local );
        queue.finish();

        end= std::chrono::steady_clock::now();
        timeSpentOnKernel += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        begin = std::chrono::steady_clock::now();
        queue.enqueueReadBuffer( buffer_matches_length, CL_TRUE, 0, sizeof(int) * matchCount, matches_length );
        queue.enqueueReadBuffer( buffer_matches_offset, CL_TRUE, 0, sizeof(int) * matchCount, matches_offset );
        queue.finish();
        end= std::chrono::steady_clock::now();
        timeSpentOnMemoryDeviceToHost += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    }
    catch(cl::Error err) {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        exit( EXIT_FAILURE );
    }
	return 0;
}



int MatcherOpenCl::FindMatchBatchUsingDeviceInput(cl::Buffer input, int inputSize, int offset, int *matches_length, int *matches_offset, int *matchSize, int deviceIndex){
    cl_int err = 0;
    auto queue = getOpenClDefaultCommandQueue(deviceIndex);
   
	
    int matchCount = inputSize;
    *matchSize = inputSize;
    
    int sizeToLaunch = matchCount;
    int blocks = sizeToLaunch / BLOCK_SIZE + ( sizeToLaunch % BLOCK_SIZE > 0?1 : 0);
    int threads = BLOCK_SIZE; 


     try{
        if(offset > 0){
            cl_buffer_region region = {offset,inputSize};
            // cl::BufferRegion region = ;
            input = input.createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION,  &region, &err);
        }
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        // cl::Buffer buffer_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(char) * bufferSize);
        cl::Buffer buffer_matches_length = cl::Buffer(context,  CL_MEM_READ_ONLY, sizeof(int) * matchCount);
        cl::Buffer buffer_matches_offset = cl::Buffer(context,  CL_MEM_READ_ONLY, sizeof(int) * matchCount);

        // queue.enqueueWriteBuffer( buffer_buffer, CL_FALSE, 0, sizeof(char) * bufferSize, buffer );
        // queue.finish();
        
        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        timeSpentOnMemoryHostToDevice += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        begin = std::chrono::steady_clock::now();
        // Set the kernel arguments
        FindMatchWithoutBufferKernel.setArg( 0, input );
        FindMatchWithoutBufferKernel.setArg( 1, inputSize );
        FindMatchWithoutBufferKernel.setArg( 2, buffer_matches_length);
        FindMatchWithoutBufferKernel.setArg( 3, buffer_matches_offset);


        int size = sizeToLaunch;
        if(sizeToLaunch % BLOCK_SIZE != 0){
            size = sizeToLaunch + BLOCK_SIZE - (sizeToLaunch % (BLOCK_SIZE) );
        }
        // Execute the kernel
        cl::NDRange global( size  );
        cl::NDRange local( BLOCK_SIZE  );
        
        #ifdef DEBUG
            std::cout 
                << "Size launch "
                << global[0]  << "."
                << local[0] << "." 
                << BLOCK_SIZE << "." 
                << sizeToLaunch << "." 
                << std::endl;
        #endif
        queue.enqueueNDRangeKernel( FindMatchWithoutBufferKernel, cl::NullRange, global, local );
        queue.finish();

        end= std::chrono::steady_clock::now();
        timeSpentOnKernel += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        begin = std::chrono::steady_clock::now();
        queue.enqueueReadBuffer( buffer_matches_length, CL_TRUE, 0, sizeof(int) * matchCount, matches_length );
        queue.enqueueReadBuffer( buffer_matches_offset, CL_TRUE, 0, sizeof(int) * matchCount, matches_offset );
        queue.finish();
        end= std::chrono::steady_clock::now();
        timeSpentOnMemoryDeviceToHost += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    }
    catch(cl::Error err) {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        exit( EXIT_FAILURE );
    }
	return 0;
}