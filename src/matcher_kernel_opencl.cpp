#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <chrono>
#include "lzlocal.h"
#include "bitfile.h"
#include "matcher_kernel_opencl.h"

int timeSpentOnMemoryHostToDevice = 0;
int timeSpentOnMemoryDeviceToHost = 0;
int timeSpentOnKernel = 0;

#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.cpp>
#else
#include <CL/cl.hpp>
#endif

cl::CommandQueue queue;
cl::Kernel findmatch_kernel;
cl::Context context;
void InitGpuOpenCL(){
    cl::Program program;
    std::vector<cl::Device> devices;
    try{

        unsigned int platform_id = 0, device_id = 0;

        // Query for platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // Get a list of devices on this platform
        platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU, &devices); // Select the platform.

        // Create a context
        context = cl::Context(devices);

        // Create a command queue
        queue = cl::CommandQueue( context, devices[device_id] );   // Select the device.

        // Read the program source
        std::ifstream sourceFile("../matcher_kernel_opencl.cl");
        std::string sourceCode( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));

        // Make program from the source code
        program = cl::Program(context, source);

        // Build the program for the devices
        program.build(devices);


        // Make kernel
        findmatch_kernel = cl::Kernel (program, "kernel_sobel");
        
    }
    catch(cl::Error err) {
        
        if (err.err() == CL_BUILD_PROGRAM_FAILURE){
            for (cl::Device dev : devices){
                // Check the build status
                cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                if (status != CL_BUILD_ERROR)
                    continue;

                // Get the build log
                std::string name     = dev.getInfo<CL_DEVICE_NAME>();
                std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                std::cerr << "Build log for " << name << ":" << std::endl
                            << buildlog << std::endl;
            }
        } else {
            std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        }
        
    }

}

int FindMatchBatchWrapperKernelOpenCL(char* buffer, int bufferSize, int* matches_length, int* matches_offset, int* matchSize, bool isLast,int currentMatchCount ) {
	int bufferSizeAdjusted = bufferSize - MAX_CODED;
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
        cl::Buffer buffer_matches_length = cl::Buffer(context,  CL_MEM_READ_ONLY, sizeof(char) * bufferSize);
        cl::Buffer buffer_matches_offset = cl::Buffer(context,  CL_MEM_READ_ONLY, sizeof(char) * bufferSize);

        queue.enqueueWriteBuffer( buffer_buffer, CL_FALSE, 0, sizeof(char) * bufferSize, buffer );
        
        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        timeSpentOnMemoryHostToDevice += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        begin = std::chrono::steady_clock::now();
        // Set the kernel arguments
        findmatch_kernel.setArg( 0, buffer_buffer );
        findmatch_kernel.setArg( 1, bufferSize );
        findmatch_kernel.setArg( 2, buffer_matches_length);
        findmatch_kernel.setArg( 3, buffer_matches_offset);
        findmatch_kernel.setArg( 4, bufferSizeAdjusted );
        findmatch_kernel.setArg( 5, currentMatchCount );
        findmatch_kernel.setArg( 6, isLast );


        int size = sizeToLaunch;
        size = size + (size % (BLOCK_SIZE) );
        // Execute the kernel
        cl::NDRange global( size  );
        cl::NDRange local( BLOCK_SIZE  );
        
        #ifdef DEBUG
            std::cout 
                << "Size launch "
                << global[0]  << "."
                << local[0] << "." 
                << std::endl;
        #endif
        queue.enqueueNDRangeKernel( findmatch_kernel, cl::NullRange, global, local );

        end= std::chrono::steady_clock::now();
        timeSpentOnKernel += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        begin = std::chrono::steady_clock::now();
        queue.enqueueReadBuffer( buffer_matches_length, CL_TRUE, 0, sizeof(char) * bufferSize, matches_length );
        queue.enqueueReadBuffer( buffer_matches_offset, CL_TRUE, 0, sizeof(char) * bufferSize, matches_offset );
        
        end= std::chrono::steady_clock::now();
        timeSpentOnMemoryDeviceToHost += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    }
    catch(cl::Error err) {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        exit( EXIT_FAILURE );
    }
	return 0;
}
