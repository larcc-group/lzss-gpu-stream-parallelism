#include "lzss.h"
#include "lzlocal.h"
#define TEST_SIZE 350000
#include "gpu_util.h"

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.cpp>
#else
#include <CL/cl.hpp>
#endif

#define EXIT_TRACE(errorCode, x)            \
    printf("Error %i: %s\n", errorCode, x); \
    exit(errorCode);

int testCpu();
int testGpu();
int testGpuExistingMem();
int testGpuExistingMemOpenCl();
int main(int argc, int **args)
{
    printf("Testing CPU...\n");
    if (testCpu() != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    printf("Testing GPU...\n");
    if (testGpu() != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    printf("Testing GPU with existing memory...\n");
    if (testGpuExistingMem() != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }


    printf("Testing OpenCL GPU with existing memory...\n");
    if (testGpuExistingMemOpenCl() != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int testCpu()
{

    unsigned char input[TEST_SIZE];
    const int outputSize = TEST_SIZE + (TEST_SIZE / 8); //Max size it could possibly be
    unsigned char output[outputSize];

    //Same seed for debugging porpuses
    srand(0);
    for (int i = 0; i < TEST_SIZE; i++)
    {
        input[i] = (unsigned char)(rand() % 10);
    }

    int compressedSize;
    int error = LzssEncodeMemory(input, TEST_SIZE, output, outputSize, &compressedSize);
    if (error != 0)
    {
        EXIT_TRACE(error, "Error encoding memory");
    }
    printf("Input Size: %i\n", TEST_SIZE);
    printf("CompressedSize: %i\n", compressedSize);

    unsigned char decompressed[TEST_SIZE];

    int decompressedSize;
    error = LzssDecodeMemory(output, compressedSize, decompressed, TEST_SIZE, &decompressedSize);
    if (error != 0)
    {
        EXIT_TRACE(error, "Error decoding memory");
    }
    if (decompressedSize != TEST_SIZE)
    {
        printf("Expected %i but got %i\n", TEST_SIZE, decompressedSize);
        EXIT_TRACE(-1, "Different sizes after decompression");
    }
    int differencesFound = 0;
    for (int i = 0; i < TEST_SIZE; i++)
    {
        if (input[i] != decompressed[i])
        {
            differencesFound++;
            
            printf("Differencess found at %i: %c <> %c ", i, input[i], decompressed[i]);
        }
    }
    if (differencesFound > 0)
    {
        printf("TEST FAILED\n");
        return EXIT_FAILURE;
    }
    else
    {
        printf("TEST PASSED\n");
        return EXIT_SUCCESS;
    }
}

int testGpu()
{

    unsigned char input[TEST_SIZE];
    const int outputSize = TEST_SIZE + (TEST_SIZE / 8); //Max size it could possibly be
    unsigned char output[outputSize];

    //Same seed for debugging porpuses
    srand(0);
    for (int i = 0; i < TEST_SIZE; i++)
    {
        input[i] = (unsigned char)(rand() % 10);
    }

    int compressedSize;
    int error = LzssEncodeMemoryGpu(input, TEST_SIZE, output, outputSize, &compressedSize);
    if (error != 0)
    {
        EXIT_TRACE(error, "Error encoding memory");
    }
    printf("Input Size: %i\n", TEST_SIZE);
    printf("CompressedSize: %i\n", compressedSize);

    unsigned char decompressed[TEST_SIZE];

    int decompressedSize;
    error = LzssDecodeMemory(output, compressedSize, decompressed, TEST_SIZE, &decompressedSize);
    if (error != 0)
    {
        EXIT_TRACE(error, "Error decoding memory");
    }
    if (decompressedSize != TEST_SIZE)
    {
        printf("Expected %i but got %i\n", TEST_SIZE, decompressedSize);
        EXIT_TRACE(-1, "Different sizes after decompression");
    }
    int differencesFound = 0;
    for (int i = 0; i < TEST_SIZE; i++)
    {
        if (input[i] != decompressed[i])
        {
            differencesFound++;
            
            printf("Differencess found at %i: %c <> %c ", i, input[i], decompressed[i]);
        }
    }
    if (differencesFound > 0)
    {
        printf("TEST FAILED\n");
        return EXIT_FAILURE;
    }
    else
    {
        printf("TEST PASSED\n");
        return EXIT_SUCCESS;
    }
}


int testGpuExistingMem()
{

    unsigned char input[TEST_SIZE];
    const int outputSize = TEST_SIZE + (TEST_SIZE / 8); //Max size it could possibly be
    unsigned char output[outputSize];

    //Same seed for debugging porpuses
    srand(0);
    for (int i = 0; i < TEST_SIZE; i++)
    {
        input[i] = (unsigned char)(rand() % 10);
    }

    int compressedSize;
    unsigned char * d_input;
    cudaMalloc((void **)&d_input,sizeof(unsigned char) * TEST_SIZE);
    cudaMemcpy(d_input, input , sizeof(unsigned char) * TEST_SIZE, cudaMemcpyHostToDevice);
    int error = LzssEncodeMemoryGpu(input, d_input, TEST_SIZE, output, outputSize, &compressedSize);
    if (error != 0)
    {
        EXIT_TRACE(error, "Error encoding memory");
    }
    printf("Input Size: %i\n", TEST_SIZE);
    printf("CompressedSize: %i\n", compressedSize);

    unsigned char decompressed[TEST_SIZE];

    int decompressedSize;
    error = LzssDecodeMemory(output, compressedSize, decompressed, TEST_SIZE, &decompressedSize);
    if (error != 0)
    {
        EXIT_TRACE(error, "Error decoding memory");
    }
    if (decompressedSize != TEST_SIZE)
    {
        printf("Expected %i but got %i\n", TEST_SIZE, decompressedSize);
        EXIT_TRACE(-1, "Different sizes after decompression");
    }
    int differencesFound = 0;
    for (int i = 0; i < TEST_SIZE; i++)
    {
        if (input[i] != decompressed[i])
        {
            differencesFound++;
            printf("Differencess found at %i: %c <> %c ", i, input[i], decompressed[i]);
        }
    }
    if (differencesFound > 0)
    {
        printf("TEST FAILED\n");
        return EXIT_FAILURE;
    }
    else
    {
        printf("TEST PASSED\n");
        return EXIT_SUCCESS;
    }
}



int testGpuExistingMemOpenCl()
{

    unsigned char input[TEST_SIZE];
    const int outputSize = TEST_SIZE + (TEST_SIZE / 8); //Max size it could possibly be
    unsigned char output[outputSize];

    //Same seed for debugging porpuses
    srand(0);
    for (int i = 0; i < TEST_SIZE; i++)
    {
        input[i] = (unsigned char)(rand() % 10);
    }

    int compressedSize;


    cl::Buffer d_input_cl(getOpenClDefaultContext(),  CL_MEM_READ_ONLY, sizeof(unsigned char) * TEST_SIZE);
   
    auto queue =  getOpenClDefaultCommandQueue(0);
    queue.enqueueWriteBuffer( d_input_cl, CL_TRUE, 0,sizeof(unsigned char) * TEST_SIZE, input );
    queue.finish();
   int error = LzssEncodeMemoryGpu(input, d_input_cl, TEST_SIZE, output, outputSize, &compressedSize);
    if (error != 0)
    {
        EXIT_TRACE(error, "Error encoding memory");
    }
    printf("Input Size: %i\n", TEST_SIZE);
    printf("CompressedSize: %i\n", compressedSize);

    unsigned char decompressed[TEST_SIZE];

    int decompressedSize;
    error = LzssDecodeMemory(output, compressedSize, decompressed, TEST_SIZE, &decompressedSize);
    if (error != 0)
    {
        EXIT_TRACE(error, "Error decoding memory");
    }
    if (decompressedSize != TEST_SIZE)
    {
        printf("Expected %i but got %i\n", TEST_SIZE, decompressedSize);
        EXIT_TRACE(-1, "Different sizes after decompression");
    }
    int differencesFound = 0;
    for (int i = 0; i < TEST_SIZE; i++)
    {
        if (input[i] != decompressed[i])
        {
            differencesFound++;
            printf("Differencess found at %i: %c <> %c ", i, input[i], decompressed[i]);
        }
    }
    if (differencesFound > 0)
    {
        printf("TEST FAILED\n");
        return EXIT_FAILURE;
    }
    else
    {
        printf("TEST PASSED\n");
        return EXIT_SUCCESS;
    }
}