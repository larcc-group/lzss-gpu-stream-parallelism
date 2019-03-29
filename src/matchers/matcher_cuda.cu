#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <chrono>
#include <iostream>
#include "lzlocal.h"
#include "lzss.h"
#include "bitfile.h"
#include "matcher_base.h"
#include "gpu_util.h"
#define checkError(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
__global__ void FillBuffer(char * buffer, unsigned char * input, int inputSize){
    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    if(idX < WINDOW_SIZE){
        buffer[idX] = ' ';
    }else if(idX - WINDOW_SIZE < inputSize){
        buffer[idX] = input[idX - WINDOW_SIZE];
    }
}
__global__ void FindMatchBatchKernel(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int bufferSizeAdjusted, int currentMatchCount, bool isLast)
{

    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int i = WINDOW_SIZE + idX;
    int beginSearch = idX;
    if (i >= bufferSizeAdjusted)
    {
        return;
    }

    int length = 0;
    int offset = 0;
    int windowHead = (currentMatchCount + idX) % WINDOW_SIZE;

    int currentOffset = 0;

    //Uncoded Lookahead optimization
    char current[MAX_CODED];
    //for (int j = 0; j < MAX_CODED && i + j < bufferSizeAdjusted; j++)
    for (int j = 0; j < MAX_CODED; j++)
    {
        current[j] = buffer[i + j];
    }
//    char* current = buffer + i;
    int j = 0;
    while (1)
    {
        if (current[0] == buffer[beginSearch + Wrap((currentOffset), WINDOW_SIZE)])
        {
            /* we matched one. how many more match? */
            j = 1;

            while (
              current[j] == buffer[beginSearch + Wrap((currentOffset + j), WINDOW_SIZE)] && (!isLast ||
                                                                                               (beginSearch + Wrap((currentOffset + j), WINDOW_SIZE) < bufferSizeAdjusted && i + j < bufferSizeAdjusted)))
            {

                if (j >= MAX_CODED)
                {
                    break;
                }
                j++;
            }

            if (j > length)
            {

                length = j;
                offset = Wrap((currentOffset + windowHead), WINDOW_SIZE);
            }
        }

        if (j >= MAX_CODED)
        {
            length = MAX_CODED;
            break;
        }

        currentOffset++;

        if (currentOffset == WINDOW_SIZE)
        {
            break;
        }
    }
    matches_offset[idX] = offset;
    matches_length[idX] = length;
}



__global__ void FindMatchBatchKernelWithoutBuffer(unsigned char *buffer, int bufferSize, int *matches_length, int *matches_offset)
{

    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idX;
    int beginSearch = idX - WINDOW_SIZE;
    if (i >= bufferSize)
    {
        return;
    }

    int length = 0;
    int offset = 0;
    int windowHead = ( idX) % WINDOW_SIZE;

    int currentOffset = 0;

    //Uncoded Lookahead optimization
    char current[MAX_CODED];
    //for (int j = 0; j < MAX_CODED && i + j < bufferSizeAdjusted; j++)
    for (int j = 0; j < MAX_CODED; j++)
    {
        current[j] = buffer[i + j];
    }

    //First WINDOW_SIZE bits will always be ' ', optimize begging where data really is
    if(beginSearch < -MAX_CODED){
        currentOffset = (beginSearch * -1) - MAX_CODED;
    }
//    char* current = buffer + i;
    int j = 0;
    while (1)
    {
        if (current[0] == (beginSearch + Wrap((currentOffset), WINDOW_SIZE) < 0? ' ': buffer[beginSearch + Wrap((currentOffset), WINDOW_SIZE)]))
        {
            /* we matched one. how many more match? */
            j = 1;

            while (
              current[j] == (beginSearch + Wrap((currentOffset + j), WINDOW_SIZE) < 0?' ':buffer[beginSearch + Wrap((currentOffset + j), WINDOW_SIZE)]) &&  
                beginSearch + Wrap((currentOffset + j), WINDOW_SIZE) < bufferSize && i + j < bufferSize)
            {

                if (j >= MAX_CODED)
                {
                    break;
                }
                j++;
            }

            if (j > length)
            {

                length = j;
                offset = Wrap((currentOffset + windowHead), WINDOW_SIZE);
            }
        }

        if (j >= MAX_CODED)
        {
            length = MAX_CODED;
            break;
        }
        
        currentOffset++;

        if (currentOffset == WINDOW_SIZE)
        {
            break;
        }
    }
    matches_offset[idX] = offset;
    matches_length[idX] = length;
}
int MatcherCuda::Init()
{
    MatcherBase::Init();
    return 0;
}
int MatcherCuda::FindMatchBatch(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int *matchSize, bool isLast, int currentMatchCount, int currentBatch)
{
    auto deviceIds = getDeviceIds();
    int deviceIdThread = deviceIds[currentBatch % deviceIds.size()];
    cudaSetDevice(deviceIdThread);
    int bufferSizeAdjusted = bufferSize - MAX_CODED;
    if (isLast)
    {
        bufferSizeAdjusted += MAX_CODED;
    }
    int matchCount = bufferSizeAdjusted - WINDOW_SIZE;
    *matchSize = matchCount;

    int sizeToLaunch = matchCount;
    int blocks = sizeToLaunch / BLOCK_SIZE + (sizeToLaunch % BLOCK_SIZE > 0 ? 1 : 0);
    int threads = BLOCK_SIZE;

    char *d_buffer;
    int *d_matches_length;
    int *d_matches_offset;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float miliseconds;

    cudaEventRecord(start, 0);
    //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    checkError(cudaMalloc((void **)&d_buffer, sizeof(char) * bufferSize));
    checkError(cudaMalloc((void **)&d_matches_length, sizeof(int) * matchCount));
    checkError(cudaMalloc((void **)&d_matches_offset, sizeof(int) * matchCount));

    checkError(cudaMemcpy(d_buffer, buffer, sizeof(char) * bufferSize, cudaMemcpyHostToDevice));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);

    timeSpentOnMemoryHostToDevice += miliseconds;
    //std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
    //timeSpentOnMemoryHostToDevice += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

    //begin = std::chrono::steady_clock::now();
    cudaEventRecord(start, 0);
    FindMatchBatchKernel<<<blocks, threads>>>(d_buffer, bufferSize, d_matches_length, d_matches_offset, bufferSizeAdjusted, currentMatchCount, isLast);
    checkError(cudaPeekAtLastError());
    //end= std::chrono::steady_clock::now();
    //timeSpentOnKernel += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&miliseconds, start, stop);
    timeSpentOnKernel += miliseconds;
    // cudaDeviceSynchronize();
    //begin = std::chrono::steady_clock::now();
    cudaEventRecord(start, 0);
    checkError(cudaMemcpy(matches_offset, d_matches_offset, sizeof(int) * matchCount, cudaMemcpyDeviceToHost));
    checkError(cudaMemcpy(matches_length, d_matches_length, sizeof(int) * matchCount, cudaMemcpyDeviceToHost));

    cudaFree(d_buffer);
    cudaFree(d_matches_length);
    cudaFree(d_matches_offset);

    //end= std::chrono::steady_clock::now();
    //timeSpentOnMemoryDeviceToHost += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&miliseconds, start, stop);
    timeSpentOnMemoryDeviceToHost += miliseconds;
    return 0;
}




int MatcherCuda::FindMatchBatchUsingDeviceInput(unsigned char *input, int inputSize, int *matches_length, int *matches_offset, int *matchSize, int deviceIndex)
{
    // auto deviceIds = getDeviceIds();
    // int deviceIdThread = deviceIds[deviceIndex];
    cudaSetDevice(deviceIndex);

    int bufferSizeAdjusted = inputSize + WINDOW_SIZE;

    int matchCount = inputSize;
    *matchSize = inputSize;

    
    // char *d_buffer;
    int *d_matches_length;
    int *d_matches_offset;

    int sizeToLaunch ;
    int blocks;
    int threads;
    

    // checkError(cudaMalloc((void **)&d_buffer, sizeof(char) * (inputSize + WINDOW_SIZE)));

    // int sizeToLaunch = bufferSizeAdjusted;
    // int blocks = sizeToLaunch / BLOCK_SIZE + (sizeToLaunch % BLOCK_SIZE > 0 ? 1 : 0);
    // int threads = BLOCK_SIZE;
    // FillBuffer<<<blocks,threads>>>(d_buffer,input,inputSize);
    
    checkError(cudaMalloc((void **)&d_matches_length, sizeof(int) * matchCount));
    checkError(cudaMalloc((void **)&d_matches_offset, sizeof(int) * matchCount));

    sizeToLaunch = matchCount;
    blocks = sizeToLaunch / BLOCK_SIZE + (sizeToLaunch % BLOCK_SIZE > 0 ? 1 : 0);
    threads = BLOCK_SIZE;

    FindMatchBatchKernelWithoutBuffer<<<blocks, threads>>>(input, inputSize, d_matches_length, d_matches_offset);
    // FindMatchBatchKernel<<<blocks, threads>>>(d_buffer, bufferSizeAdjusted, d_matches_length, d_matches_offset, bufferSizeAdjusted, 0, true);
    checkError(cudaPeekAtLastError());

    checkError(cudaMemcpy(matches_offset, d_matches_offset, sizeof(int) * matchCount, cudaMemcpyDeviceToHost));
    checkError(cudaMemcpy(matches_length, d_matches_length, sizeof(int) * matchCount, cudaMemcpyDeviceToHost));

    // cudaFree(d_buffer);
    cudaFree(d_matches_length);
    cudaFree(d_matches_offset);

    return 0;
}

