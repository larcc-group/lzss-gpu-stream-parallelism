#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <chrono>
#include <iostream>
#include "lzlocal.h"
#include "bitfile.h"
#include "matcher_base.h"

#define KERNEL_PART_SIZE 1024
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

__global__ void FindMatchBatchKernel(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int bufferSizeAdjusted, int currentMatchCount, bool isLast)
{
    int startAt = (blockIdx.x * blockDim.x + threadIdx.x) * KERNEL_PART_SIZE;
    
    char bufferCache[WINDOW_SIZE + KERNEL_PART_SIZE];
    
    for( int k = 0; k < WINDOW_SIZE + KERNEL_PART_SIZE && startAt + k < bufferSize;k++){
        bufferCache[k] = buffer[startAt+k];
    }
    for(int k = 0; k < KERNEL_PART_SIZE; k++){
        int idX = startAt + k;
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
        for (int j = 0; j < MAX_CODED && i + j < bufferSizeAdjusted; j++)
        {
            current[j] = buffer[i + j];
        }
        int j = 0;
        while (1)
        {
            if (current[0] == bufferCache[k + Wrap((currentOffset), WINDOW_SIZE)])
            {
                /* we matched one. how many more match? */
                j = 1;

                while (
                    current[j] == bufferCache[k + Wrap((currentOffset + j), WINDOW_SIZE)] && (!isLast ||
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
        // matches_length[idX] = -1;
        if(length > 2 && k > 18){
            k += length -1;
        }
    }
}
int MatcherCuda::Init()
{
    MatcherBase::Init();
    return 0;
}
int MatcherCuda::FindMatchBatch(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int *matchSize, bool isLast, int currentMatchCount)
{
    int bufferSizeAdjusted = bufferSize - MAX_CODED;
    if (isLast)
    {
        bufferSizeAdjusted += MAX_CODED;
    }
    int matchCount = bufferSizeAdjusted - WINDOW_SIZE;
    *matchSize = matchCount;

    int sizeToLaunch = matchCount / KERNEL_PART_SIZE;
    int blocks = sizeToLaunch / BLOCK_SIZE + (sizeToLaunch % BLOCK_SIZE > 0 ? 1 : 0);
    int threads = BLOCK_SIZE;
    // printf("SizeToLaunch %i\n",sizeToLaunch);
    // printf("Blocks %i\n",blocks);
    // printf("Threads %i\n",threads);

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
