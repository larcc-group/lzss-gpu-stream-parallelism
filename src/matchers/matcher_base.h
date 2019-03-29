#pragma once
#include <iostream>
#include <vector>
#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.cpp>
#else
#include <CL/cl.hpp>
#endif
class MatcherBase
{
  public:
    float timeSpentOnMemoryHostToDevice;
    float timeSpentOnMemoryDeviceToHost;
    float timeSpentOnKernel;

    virtual int Init()
    {
        timeSpentOnMemoryHostToDevice = 0;
        timeSpentOnMemoryDeviceToHost = 0;
        timeSpentOnKernel = 0;

        return 0;
    };
    virtual int FindMatchBatchUsingDeviceInput(unsigned char *input, int inputSize, int *matches_length, int *matches_offset, int *matchSize, int deviceIndex)
    {
        printf("Operation not supported");
        exit(ENOTSUP);
    };

    virtual int FindMatchBatchUsingDeviceInput(cl::Buffer input, int inputSize, int offset, int *matches_length, int *matches_offset, int *matchSize)
    {
        printf("Operation not supported cl");
        exit(ENOTSUP);
    };
    virtual int FindMatchBatch(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int *matchSize, bool isLast, int currentMatchCount, int currentBatch)
    {
        return -1;
    };
};

class MatcherSequential : public MatcherBase
{
  public:
    virtual int Init();
    virtual int FindMatchBatch(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int *matchSize, bool isLast, int currentMatchCount, int currentBatch);
   
};
class MatcherCuda : public MatcherBase
{
  public:
    virtual int Init();
    virtual int FindMatchBatch(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int *matchSize, bool isLast, int currentMatchCount, int currentBatch);
    //Find all at once
    virtual int FindMatchBatchUsingDeviceInput(unsigned char *input, int inputSize, int *matches_length, int *matches_offset, int *matchSize, int deviceIndex);
};
class MatcherOpenAcc : public MatcherBase
{
  public:
    virtual int Init();
    virtual int FindMatchBatch(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int *matchSize, bool isLast, int currentMatchCount, int currentBatch);
};
class MatcherOpenCl : public MatcherBase
{
  public:
    virtual int Init();
    virtual int FindMatchBatch(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int *matchSize, bool isLast, int currentMatchCount, int currentBatch);
    virtual int FindMatchBatchUsingDeviceInput(cl::Buffer input, int inputSize, int offsetLength, int *matches_length, int *matches_offset, int *matchSize,int deviceIndex);
   
  private:
    cl::Kernel FindMatchKernel;
    cl::Kernel FindMatchWithoutBufferKernel;
    cl::Context context;
};
