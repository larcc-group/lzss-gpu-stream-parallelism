#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include "gpu_util.h"

std::vector<int> deviceIds = {0};
std::vector<cl::Device> devices;
cl::Context context;
std::vector<cl::CommandQueue> queue;

bool initialized = false;
void setDeviceIds(std::vector<int> item)
{
    deviceIds = item;
}
std::vector<int> getDeviceIds()
{
    return deviceIds;
}
void initOpenCl()
{

    try
    {
        // printf("initOpenCL\n");

        unsigned int platform_id = 0;

        // Query for platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU, &devices); // Select the platform.
        std::vector<cl::Device> devicesUsed = {};
        
        for (size_t i = 0; i < deviceIds.size(); i++)
        {
            if (devices.size() - 1 < deviceIds[i])
            {
                printf("Device %i not found\n",deviceIds[i]);
                exit(-1);
            }
            devicesUsed.push_back(devices[deviceIds[i]]); // Select the device.
        }
        
        // context = cl::Context(devicesUsed);
        context = cl::Context(devices);
        
        for (size_t i = 0; i < deviceIds.size(); i++)
        {
            printf("Using device %i \n",deviceIds[i]);
            if (devices.size() - 1 < deviceIds[i])
            {
                printf("Device %i not found\n",deviceIds[i]);
                exit(-1);
            }
            queue.push_back(cl::CommandQueue(context, devices[deviceIds[i]])); // Select the device.
        }
    }
    catch (cl::Error err)
    {
        printf( "Error: %i(%s) ", err.what(), err.err());
        exit(EXIT_FAILURE);
    }
    initialized = true;
}
cl::Context getOpenClDefaultContext()
{
    if (!initialized)
    {
        initOpenCl();
    }
    return context;
}
std::vector<cl::Device> getOpenClDevices()
{
    if (!initialized)
    {
        initOpenCl();
    }
    return devices;
}
cl::CommandQueue getOpenClDefaultCommandQueue(int index)
{
    if (!initialized)
    {
        initOpenCl();
    }
    return queue[index % queue.size()];
}

int logOpenClBuildError(cl::Program program, cl::Error err)
{
    if (err.err() == CL_BUILD_PROGRAM_FAILURE)
    {
        for (cl::Device dev : devices)
        {
            // Check the build status
            cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
            if (status != CL_BUILD_ERROR)
                continue;

            // Get the build log
            std::string name = dev.getInfo<CL_DEVICE_NAME>();
            std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
            // std::cerr << "Build log for " << name << ":" << std::endl
            //             << buildlog << std::endl;

            printf("Error in build for %s: %s\n", name.c_str(), buildlog.c_str());
        }
        exit(EXIT_FAILURE);
    }
    return 0;
}