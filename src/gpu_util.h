

#pragma once 

#include <CL/cl.hpp>
#include <vector>

void setDeviceIds(std::vector<int> deviceIds);
std::vector<int> getDeviceIds();
std::vector<cl::Device> getOpenClDevices();
cl::Context getOpenClDefaultContext();
cl::CommandQueue getOpenClDefaultCommandQueue(int index);


int logOpenClBuildError(cl::Program program, cl::Error err);