// =============================================================================
// cuda_utils.cuh - CUDA helpers: error checking, GPU timer, device information
// CSC14120 Parallel Programming - Final Project - Group 09
// Authors: Le Nhat Khoi (23127004), Nguyen Hoang Gia Bao (23127157),
//          Nguyen Hai Dang (23127165)
// Adapted from the CHECK macro and the GpuTimer struct of the course demo code
// (CSC14120 lecture demos, e.g. 01-AddVector.cu).
//
// V0 does not launch any kernel; these helpers are used by main.cu (GPU name)
// and will be used by V1+ (CUDA events around each stage).
// =============================================================================
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                          \
{                                                                            \
	const cudaError_t error = call;                                          \
	if (error != cudaSuccess)                                                \
	{                                                                        \
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);               \
		fprintf(stderr, "code: %d, reason: %s\n", error,                     \
				cudaGetErrorString(error));                                  \
		exit(EXIT_FAILURE);                                                  \
	}                                                                        \
}

// Timer based on CUDA events (GPU timing rule of the project)
struct GpuTimer
{
	cudaEvent_t start, stop;
	GpuTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
	~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
	void Start() { cudaEventRecord(start, 0); cudaEventSynchronize(start); }
	void Stop()  { cudaEventRecord(stop, 0); }
	float Elapsed()
	{
		float ms;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
		return ms;
	}
};

// Name of GPU 0, or "no CUDA device" (so that V0 can also run without a GPU)
inline void getGpuName(char * name, size_t size)
{
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
	{
		cudaGetLastError(); // clear the error
		snprintf(name, size, "no CUDA device");
		return;
	}
	cudaDeviceProp prop;
	CHECK(cudaGetDeviceProperties(&prop, 0));
	snprintf(name, size, "%s (compute capability %d.%d)", prop.name, prop.major, prop.minor);
}
