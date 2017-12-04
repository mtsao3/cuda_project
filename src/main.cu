// CUDA
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <stdio.h>
#include <iostream>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/opengl_interop.hpp>

#define THREADS_PER_WARP 32
#define BLOCK_SIZE 32

template<class T>
__device__ void changeStatistics(const cv::gpu::PtrStepSz<T> src,
									cv::gpu::PtrStepSz<float> dst,
									const float *means, 
									const float *stds, 
									const float sigma)
{
	const int tid_x = threadIdx.x;
	const int tid_y = threadIdx.y;
	const int x = (blockIdx.x * blockDim.x) + tid_x;
	const int y = (blockIdx.y * blockDim.y) + tid_y;
	const int rows = src.rows;
	const int cols = src.cols;

	if (y < rows && x < cols) {
		dst.ptr(y)[x] = (((float(src.ptr(y)[x]) - means[0])/stds[0]) * (sigma + stds[0])) + means[0];
		// Handle cases for when image is larger than bounds
		if (dst.ptr(y)[x] > 255)
			dst.ptr(y)[x] = 255.f;
		if (dst.ptr(y)[x] < 0)
			dst.ptr(y)[x] = 0.f;
	}
	__syncthreads();
}

template<class T>
__device__ void minusMeanSquared(const cv::gpu::PtrStepSz<T> src, cv::gpu::PtrStepSz<float> dst, float *means)
{
	const int tid_x = threadIdx.x;
	const int tid_y = threadIdx.y;
	const int x = (blockIdx.x * blockDim.x) + tid_x;
	const int y = (blockIdx.y * blockDim.y) + tid_y;
	const int rows = src.rows;
	const int cols = src.cols;

	if (y < rows && x < cols) {
		dst.ptr(y)[x] = powf((float(src.ptr(y)[x]) - means[0]), 2.f);
	}
	__syncthreads();
}

template<class T>
__device__ void calcBlockMean(const cv::gpu::PtrStepSz<T> src, float *dst) {
	/*
	Calculate the mean per block
	*/
	const int tid_x = threadIdx.x;
	const int tid_y = threadIdx.y;
	const int x_bound = (blockDim.x / 2);
	const int block_size = blockDim.x*blockDim.y;
	const int x = (blockIdx.x * blockDim.x) + tid_x;
	const int y = (blockIdx.y * blockDim.y) + tid_y;
	const int block_ind = (x_bound * threadIdx.y) + tid_x;
	const int block_num = (blockIdx.y * gridDim.x) + blockIdx.x;
	const int tid = (tid_y * blockDim.x) + tid_x;
	const int rows = src.rows;
	const int cols = src.cols;

	// Load global data into shared memory while completing first add for speed
	//		and make sure we are within image boundaries and block boundaries
	if (y < rows && x < cols && tid_x < x_bound) {
		dst[block_ind] = float(src.ptr(y)[x]) + float(src.ptr(y)[x + x_bound]);
	}
	__syncthreads();

	// Finish reduction
	for (int s = block_size / 4; s > 0; s >>= 1) {
		if (tid < s) {
			dst[tid] += dst[tid + s];
		}
		__syncthreads();
	}

	// Calculate mean
	if (tid == 0) {
		dst[0] /= float(block_size);
	}
	__syncthreads();
}

__global__ void lace(const cv::gpu::PtrStepSz<unsigned char> src,
					cv::gpu::PtrStepSz<float> dst,
					const double sigma)
{
	const int tid_x = threadIdx.x;
	const int tid_y = threadIdx.y;
	const int tid = (tid_y * blockDim.x) + tid_x;
	const int block_num = (blockIdx.y * gridDim.x) + blockIdx.x;

	// Create shared memory for faster calculations
	__shared__ float means[BLOCK_SIZE*BLOCK_SIZE / 2];
	__shared__ float stds[BLOCK_SIZE*BLOCK_SIZE / 2];

	// Calculate mean per block
	calcBlockMean<unsigned char>(src, means);

	// Calculate standard deviation per block
	minusMeanSquared(src, dst, means);
	calcBlockMean<float>(dst, stds);
	if (tid == 0) {
		stds[0] = sqrt(stds[0]);
	}
	__syncthreads();

	// Adjust statistics of image locally
	changeStatistics<unsigned char>(src, dst, means, stds, sigma);
}

void local_contrast_enhancement(cv::gpu::GpuMat& src,
								cv::gpu::GpuMat& dst,
								float sigma_increase) 
{
	// Get intensity image to enhance contrast
	cv::gpu::cvtColor(src, dst, CV_BGR2HSV);
	std::vector<cv::gpu::GpuMat> channels;
	cv::gpu::split(dst, channels);
	// Third channel of image is the intensity image
	cv::gpu::GpuMat intensity = channels[2];

	// Determine number of threads per block
	//	Make sure it is a divisible by threads per warp
	int rows = intensity.rows;
	int cols = intensity.cols;
	int grid_x = cols / BLOCK_SIZE + 0.99;
	int grid_y = rows / BLOCK_SIZE + 0.99;

	// Allocate room for destination matrix
	cv::gpu::GpuMat laced(rows, cols, CV_32FC1);

	// Call LACE Kernel
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim(grid_x, grid_y);
	lace << <gridDim, blockDim >> > (intensity, laced, sigma_increase);

	// Convert the new luminance band to an 8 bit image and merge the channels again to be converted to BGR
	laced.convertTo(intensity, CV_8UC1);
	cv::gpu::merge(channels, dst);
	cv::gpu::cvtColor(dst, dst, CV_HSV2BGR);
}

int main()
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	// Read image
	cv::Mat image = cv::imread("../test.jpg");

	// Resize image so I don't have to worry about edge cases
	int subtract_rows = image.rows%BLOCK_SIZE;
	int subtract_cols = image.cols%BLOCK_SIZE;
	cv::resize(image, image, cv::Size(image.cols - subtract_cols, image.rows - subtract_rows));

	// Place image on GPU
	cv::gpu::GpuMat gpu_image;
	gpu_image.upload(image);

	// Create a handle for output
	cv::gpu::GpuMat laced;
	
	// Apply contrast enhancement
	local_contrast_enhancement(gpu_image, laced, 6.5f);

	// Send data back from GPU to host
	cv::Mat laced_host;
	laced.download(laced_host);

	// Display images
	cv::imwrite("../test_image_laced.jpg", laced_host);
	cv::imshow("laced", laced_host);
	cv::imshow("original", image);
	cv::waitKey(0);

	// Release resources
	cudaDeviceReset();
    return 0;
}