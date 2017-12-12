#include "lace.h"

template<class T>
__device__ void changeStatistics(const cv::gpu::PtrStepSz<T> src,
									cv::gpu::PtrStepSz<float> dst,
									const float *means, 
									const float *stds, 
									const float sigma,
									const int tid_x,
									const int tid_y,
									const int x,
									const int y,
									const int rows,
									const int cols)
{
	
	// Set the desired value of the pixel based of the statistics of the region. Also handle edge cases where the standard deviation is close to zero
	if (y < rows && x < cols) {
		float sigma_target;
		// If the standard deviation is too low, don't enhance the noise
		if (stds[0] < MIN_STD) {
			sigma_target = 1.f;
		}
		else {
			sigma_target = sigma / stds[0];
		}

		dst.ptr(y)[x] = (float(src.ptr(y)[x]) - means[0]) * sigma_target + means[0];

		// Handle cases for when image is larger than bounds
		if (dst.ptr(y)[x] > 255)
			dst.ptr(y)[x] = 255.f;
		if (dst.ptr(y)[x] < 0)
			dst.ptr(y)[x] = 0.f;
	}
	__syncthreads();
}

template<class T>
__device__ void calcBlockMean(const cv::gpu::PtrStepSz<T> src,
								float *dst,
								const int tid_x,
								const int tid_y,
								const int x,
								const int y,
								const int block_size,
								const int block_num,
								const int rows,
								const int cols) {
	/*
	Calculate the mean per block efficently 
	*/
	const int x_bound = (blockDim.x / 2);
	const int block_ind = (x_bound * threadIdx.y) + tid_x;
	const int tid = (tid_y * blockDim.x) + tid_x;

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
	// Declare constants to be used in kernel
	const int tid_x = threadIdx.x;
	const int tid_y = threadIdx.y;
	const int tid = (tid_y * blockDim.x) + tid_x;
	const int block_num = (blockIdx.y * gridDim.x) + blockIdx.x;
	const int x = (blockIdx.x * blockDim.x) + tid_x;
	const int y = (blockIdx.y * blockDim.y) + tid_y;
	const int block_size = blockDim.x*blockDim.y;
	const int rows = src.rows;
	const int cols = src.cols;
	

	// Create shared memory for faster calculations
	__shared__ float means_shared[BLOCK_SIZE*BLOCK_SIZE / 2];
	__shared__ float stds_shared[BLOCK_SIZE*BLOCK_SIZE / 2];

	// Calculate mean per block
	calcBlockMean<unsigned char>(src, means_shared, tid_x, tid_y, x, y, block_size, block_num, rows, cols);

	// Calculate standard deviation per block
	// Subtract mean from pixel and square it for standard deviation calculation
	if (y < rows && x < cols) {
		dst.ptr(y)[x] = powf((float(src.ptr(y)[x]) - means_shared[0]), 2.f);
	}
	__syncthreads();
	// Average the squared values
	calcBlockMean<float>(dst, stds_shared, tid_x, tid_y, x, y, block_size, block_num, rows, cols);
	// Complete standard deviation calculation
	if (tid == 0) {
		stds_shared[0] = sqrt(stds_shared[0]);
	}
	__syncthreads();

	// Adjust statistics of image locally
	changeStatistics<unsigned char>(src, dst, means_shared, stds_shared, sigma, tid_x, tid_y, x, y, rows, cols);
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
	cv::gpu::GpuMat means(rows, cols, CV_32FC1);

	// Call LACE Kernel
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim(grid_x, grid_y);
	lace << <gridDim, blockDim >> > (intensity, laced, sigma_increase);

	// Convert the new luminance band to an 8 bit image and merge the channels again to be converted to BGR
	laced.convertTo(intensity, CV_8UC1);
	cv::gpu::merge(channels, dst);
	cv::gpu::cvtColor(dst, dst, CV_HSV2BGR);
}
