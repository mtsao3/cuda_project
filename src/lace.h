// CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <device_functions.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/opengl_interop.hpp>

#define THREADS_PER_WARP 32
#define BLOCK_SIZE 16
__constant__ float MIN_STD = 0.1f;

void local_contrast_enhancement(cv::gpu::GpuMat& src,
	cv::gpu::GpuMat& dst,
	float sigma_increase);
