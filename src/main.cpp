// CUDA
#include <stdio.h>
#include <iostream>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/opengl_interop.hpp>

// LACE
#include "lace.h"

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
	local_contrast_enhancement(gpu_image, laced, 8.0f);

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