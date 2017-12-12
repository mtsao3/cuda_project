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

#define ITERATIONS 10

// Performance measure convenience functions
void start_measure(cudaEvent_t* start, cudaEvent_t* stop) {
	cudaEventCreate(start, 0);
	cudaEventCreate(stop, 0);
	cudaEventRecord(*start, 0);
}

void stop_measure(float &time, cudaEvent_t* start, cudaEvent_t* stop) {
	cudaEventRecord(*stop, 0);
	cudaEventSynchronize(*stop);
	cudaEventElapsedTime(&time, *start, *stop);
}

int main(int argc, char *argv[])
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	// Handle input
	char* image_name;
	if (argc > 1) {
		image_name = argv[1];
	} else {
		image_name = "../test.jpg";
	}

	// Read image
	cv::Mat image = cv::imread(image_name);

	// Resize image so I don't have to worry about edge cases
	int subtract_rows = image.rows%BLOCK_SIZE;
	int subtract_cols = image.cols%BLOCK_SIZE;
	cv::resize(image, image, cv::Size(image.cols - subtract_cols, image.rows - subtract_rows));
	std::cout << "Processing image of size " << image.rows - subtract_rows << "x" << image.cols - subtract_cols << std::endl;

	// Place image on GPU
	cv::gpu::GpuMat gpu_image;
	gpu_image.upload(image);

	// Create a handle for output
	cv::gpu::GpuMat laced;
	
	// Apply contrast enhancement and measure speed
	cudaEvent_t start, stop;
	float elapsed;
	start_measure(&start, &stop);
	for (int i = 0; i < ITERATIONS; i++) {
		local_contrast_enhancement(gpu_image, laced, 8.0f);
	}
	stop_measure(elapsed, &start, &stop);
	std::cout << "LACE with block size " << BLOCK_SIZE << " took " << elapsed/float(ITERATIONS) << " msecs" << std::endl;

	// Send data back from GPU to host
	cv::Mat laced_host;
	laced.download(laced_host);

	// Display images
	cv::imwrite("../output.jpg", laced_host);
	cv::imshow("laced", laced_host);
	cv::imshow("original", image);
	cv::waitKey(0);

	// Release resources
	cudaDeviceReset();
    return 0;
}