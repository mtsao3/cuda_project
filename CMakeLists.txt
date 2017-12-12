cmake_minimum_required(VERSION 2.8)

# Set the name of the output executable
set (my_target gpu_project)
project(${my_target})

# +-----------------------------+
# | CUDA
# +-----------------------------+
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
   message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
   message(STATUS "CUDA_INCLUDES ${CUDA_INCLUDE_DIRS}")
   message(STATUS "CUDA_CUDA_LIBRARY ${CUDA_CUDA_LIBRARY}")
endif ()
set(CUDA_NVCC_FLAGS -arch sm_50)
# +-----------------------------+
# | OpenCV
# +-----------------------------+
find_package(OpenCV REQUIRED )

include_directories(
   ${CUDA_INCLUDES}
   ${CUDA_INCLUDE_DIRS}
   src
)

# CUDA Target
cuda_add_executable(${my_target}
   src/main.cpp
   src/lace.cu
)

target_link_libraries(${my_target}
   ${OpenCV_LIBS}
   ${CUDA_CUDA_LIBRARIES}
   ${CUDA_LIBRARIES}
   ${CUDA_CUBLAS_LIBRARIES}    
   ${CUDA_CUFFT_LIBRARIES}
   ${CUDA_npp_LIBRARIES}
   ${CUDA_nppc_LIBRARIES}
   ${CUDA_npps_LIBRARIES}
   ${CUDA_nppi_LIBRARIES}
)

