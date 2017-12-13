# cuda_project
JHU GPU Project

# Dependencies
- OpenCV
  - Built with CUDA

# Local Area Contrast Enhancement
In the code here, I attempt to write my own local area contrast enhancement(LACE) algorithm. In my code, I use OpenCV built with CUDA to help assist with some of the helper functions not related to contrast enhancement. For my version of LACE, I change the local standard deviations of the image in attempts to increase contrast. 

# Steps
1) Convert image from RGB (red, green, blue) to HSV (hue, saturation, and value)
    - You want to grab the intensity channel of the image to enhance it's contrast (not affect the color)
2) Calculate the mean and standard deviations of the image in blocks
    - This is where the local in local area contrast enhancement comes in. You want to increase the standard deviation within local regions
3) Then change the standard deviation per pixel
4) Set the new intensity image with the Hue and Saturation channels
5) Convert the image back to RGB
6) Display the image
