
////////////////////////////////// Includes: ///////////////////////////////////

#include <stdio.h>
//#include <unistd.h>

// CUDA stuff:
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

// OpenCV stuff:
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

////////////////////////////// Global variables: ///////////////////////////////

// These are really only global to match/simplify the pseudocode in the slides.

uchar *old_image;  // original image, CPU-side
uchar *new_image;  // processed image, CPU-side
int image_height;  // number of rows in image
int image_width;   // number of columns in image
int chunk_size;    // how many px one CPU thread will process
// GPU versions of the above:
uchar *old_image_G;
uchar *new_image_G;
__constant__ int image_height_G;
__constant__ int image_width_G;

////////////////////////// GPU functions and kernels: //////////////////////////

// Average the pixels in a 3*3 box around pixel (px_x,px_y) of old_image_G.
// Pixels that fall outside of the image will be excluded from the average.
__device__ uchar average_3x3(uchar *old_image_G, int px_x, int px_y) {
  float total = 0;
  int px_used = 0;

  int i,j;
  for (i=px_y-1; i<=px_y+1; i++) {  // loop over 3 rows
    if ((i<0) || (i>=image_height_G)) { continue; }
    for (j=px_x-1; j<=px_x+1; j++) {  // loop over 3 cols
      if ((j < 0) || (j >= image_width_G)) { continue; }
      total += old_image_G[ i*image_width_G+j ];  // old_image_G[i][j];
      px_used++;
    }
  }
  total = total / px_used;

  return (uchar)total;
}

// Replace each pixel with a smoothed value.
// This one uses global memory, and 1d thread/block indexing:
__global__ void avg_filter(uchar *old_image_G, uchar *new_image_G) {
}

// Replace each pixel with a smoothed value.
// This one uses global memory, and 2d thread/block indexing:
__global__ void avg_filter_2D(uchar *old_image_G, uchar *new_image_G) {
  int px_x = blockIdx.x * blockDim.x + threadIdx.x;  // column of image
  int px_y = blockIdx.y * blockDim.y + threadIdx.y;  // row of image
  new_image_G[px_y*image_width_G+px_x] = average_3x3(old_image_G, px_x, px_y);
  //new_image_G[px_x][px_y] = average_3x3(px_x, px_y);
}

/*
// Replace each pixel with a smoothed value.  (shared memory version,
// each thread probably has many pixels to process)
__global__ void lab6_kernel_shared(uchar *GPU_i, uchar *GPU_o, int M, int N, int width)
{
  extern __shared__ uchar shared_GPU_i[];

  // Size of the large box of pixels we're working on.  We expect to
  // have several pixels per thread:
  int sm_box_height = M / gridDim.y;
  int sm_box_width  = N / gridDim.x;
  int px_per_th_y = sm_box_height/blockDim.y;
  int px_per_th_x = sm_box_width/blockDim.x;

  // Indices of first row and column of the global image that this
  // block will need to process:
  int top_row  = blockIdx.y * sm_box_height;
  int left_col = blockIdx.x * sm_box_width;

  // Load data into shared memory.  Note: pixels extending past the
  // borders of this block will not be included, so there will be
  // filtering artifacts at the boundaries of each block.
  int i, j, local_r, local_c, global_offset, local_offset;
  for (i=0; i < px_per_th_y; i++) {
    for (j=0; j < px_per_th_x; j++) {
      local_r = threadIdx.y*px_per_th_y + i;
      local_c = threadIdx.x*px_per_th_x + j;

      //                  row_number     * row_width   +    column_number
      global_offset = (top_row + local_r)*N            + (left_col + local_c);
      local_offset  = (          local_r)*sm_box_width + (           local_c);

      shared_GPU_i[ local_offset ] = GPU_i[ global_offset ];
    }
  }
  __syncthreads();

  // Do the filtering.  It will span the same region we just pre-loaded.
  for (i=0; i < px_per_th_y; i++) {
    for (j=0; j < px_per_th_x; j++) {
      local_r = threadIdx.y*px_per_th_y + i;
      local_c = threadIdx.x*px_per_th_x + j;

      //                  row_number     * row_width   +    column_number
      global_offset = (top_row + local_r)*N            + (left_col + local_c);
      local_offset  = (          local_r)*sm_box_width + (         + local_c);

      GPU_o[ global_offset ] = smoothing_filter(shared_GPU_i, sm_box_height, sm_box_width,
      				 		local_r, local_c, width, width);
    }
  }
}
*/

////////////////////////////////// CPU code: ///////////////////////////////////

int main(int argc, char *argv[])
{
  if ( argc != 4 ) {
    printf("Usage: %s <input image> <output image> <mode>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Load image:
  Mat image;	// see http://docs.opencv.org/modules/core/doc/basic_structures.html#mat
  image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
  // we could load it as CV_LOAD_IMAGE_COLOR, but we don't want to worry about
  // that extra dimension in our example
  if(! image.data ) {
    fprintf(stderr, "Could not open or find the image.\n");
    exit(EXIT_FAILURE);
  }
  printf("Loaded image '%s', size = %dx%d (dims = %d).\n", argv[1], image.rows, image.cols, image.dims);

  // Set up global variables based on image size:
  image_height = image.rows;
  image_width  = image.cols;

  int image_size = image_height*image_width;
  //int N = 4;  // number of CPU threads
  //int chunk_size = image_size / N;

  // Prepare CPU output array:
  new_image = (uchar*)malloc( image_height*image_width*sizeof(uchar) );
  if (new_image == NULL) {
    fprintf(stderr, "Can't malloc space for new_image.  Exiting abruptly.\n");
    exit(EXIT_FAILURE);
  }

  ///////////////////////////////// GPU code: //////////////////////////////////

  // Choose the fastest GPU (optional):
  int devID = gpuGetMaxGflopsDeviceId();
  checkCudaErrors( cudaSetDevice(devID) );
  // checkCudaErrors( cudaSetDevice(0) );  // or just use the first GPU

  // Prepare input and output arrays on GPU:
  checkCudaErrors( cudaMalloc((void**)&old_image_G,image_size*sizeof(uchar)) );
  checkCudaErrors( cudaMalloc((void**)&new_image_G,image_size*sizeof(uchar)) );

  // Copy CPU data to GPU:
  checkCudaErrors(  cudaMemcpy( old_image_G,
				image.data,
				image_size*sizeof(uchar),
				cudaMemcpyHostToDevice )  );

  checkCudaErrors( cudaMemcpyToSymbol(image_height_G, &image_height, sizeof(int)) );
  checkCudaErrors( cudaMemcpyToSymbol(image_width_G,  &image_width,  sizeof(int)) );

  // Run the kernel:
  dim3 block_dim = dim3(32, 32);
  dim3 grid_dim = dim3(image_width / block_dim.x, image_height / block_dim.y);  // TODO: round up
  avg_filter_2D<<<grid_dim,block_dim>>>(old_image_G, new_image_G);
  getLastCudaError("Kernel execution failed (avg_filter).");
  // checkCudaErrors( cudaDeviceSynchronize() );  // not needed

  // Copy result back:
  checkCudaErrors(  cudaMemcpy( new_image,
				new_image_G,
				image_size*sizeof(uchar),
				cudaMemcpyDeviceToHost )  );

  // Save the output to disk:
  Mat result = Mat(image_height, image_width, CV_8UC1, new_image);
  string output_filename = argv[2];
  if (!imwrite(output_filename, result)) {
    fprintf(stderr, "Couldn't write output to disk!  Exiting abruptly.\n");
    exit(EXIT_FAILURE);
  }
  printf("Saved image '%s', size = %dx%d (dims = %d).\n",
	 output_filename.c_str(), result.rows, result.cols, result.dims);

  // Clean up:
  free(new_image);
  checkCudaErrors( cudaFree(new_image_G) );
  checkCudaErrors( cudaFree(old_image_G) );
  checkCudaErrors( cudaDeviceReset() );

  // Done.
  exit(EXIT_SUCCESS);
}
