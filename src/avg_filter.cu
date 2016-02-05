
////////////////////////////////// Includes: ///////////////////////////////////

#include <stdio.h>

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
// TODO: test speedup from using constant memory

////////////////////////// GPU functions and kernels: //////////////////////////

// Average the pixels in a 3*3 box around pixel (px_x,px_y) of old_image_G.
// Pixels that fall outside of the image will be excluded from the average.
__device__ uchar average_3x3(uchar *old_image_G, int px_x, int px_y) {
  float total = 0;
  int px_used = 0;

  int i,j;
  for (i=px_y-1; i<=px_y+1; i++) {  // loop over 3 rows
    if ((i < 0) || (i >= image_height_G)) { continue; }
    for (j=px_x-1; j<=px_x+1; j++) {  // loop over 3 cols
      if ((j < 0) || (j >= image_width_G)) { continue; }
      total += old_image_G[ i*image_width_G+j ];  // read from old_image_G[i][j];
      px_used++;
    }
  }
  total = total / px_used;

  return (uchar)total;
}

// Replace each pixel with a smoothed value.
// This one uses global memory, and 1d thread/block indexing:
__global__ void avg_filter(uchar *old_image_G, uchar *new_image_G) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 1d index into image
  if (idx >= image_height_G*image_width_G) { return; }  // off image
  int px_x = idx % image_width_G;  // column of image
  int px_y = idx / image_width_G;  // row of image
  new_image_G[idx] = average_3x3(old_image_G, px_x, px_y);
}

// Replace each pixel with a smoothed value.
// This one uses global memory, and 2d thread/block indexing:
__global__ void avg_filter_2D(uchar *old_image_G, uchar *new_image_G) {
  int px_x = blockIdx.x * blockDim.x + threadIdx.x;  // column of image
  int px_y = blockIdx.y * blockDim.y + threadIdx.y;  // row of image
  if ((px_x >= image_width_G) || (px_y >= image_height_G)) { return; }  // off image
  new_image_G[px_y*image_width_G+px_x] = average_3x3(old_image_G, px_x, px_y);
  //new_image_G[px_x][px_y] = average_3x3(px_x, px_y);
}
// The 2D version seems to be about the same speed as the 1D version.
// Apparently cache isn't helping us.

// Replace each pixel with a smoothed value.  Start at offset into image, not 0.
// This one uses global memory, and 1d thread/block indexing:
__global__ void avg_filter_part(uchar *old_image_G, uchar *new_image_G, int offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;  // 1d index into image
  if (idx >= image_height_G*image_width_G) { return; }  // off image
  int px_x = idx % image_width_G;  // column of image
  int px_y = idx / image_width_G;  // row of image
  new_image_G[idx] = average_3x3(old_image_G, px_x, px_y);
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
    printf("       mode 0 = CPU, single thread\n");
    printf("       mode 1 = CPU, 4 threads\n");
    printf("       mode 2 = GPU, global memory, 1d grid\n");
    printf("       mode 3 = GPU, global memory, 2d grid\n");
    printf("       mode 4 = GPU, global memory, 1d grid, 4 streams\n");
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
  int mode = atoi(argv[3]);

  // Set up global variables based on image size:
  image_height = image.rows;
  image_width  = image.cols;

  int image_size = image_height*image_width;
  //int N = 4;  // number of CPU threads
  //int chunk_size = image_size / N;

  // Prepare CPU output array:
  if (mode != 4) {
    new_image = (uchar*)malloc( image_height*image_width*sizeof(uchar) );
  }
  else if (mode == 4) {
    // need to use pinned memory for streaming
    checkCudaErrors( cudaMallocHost((void**)&new_image, image_height*image_width*sizeof(uchar)) );
  }

  if (new_image == NULL) {
    fprintf(stderr, "Can't malloc space for new_image.  Exiting abruptly.\n");
    exit(EXIT_FAILURE);
  }

  ///////////////////////////////// GPU code: //////////////////////////////////

  // TODO: only come in here if mode>1

  // Choose the fastest GPU (optional):
  int devID = gpuGetMaxGflopsDeviceId();
  checkCudaErrors( cudaSetDevice(devID) );
  // checkCudaErrors( cudaSetDevice(0) );  // or just use the first GPU

  // Prepare input and output arrays on GPU:
  checkCudaErrors( cudaMalloc((void**)&old_image_G,image_size*sizeof(uchar)) );
  checkCudaErrors( cudaMalloc((void**)&new_image_G,image_size*sizeof(uchar)) );

  // Set GPU output to zero (optional):
  //checkCudaErrors( cudaMemset(new_image_G, 0, image_size) );

  // Copy CPU variables to GPU:
  checkCudaErrors( cudaMemcpyToSymbol(image_height_G, &image_height, sizeof(int)) );
  checkCudaErrors( cudaMemcpyToSymbol(image_width_G,  &image_width,  sizeof(int)) );

  if ((mode == 2) || (mode == 3)) { ////////// Non-streaming examples //////////

    // Copy input image to GPU:
    checkCudaErrors(  cudaMemcpy( old_image_G,
				  image.data,
				  image_size*sizeof(uchar),
				  cudaMemcpyHostToDevice )  );

    // Run the kernel.  ceil() so we don't miss the end(s) of the image:
    if (mode == 2) {
      avg_filter<<< ceil((float)image_size/256), 256 >>>(old_image_G, new_image_G);
    }
    else if (mode == 3) {
      dim3 block_dim = dim3(32, 32);
      dim3 grid_dim = dim3( ceil((float)image_width / block_dim.x),
			    ceil((float)image_height / block_dim.y) );
      avg_filter_2D<<<grid_dim,block_dim>>>(old_image_G, new_image_G);
    }
    getLastCudaError("Kernel execution failed (avg_filter).");

    // Copy result back from GPU:
    checkCudaErrors(  cudaMemcpy( new_image,
				  new_image_G,
				  image_size*sizeof(uchar),
				  cudaMemcpyDeviceToHost )  );
  }

  else if (mode == 4) { ////////////////// Streaming example ///////////////////

    // Prepare streams:
    int nStreams = 4;
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; i++) {
      checkCudaErrors( cudaStreamCreate(&stream[i]) );
    }

    // do {copy to device, launch kernel, copy to host} for each stream:
    for (int i = 0; i < nStreams; i++) {
      // note, we assume image size is evenly divisible by nStreams
      int offset = i * image_size*sizeof(uchar)/nStreams;

      checkCudaErrors(  cudaMemcpyAsync( &old_image_G[offset],
					 &image.data[offset],
					 image_size*sizeof(uchar)/nStreams,
					 cudaMemcpyHostToDevice,
					 stream[i] )  );
      // TODO: copy a bit extra to handle boundaries

      avg_filter_part<<< ceil((float)image_size/256/nStreams), 256, 0, stream[i] >>>
	(old_image_G, new_image_G, offset);
      // TODO: will probably want getLastCudaError() here

      checkCudaErrors(  cudaMemcpyAsync( &new_image[offset],
					 &new_image_G[offset],
					 image_size*sizeof(uchar)/nStreams,
					 cudaMemcpyDeviceToHost,
					 stream[i])  );
    }

    // Destroy streams:
    for (int i = 0; i < nStreams; i++) {
      checkCudaErrors( cudaStreamDestroy(stream[i]) );
      // cudaStreamDestroy() blocks until all commands in given stream complete
    }
  }

  ////////////////////////// Save the output to disk: //////////////////////////

  Mat result = Mat(image_height, image_width, CV_8UC1, new_image);
  string output_filename = argv[2];
  if (!imwrite(output_filename, result)) {
    fprintf(stderr, "Couldn't write output to disk!  Exiting abruptly.\n");
    exit(EXIT_FAILURE);
  }
  printf("Saved image '%s', size = %dx%d (dims = %d).\n",
	 output_filename.c_str(), result.rows, result.cols, result.dims);

  ///////////////////////////// Clean up and exit: /////////////////////////////

  if (mode != 4) {
    free(new_image);
  }
  else if (mode == 4) {
    checkCudaErrors( cudaFreeHost(new_image) );
  }

  checkCudaErrors( cudaFree(new_image_G) );
  checkCudaErrors( cudaFree(old_image_G) );
  checkCudaErrors( cudaDeviceReset() );

  exit(EXIT_SUCCESS);
}

//////////////////////////////////// Done. /////////////////////////////////////
