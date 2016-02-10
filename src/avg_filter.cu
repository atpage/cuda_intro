
////////////////////////////////// Includes: ///////////////////////////////////

#include <stdio.h>
#include <time.h>

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

uchar *old_image;  // original image, CPU-side.  only used for pinned input.
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

// Average the pixels in a 3*3 box around pixel (px_x,px_y) of old_image, which
// is intended to be a 32x32 array in shared memory.  Should only be called for
// "interior" pixels of full image.
__device__ uchar average_3x3_shared(uchar *old_image, int px_x, int px_y) {
  float total = 0;
  //int px_used = 0;

  int i,j;
  for (i=px_y-1; i<=px_y+1; i++) {  // loop over 3 rows
    for (j=px_x-1; j<=px_x+1; j++) {  // loop over 3 cols
      total += old_image[ i*32+j ];
      //px_used++;
    }
  }
  //total = total / px_used;
  total = total / 9;

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

// Replace each pixel with a smoothed value.
// This one uses shared memory, with 2d thread/block indexing:
__global__ void avg_filter_shared(uchar *old_image_G, uchar *new_image_G) {
  extern __shared__ uchar old_image_shared[];  // extern means we define the size in kernel launch

  // blocks are 32x32 to process a 30x30 region, so we adjust size and offset to
  // handle the overlap:
  int px_x = blockIdx.x * (blockDim.x-2) + (threadIdx.x-1);  // column of image
  int px_y = blockIdx.y * (blockDim.y-2) + (threadIdx.y-1);  // row of image

  // 32x32 block = 1KB shared memory per block since we're only using
  // uchars.  but 32x32 blocks using floats or doubles would put us at 32KB or
  // 64KB assuming 8 blocks per SM.

  // do my part to populate the shared memory:
  if ((px_y < 0) || (px_y >= image_height_G)) { return; }
  if ((px_x < 0) || (px_x >= image_width_G )) { return; }
  old_image_shared[ threadIdx.y*blockDim.x + threadIdx.x ] = old_image_G[ px_y*image_width_G+px_x ];
  __syncthreads();  // make sure all the data is here before we continue

  // kill the threads that aren't in the inner 30x30 block:
  if ( (threadIdx.x ==  0) ||
       (threadIdx.x == 31) ||
       (threadIdx.y ==  0) ||
       (threadIdx.y == 31) )
    { return; }

  // compute, and write result back to global memory:
  new_image_G[px_y*image_width_G+px_x] = average_3x3_shared(old_image_shared, threadIdx.x, threadIdx.y);
  // TODO: handle borders of image, i.e. when px_x==0 and so on
}

/////////////////////////////// CPU functions: /////////////////////////////////

// Average the pixels in a 3*3 box around pixel (px_x,px_y) of old_image.
// Pixels that fall outside of the image will be excluded from the average.
uchar average_3x3_CPU(uchar *old_image, int px_x, int px_y) {
  float total = 0;
  int px_used = 0;

  int i,j;
  for (i=px_y-1; i<=px_y+1; i++) {  // loop over 3 rows
    if ((i < 0) || (i >= image_height)) { continue; }
    for (j=px_x-1; j<=px_x+1; j++) {  // loop over 3 cols
      if ((j < 0) || (j >= image_width)) { continue; }
      total += old_image[ i*image_width+j ];
      px_used++;
    }
  }
  total = total / px_used;

  return (uchar)total;
}

/////////////////////////////////// main(): ////////////////////////////////////

int main(int argc, char *argv[])
{
  if ( argc != 4 ) {
    printf("Usage: %s <input image> <output image> <mode>\n", argv[0]);
    printf("       mode 0 = CPU, single thread\n");
    printf("       mode 1 = CPU, 4 threads (not implemented)\n");
    printf("       mode 2 = GPU, global memory, 1d grid\n");
    printf("       mode 3 = GPU, global memory, 2d grid\n");
    printf("       mode 4 = GPU, shared memory, 2d grid\n");
    printf("       mode 5 = GPU, global memory, 1d grid, 4 streams\n");
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
  if (mode != 5) {
    new_image = (uchar*)malloc( image_height*image_width*sizeof(uchar) );
    if (new_image == NULL) {
      fprintf(stderr, "Can't malloc space for new_image.  Exiting abruptly.\n");
      exit(EXIT_FAILURE);
    }
  }
  else if (mode == 5) {
    // need to use pinned memory for streaming
    checkCudaErrors( cudaMallocHost((void**)&new_image, image_height*image_width*sizeof(uchar)) );
    checkCudaErrors( cudaMallocHost((void**)&old_image, image_height*image_width*sizeof(uchar)) );
    memcpy(old_image, image.data, image_size);
  }

  if (mode==0) { ///////////////// Single-threaded CPU version /////////////////
    clock_t tic, toc;
    tic = clock();
    for (int row=0; row<image_height; row++) {
      for (int col=0; col<image_width; col++) {
	  new_image[row*image_width + col] = average_3x3_CPU(image.data, col, row);
      }
    }    
    toc = clock();
    printf("CPU time (single thread): %f ms\n", (double)(toc - tic)*1000 / CLOCKS_PER_SEC);
  }

  ///////////////////////////////// GPU code: //////////////////////////////////

  int devID;  // which GPU to use

  // for timing:
  cudaEvent_t start, lap1, lap2, stop;
  float todevms, kernelms, tohostms, total;

  if (mode > 1) {
    // create timers:
    cudaEventCreate(&start);
    cudaEventCreate(&lap1);  // after memcpy to GPU
    cudaEventCreate(&lap2);  // after kernel completes
    cudaEventCreate(&stop);  // after memcpy to CPU

    // Choose the fastest GPU (optional):
    devID = gpuGetMaxGflopsDeviceId();
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
  }

  if ((mode == 2) || 
      (mode == 3) || 
      (mode == 4)) { ///////////////// Non-streaming examples //////////////////

    cudaEventRecord(start);

    // Copy input image to GPU:
    checkCudaErrors(  cudaMemcpy( old_image_G,
				  image.data,
				  image_size*sizeof(uchar),
				  cudaMemcpyHostToDevice )  );
    cudaEventRecord(lap1);

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
    else if (mode == 4) {
      dim3 block_dim = dim3(32, 32);
      dim3 grid_dim = dim3( ceil((float)image_width / 30),
			    ceil((float)image_height / 30) );
      avg_filter_shared<<<grid_dim,block_dim,32*32*sizeof(uchar)>>>
	(old_image_G, new_image_G);
    }
    getLastCudaError("Kernel execution failed (avg_filter).");

    cudaEventRecord(lap2);

    // Copy result back from GPU:
    checkCudaErrors(  cudaMemcpy( new_image,
				  new_image_G,
				  image_size*sizeof(uchar),
				  cudaMemcpyDeviceToHost )  );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&todevms,  start, lap1);
    cudaEventElapsedTime(&kernelms, lap1,  lap2);
    cudaEventElapsedTime(&tohostms, lap2,  stop);
    cudaEventElapsedTime(&total,    start, stop);
    // printf("GPU time (memcpy, kernel, memcpy, total): %f, %f, %f, %f ms\n",
    // 	   todevms, kernelms, tohostms, total);
    printf("GPU time (incl. main memcpy's): %f ms\n", total);
  }

  else if (mode == 5) { ////////////////// Streaming example ///////////////////

    // Prepare streams:
    int nStreams = 4;
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; i++) {
      checkCudaErrors( cudaStreamCreate(&stream[i]) );
    }

    cudaEventRecord(start);  // TODO?: force this to happen before any other stream starts

    // do {copy to device, launch kernel, copy to host} for each stream:
    for (int i = 0; i < nStreams; i++) {
      // note, we assume image size is evenly divisible by nStreams
      int offset = i * image_size*sizeof(uchar)/nStreams;

      checkCudaErrors(  cudaMemcpyAsync( &old_image_G[offset],
					 &old_image[offset],
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

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&total, start, stop);
    printf("GPU time (incl. main memcpy's): %f ms\n", total);
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

  if (mode != 5) {
    free(new_image);
  }
  else if (mode == 5) {
    checkCudaErrors( cudaFreeHost(new_image) );
    checkCudaErrors( cudaFreeHost(old_image) );
  }
  if (mode > 1) {
    checkCudaErrors( cudaFree(new_image_G) );
    checkCudaErrors( cudaFree(old_image_G) );

    cudaEventDestroy(start);
    cudaEventDestroy(lap1);
    cudaEventDestroy(lap2);
    cudaEventDestroy(stop);

    checkCudaErrors( cudaDeviceReset() );
  }
  exit(EXIT_SUCCESS);
}

//////////////////////////////////// Done. /////////////////////////////////////
