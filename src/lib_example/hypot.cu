#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

using namespace std;

__global__ void hypotKernel(float* A, float* B, float* C, int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx > len) { return; }
  C[idx] = sqrt( A[idx]*A[idx] + B[idx]*B[idx] );
}

extern "C"
int gpuHypot(float* A, float* B, float* C, int len) {
  // pick best GPU:
  int devID = gpuGetMaxGflopsDeviceId();
  checkCudaErrors( cudaSetDevice(devID) );

  // allocate and initialize GPU memory:
  float* A_G;
  float* B_G;
  float* C_G;
  checkCudaErrors( cudaMalloc((float**) &A_G, sizeof(float) * len) );
  checkCudaErrors( cudaMalloc((float**) &B_G, sizeof(float) * len) );
  checkCudaErrors( cudaMalloc((float**) &C_G, sizeof(float) * len) );
  checkCudaErrors( cudaMemcpy(A_G, A, len*sizeof(float), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(B_G, B, len*sizeof(float), cudaMemcpyHostToDevice) );

  // run kernel:
  hypotKernel <<< len/128 + 1, 128 >>> (A_G, B_G, C_G, len);
  getLastCudaError("Kernel execution failed (hypotKernel)");

  // copy results back to CPU:
  checkCudaErrors( cudaMemcpy(C, C_G, len*sizeof(float), cudaMemcpyDeviceToHost) );

  // Clean up:
  checkCudaErrors( cudaFree(A_G) );
  checkCudaErrors( cudaFree(B_G) );
  checkCudaErrors( cudaFree(C_G) );
  cudaDeviceReset();

  return EXIT_SUCCESS;
}
