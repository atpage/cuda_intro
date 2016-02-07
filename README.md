# Intro to CUDA programming

This code is designed to walk you through optimizing an example GPU application, starting from the basic CPU version.  The accompanying slides are available [here](https://docs.google.com/presentation/d/1cHeB_jPzqN0Im98ECfBQDQwzlBp3nn0EZ-Yq1ZHGEwA/edit?usp=sharing).

## Description of source files

* `avg_filter.cu` is the main code, which demonstrates several different ways to implement a 3x3 image smoothing (averaging) filter on the GPU.
* `helper_cuda.h` and `helper_string.h` are from the CUDA samples.  They provide handy functions like `gpuGetMaxGflopsDeviceId()` and `checkCudaErrors()`.

## TODO

* Finish slides
* Make CPU example(s)
* Add timing to each thing
* Libraries (how to use, how to make our own)
