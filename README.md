# RustyViT

A vision transformer library built from scratch.

## CUDA Setup ##

The library requires the CUDA toolkit to be downloaded. The folder which contains the `nvcc`
compiler should be
added to `PATH`.

## Vulkan Setup ##

To compile the glsl shaders, the `shaderc` library is required which is part of the Vulkan SDK
download. The
path which contains the `libshaderc` library should be declared in the `SHADERC_DIR_PATH`
environment variable.

## Developing ##

### Vulkan - Renderdoc

For debugging Vulkan GLSL shaders, Renderdoc is an indispensable tool for digging into issues with
when running shaders.
Though it can be a bit tricky getting things to work when running Vulkan in headless compute mode.
The renderdoc
API is used to capture frames in the Vulkan code, though ensure that before running Renderdoc you
set
the env variable `DISABLE_VULKAN_RENDERDOC_CAPTURE_1_40=0` otherwise no capture will occur.