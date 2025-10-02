#version 450
#extension GL_EXT_shader_8bit_storage : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: enable
#extension GL_EXT_shader_explicit_arithmetic_types: enable

layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0, set = 1) buffer ImageInOut {
    PIXELTYPE data[];
} src_image;

layout(binding = 0, set = 0) uniform ImageInfo {
    uint width;
    uint height;
} image_info;

void main() {

    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    
    uint halfHeight = image_info.height / 2;

    if (x >= image_info.width || y >= halfHeight) {
        return;
    }

    uint yIdxBottom = image_info.height - y - 1;

    PIXELTYPE tmp = src_image.data[x + image_info.width * yIdxBottom];
    src_image.data[x + image_info.width * yIdxBottom] = src_image.data[x + image_info.width * y];
    src_image.data[x + image_info.width * y] = tmp;
}

