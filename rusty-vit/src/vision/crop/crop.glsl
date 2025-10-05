#version 450

#extension GL_EXT_shader_8bit_storage : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: enable
#extension GL_EXT_shader_explicit_arithmetic_types: enable

layout(local_size_x = 32, local_size_y = 8) in;

layout (binding = 0, set = 0) uniform ImageInfo 
{
    uint src_width;
    uint src_height;
    uint dst_width;
    uint dst_height;
    uint x_offset;
    uint y_offset;
} image_info;

layout (binding = 0, set = 1) buffer SrcImage
{
    PIXELTYPE data[];
} src_image;

layout (binding = 1, set = 1) buffer DstImage
{
    PIXELTYPE data[];
} dst_image;

void main()
{
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    if (x >= image_info.dst_width || y >= image_info.dst_height)
    {
        return;
    }

    PIXELTYPE p = src_image.data[x + image_info.x_offset + image_info.src_width * y + image_info.y_offset];
    dst_image.data[x + image_info.dst_width * y] = p; 
}