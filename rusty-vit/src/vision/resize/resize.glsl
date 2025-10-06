#version 450

#extension GL_EXT_shader_8bit_storage : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: enable
#extension GL_EXT_shader_explicit_arithmetic_types: enable

#define FILTER_OP_BILINEAR 0
layout (constant_id = 0) const uint FILTER_OP = FILTER_OP_BILINEAR;

layout(local_size_x = 32, local_size_y = 8) in;

layout (binding = 0, set = 0) uniform ImageInfo
{
    uint src_width;
    uint src_height;
    uint dst_width;
    uint dst_height;
    float scale_x;
    float scale_y;
} image_info;

layout (binding = 0, set = 1) readonly buffer SrcImage
{
    PIXELTYPE data[];
} src_image;

layout (binding = 1, set = 1) writeonly buffer DstImage
{
    PIXELTYPE data[];
} dst_image;

float lerp(float a, float b, float c) 
{
    return fma(c, b, fma(-c, a, a));
}

float getPixel(float x, float y, uint width, uint height, uint stride)
{
    // If the xy co-ords are outside the bounds of the src image, then clamp to border.
    float xPos = x < 0.0 ? 0.0 : x >= width ? width - 1.0 : x;
    float yPos = y < 0.0 ? 0.0 : y >= height ? height - 1.0 : y;
    return float(src_image.data[int(round(xPos)) + stride * int(round(yPos))]);
}

PIXELTYPE bilinearOp(uint srcWidth, uint srcHeight, float x, float y, vec2 scale)
{
    float scaledX = (x + 0.5) * scale.x - 0.5;
    float scaledY = (y + 0.5) * scale.y - 0.5;

    // Floor the coordinates to get to the nearest valid pixel.
    const float px = floor(scaledX);
    const float py = floor(scaledY);

    // Set weights of pixels according to distance from the actual location.
    const float fractX = scaledX - px;
    const float fractY = scaledY - py;

    float p0 = getPixel(px, py, srcWidth, srcHeight, srcWidth);
    float p1 = getPixel(px + 1.0, py, srcWidth, srcHeight, srcWidth);
    float p2 = getPixel(px, py + 1.0, srcWidth, srcHeight, srcWidth);
    float p3 = getPixel(px + 1.0, py + 1.0, srcWidth, srcHeight, srcWidth);

    return PIXELTYPE(
        lerp(lerp(p0, p1, fractX), lerp(p2, p3, fractX), fractY));
}

void main()
{
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    if (x >= image_info.dst_width || y >= image_info.dst_height)
    {
        return;
    }

    PIXELTYPE p;
    if (FILTER_OP == FILTER_OP_BILINEAR)
    {
        p = bilinearOp(image_info.src_width, image_info.src_height, float(x), float(y), vec2(image_info.scale_x, image_info.scale_y));
    }
    
    dst_image.data[x + image_info.dst_width * y] = p;
}