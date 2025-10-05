#version 450

#extension GL_EXT_shader_8bit_storage : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: enable
#extension GL_EXT_shader_explicit_arithmetic_types: enable

layout(local_size_x = 32, local_size_y = 8) in;

#define CONSTANT_BORDER 0
#define CLAMP_TO_EDGE_BORDER 1
#define MIRRORED_BORDER 2
layout (constant_id = 0) const uint BORDER_OP = CONSTANT_BORDER;

layout (binding = 0, set = 0) uniform ImageInfo
{
    uint width;
    uint height;
    uint padding;
} image_info;

layout (binding = 0, set = 1) buffer SrcImage
{
    PIXELTYPE data[];
} src_image;

layout (binding = 1, set = 1) buffer DstImage
{
    PIXELTYPE data[];
} dst_image;


bool clampToEdgeBorder(int pos, int srcDim, int padding, inout int outIdx)
{
    if (pos - padding < 0)
    {
        outIdx = 0;
        return true;
    }
    else if (pos >= srcDim)
    {
        outIdx = srcDim - 1;
        return true;
    }
    return false;
}

bool mirrorBorder(int pos, int srcDim, int padding, inout int outIdx)
{
    if (pos - padding < 0)
    {
        outIdx = padding - abs(pos);
        return true;
    }
    else if (pos >= srcDim)
    {
        outIdx = srcDim - (pos - srcDim) - 1;
        return true;
    }
    return false;
}

void main()
{
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    if (x >= image_info.width || y >= image_info.height)
    {
        return;
    }

    uint newWidth = image_info.width + 2 * image_info.padding;
    uint newHeight = image_info.height + 2 * image_info.padding;

    int xPos = 0;
    int yPos = 0;
    bool isXBorder = false;
    bool isYBorder = false;
    if (BORDER_OP == CLAMP_TO_EDGE_BORDER)
    {
        isXBorder = clampToEdgeBorder(int(x), int(image_info.width), int(image_info.padding), xPos);
        isYBorder = clampToEdgeBorder(int(y), int(image_info.height), int(image_info.padding), yPos);
    }
    else if (BORDER_OP == MIRRORED_BORDER)
    {
        isXBorder = mirrorBorder(int(x), int(image_info.width), int(image_info.padding), xPos);
        isYBorder = mirrorBorder(int(y), int(image_info.height), int(image_info.padding), yPos);
    }

    if (isXBorder && isYBorder) 
    { 
        dst_image.data[x + newWidth * y] = src_image.data[xPos + image_info.width * yPos]; 
    } 
    else if (isYBorder) 
    { 
        dst_image.data[x + newWidth * y] = src_image.data[x + image_info.width * yPos]; 
    } 
    else if (isXBorder) 
    { 
        dst_image.data[x + newWidth * y] = src_image.data[xPos + image_info.width * y]; 
    } 
 
    uint xPad = x + image_info.padding;
    uint yPad = y + image_info.padding;
    dst_image.data[xPad + newWidth * yPad] = src_image.data[x + image_info.width * y];
}