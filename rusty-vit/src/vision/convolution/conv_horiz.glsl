#version 450

#extension GL_EXT_shader_8bit_storage : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: enable
#extension GL_EXT_shader_explicit_arithmetic_types: enable

layout (constant_id = 0) const int TILE_WIDTH = 16;
layout (constant_id = 1) const int TILE_HEIGHT = 4;
layout (constant_id = 2) const int HALO_STEPS = 1;
layout (constant_id = 3) const int RESULT_STEPS = 8;

layout(local_size_x_id = 0, local_size_y_id = 1) in;

shared PIXELTYPE sData[TILE_HEIGHT][(RESULT_STEPS + 2 * HALO_STEPS) * TILE_WIDTH];

layout (binding = 0, set = 0) uniform ImageInfo
{
    int width;
    int height;
    int kernelDim;
} image_info;

layout (binding = 0, set = 1) buffer SrcImage
{
    PIXELTYPE data[];
} src_image;

layout (binding = 1, set = 1) buffer DstImage
{
    PIXELTYPE data[];
} dst_image;

layout (binding = 2, set = 1) buffer KernelData
{
    float data[];
} kernel;

void main() 
{
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);

    int baseX = (int(gl_WorkGroupID.x) * RESULT_STEPS - HALO_STEPS) * TILE_WIDTH + x;
    int baseY = int(gl_WorkGroupID.y) * TILE_HEIGHT + y;
    int offset = baseX + image_info.width * baseY;
    
    for (int i = HALO_STEPS; i < RESULT_STEPS + HALO_STEPS; ++i)
    {
        sData[y][x + i * TILE_WIDTH] = src_image.data[offset + (i * TILE_WIDTH)];
    }
    for (int i = 0; i < HALO_STEPS; ++i)
    {
        sData[y][x + i * TILE_WIDTH] = (baseX >= -i * TILE_WIDTH) ? src_image.data[offset + (i * TILE_WIDTH)] : PIXELTYPE(0);
    }
    for (int i = HALO_STEPS + RESULT_STEPS; i < RESULT_STEPS + 2 * HALO_STEPS; ++i)
    {
        sData[y][x + i * TILE_WIDTH] = (image_info.width - baseX > i * TILE_WIDTH) ? src_image.data[offset + (i * TILE_WIDTH)] : PIXELTYPE(0);
    }
    // Sync the threads to load the pixels into shared memory.
    memoryBarrierShared();
    barrier();

     for (int i = HALO_STEPS; i < HALO_STEPS + RESULT_STEPS; i++)
    {
        float sum = 0.0;
        for (int j = 0; j < int(image_info.kernelDim); j++)
        {
            sum += kernel.data[j] * float(sData[y][x + i * TILE_WIDTH + (j - (image_info.kernelDim >> 1))]);
        }
        dst_image.data[offset + (i * TILE_WIDTH)] = PIXELTYPE(sum);
    }
}