#version 450

// For 16bit float type.
#extension GL_EXT_shader_explicit_arithmetic_types : enable

layout (binding = 0, set = 0) uniform MatrixParams
{
    uint size;
} params;

layout (binding = 0, set = 1) readonly buffer InMatrix
{
    TYPE_A data[];
} in_matrix;

layout (binding = 1, set = 1) writeonly buffer OutMatrix
{
    TYPE_B data[];
} out_matrix;

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main()
{
    uint i = /*gl_WorkGroupID.x * gl_WorkGroupSize.x +*/ gl_GlobalInvocationID.x;
    if (i >= 100) {
        return;
    }
    //for (uint i = x; i < params.size; i += gl_WorkGroupSize.x * gl_NumWorkGroups.x)
   // {
        out_matrix.data[i] = TYPE_B(gl_GlobalInvocationID.x); //(in_matrix.data[i]);
    //}
}