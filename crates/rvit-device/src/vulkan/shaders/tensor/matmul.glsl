#version 450

#extension GL_KHR_memory_scope_semantics : enable
#extension GL_KHR_cooperative_matrix : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_KHR_shader_subgroup_basic : enable
// For [[ unroll ]]
#extension GL_EXT_control_flow_attributes : enable

layout (binding = 0, set = 0) uniform MatrixParams 
{
    uint M;
    uint K;
    uint N;
    uint lda;
    uint ldb;
    uint ldc;
    FLOAT_TYPE alpha;
    FLOAT_TYPE beta;
} m_params;

layout (binding = 0, set = 1) readonly buffer MatrixA
{
    FLOAT_TYPE data[];
} m_a;

layout (binding = 1, set = 1) readonly buffer MatrixB
{
    FLOAT_TYPE data[];
} m_b;

layout (binding = 2, set = 1) readonly buffer MatrixC
{
    FLOAT_TYPE data[];
} m_c;

layout (binding = 3, set = 1) writeonly buffer MatrixD
{
    FLOAT_TYPE data[];
} m_d;

layout (constant_id = 0) const uint TILE_M = 64;
layout (constant_id = 1) const uint TILE_K = 16;
layout (constant_id = 2) const uint TILE_N = 64;
layout (constant_id = 3) const uint lM = 16;
layout (constant_id = 4) const uint lK = 16;
layout (constant_id = 5) const uint lN = 16;
layout (constant_id = 6) const uint WG_WIDTH = 4;
layout (constant_id = 7) const uint WG_HEIGHT = 2;
layout (constant_id = 8) const uint WARP_SIZE = 32;

#define SUBGROUP_COUNT WG_WIDTH * WG_HEIGHT
const uint WORKGROUP_INVS = WARP_SIZE * SUBGROUP_COUNT;

#define SDATA_PAD 1
#define SDATA_STRIDE_A (TILE_K + SDATA_PAD)
#define SDATA_STRIDE_B (TILE_N + SDATA_PAD)

shared FLOAT_TYPE sDataA[TILE_M * TILE_K + SDATA_PAD];
shared FLOAT_TYPE sDataB[TILE_K * TILE_N + SDATA_PAD];

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main()
{
    uvec2 tileId = uvec2(gl_WorkGroupID.xy);
    uvec2 warpInTile = uvec2(gl_SubgroupID % WG_WIDTH, gl_SubgroupID / WG_WIDTH);

    const uint c_rows = TILE_M / WG_HEIGHT / lM;
    const uint c_cols = TILE_N / WG_WIDTH / lN;

    coopmat<COOP_FLOAT_TYPE, gl_ScopeSubgroup, lM, lK, gl_MatrixUseA> matA[c_rows];
    coopmat<COOP_FLOAT_TYPE, gl_ScopeSubgroup, lK, lN, gl_MatrixUseB> matB;
    coopmat<COOP_FLOAT_TYPE, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> matC[c_cols];
    coopmat<COOP_FLOAT_TYPE, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> result[c_rows * c_cols];

    // Initialise the output buffer.
    [[unroll]] for (uint i = 0; i < c_rows * c_cols; i++) 
    {
        result[i] = coopmat<COOP_FLOAT_TYPE, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(0.0);
    }

    uint aTileK = gl_LocalInvocationID.x % TILE_K;
    const uint WORK_INV_PER_ROW_A = WORKGROUP_INVS / TILE_K;

    FLOAT_TYPE tmpA[TILE_M / WORK_INV_PER_ROW_A];
    uint baseA = m_params.lda * (TILE_M * tileId.y); 
    
    [[unroll]] for (uint i = 0; i < TILE_M; i += WORK_INV_PER_ROW_A) 
    {
        uint aTileI = i + gl_LocalInvocationID.x / TILE_K;
        tmpA[i / WORK_INV_PER_ROW_A] = m_a.data[baseA + m_params.lda * aTileI + aTileK];
    }

    const uint WORK_INV_PER_ROW_B = WORKGROUP_INVS / TILE_N;
    uint bTileJ = gl_LocalInvocationID.x % TILE_N;

    FLOAT_TYPE tmpB[TILE_K / WORK_INV_PER_ROW_B];
    uint baseB = m_params.ldb * 0 + (TILE_K * tileId.x);

    [[unroll]] for (uint k = 0; k < TILE_K; k += WORK_INV_PER_ROW_B) 
    {
        uint bTileK = k + gl_LocalInvocationID.x / TILE_N;
        tmpB[k / WORK_INV_PER_ROW_B] = m_b.data[baseB + m_params.ldb * bTileK + bTileJ];
    }

    for (uint chunkK = 0; chunkK < m_params.K; chunkK += TILE_K) 
    {
        bool last = ((chunkK + TILE_K) >= m_params.K);
        // Ensure the work from the pre-fetch in the previous loop has finished.
        barrier();

        [[unroll]] for (uint i = 0; i < TILE_M; i += WORK_INV_PER_ROW_A) 
        {
            uint si = i + gl_LocalInvocationID.x / TILE_K;
            sDataA[SDATA_STRIDE_A * si + aTileK] = tmpA[i / WORK_INV_PER_ROW_A];
        }

        [[unroll]] for (uint k = 0; k < TILE_K; k += WORK_INV_PER_ROW_B) 
        {
            uint sk = k + gl_LocalInvocationID.x / TILE_N;
            sDataB[SDATA_STRIDE_B * sk + bTileJ] = tmpB[k / WORK_INV_PER_ROW_B];
        }
        barrier();

        baseA += TILE_K;
        [[unroll]] for (uint i = 0; i < TILE_M; i += WORK_INV_PER_ROW_A) 
        {
            uint aTileI = i + gl_LocalInvocationID.x / TILE_K;
            if (!last) 
            {
                tmpA[i / WORK_INV_PER_ROW_A] = m_a.data[baseA + m_params.lda * aTileI + aTileK];
            }
        }

        baseB += TILE_K * m_params.ldb;
        [[unroll]] for (uint k = 0; k < TILE_K; k += WORK_INV_PER_ROW_B) 
        {
            uint bTileK = k + gl_LocalInvocationID.x / TILE_N;
            if (!last) 
            { 
                tmpB[k / WORK_INV_PER_ROW_B] = m_b.data[baseB + m_params.ldb * bTileK + bTileJ];
            }
        }

        [[unroll]] for (uint k = 0; k < TILE_K / m_params.K; ++k)
        {
            uint sk = m_params.K * k;
            
            [[unroll]] for (uint i = 0; i < c_rows; ++i) 
            {
                uint si = lM * (c_rows * warpInTile.y + i);
                uint offset = SDATA_STRIDE_A * si + sk;
                coopMatLoad(matA[i], sDataA, offset, SDATA_STRIDE_A, gl_CooperativeMatrixLayoutRowMajor);
            }
            [[unroll]] for (uint j = 0; j < c_cols; ++j) 
            {
                uint sj = lN * (c_cols * warpInTile.x + j);
                uint offset = SDATA_STRIDE_B * sk + sj;
                coopMatLoad(matB, sDataB, offset, SDATA_STRIDE_B, gl_CooperativeMatrixLayoutRowMajor);

                [[unroll]] for (uint i = 0; i < c_rows; ++i) 
                {
                    result[i + c_cols * j] = coopMatMulAdd(matA[i], matB, result[i + c_cols * j]);
                }
            }
        }
    }

    [[unroll]] for (uint i = 0; i < c_rows; ++i) 
    {
        uint gi = TILE_M * tileId.y + lM * (c_rows * warpInTile.y + i);
        
        [[unroll]] for (uint j = 0; j < c_cols; ++j) 
        {
            uint gj = TILE_N * tileId.x + lN * (c_cols * warpInTile.x + j);
            uint offset = m_params.ldc * gi + gj;
            coopMatLoad(matC[j], m_c.data, offset, m_params.ldc, gl_CooperativeMatrixLayoutRowMajor);
        }

        [[unroll]] for (uint j = 0; j < c_cols; ++j) 
        {
            uint gj = TILE_N * tileId.x + lN * (c_cols * warpInTile.x + j);
            uint offset = m_params.ldc * gi + gj;
            result[i * c_cols + j] = m_params.alpha * result[i * c_cols + j] + m_params.beta * matC[j];
            coopMatStore(result[i * c_cols + j], m_d.data, offset, m_params.ldc, gl_CooperativeMatrixLayoutRowMajor);
        }
    }
}