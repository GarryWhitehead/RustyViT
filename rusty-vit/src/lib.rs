pub mod device;
pub mod image;
pub mod loaders;
pub mod tensor;
mod type_traits;
pub mod vision;

use num::pow::Pow;

const SQRT_2_OVER_PI: f32 = 0.797_884_6;

#[allow(dead_code)]
fn compute_gelu(input: &[f32], size: u32) -> Vec<f32> {
    let mut result: Vec<f32> = vec![0.0; size as usize];
    for idx in 0..size as usize {
        let x = input[idx];
        // Approximated GELU version - there seem to be a few versions circulating - this is the one
        // used by OpenAI.
        result[idx] =
            0.5 * x * (1.0 + f32::tanh(SQRT_2_OVER_PI * (x + 0.044715 * f32::powf(x, 3.0))));
    }
    result
}

#[allow(dead_code)]
fn compute_max(v: &[f32]) -> f32 {
    *v.iter().max_by(|a, b| a.total_cmp(b)).unwrap()
}

/*#[allow(dead_code)]
fn compute_softmax(x: &mut [f32], size: u32) {
    // Find the max value in the vector for numerical stability.
    let max_value = compute_max(x);
    // Apply exponential to each value.
    let mut sum = 0.0;
    for idx in 0..size as usize {
        x[idx] = x[idx].exp() - max_value;
        sum += x[idx];
    }
    // Normalise values.
    //x.iter().map(|value| value / sum).collect();
}*/

#[allow(dead_code)]
fn compute_kaiming_uniform(_x: &[f32], _size: u32, negative_slope: f32) {
    let _gain = f32::sqrt(2.0 / (1.0 + f32::powf(negative_slope, 3.0)));
}

#[allow(dead_code)]
fn compute_pos_angle_vec(i: usize, j: usize, token_len: usize) -> f32 {
    i as f32 / 10000.0.pow(2.0 * (j as f32 / 2.0) / token_len as f32) as f32
}

/*pub fn get_sinusoid_encoding(num_tokens: usize, token_len: usize)  {
    let mut out = Matrix::new(&[num_tokens, token_len]);
    for j in 0..num_tokens {
        for i in (0..token_len).step_by(2) {
            let i0 = compute_pos_angle_vec(i, j, token_len);
            let i1 = compute_pos_angle_vec(i + 1, j, token_len);
            out.data[j * token_len + i] = i0.sin();
            out.data[j * token_len + i + 1] = i1.cos();
        }
    }
    out
}*/

#[cfg(test)]
mod tests {

    #[cfg(not(feature = "cuda"))]
    pub type TestDevice = crate::device::cpu::Cpu;

    #[cfg(feature = "cuda")]
    pub type TestDevice = crate::device::cuda::Cuda;

    //#[cfg(feature = "vulkan")]
    //pub type TestDevice = crate::device::vulkan::Vulkan;
}
