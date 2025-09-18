fn layernorm(
    input: &[f32],
    means: &mut [f32],
    rstds: &mut [f32],
    weight: &[f32],
    bias: &[f32],
    shape: &[usize; 3],
    out: &mut [f32],
) {
    // Expected shape - [N, C, H, W]
    let (B, T, C) = (shape[0], shape[1], shape[2]);
    for b in 0..B {
        for t in 0..T {
            let idx = b * T * C + t * C;

            // Calculate the mean.
            let mut mean = 0.0;
            for i in 0..C {
                mean += input[i];
            }
            mean /= C as f32;

            // Calculate the variance.
            let mut var = 0.0;
            for i in 0..C {
                let shift = input[i] - mean;
                var += shift * shift;
            }
            var /= C as f32;

            // Calculate the reciprocal std deviation.
            let rstd_dev = 1.0 / f32::sqrt(var + 1.0e-5);

            for i in 0..C {
                // Normalise (with shift).
                let norm = rstd_dev * (input[idx] - mean);
                // Scale (weight) and shift (bias).
                out[idx] = norm * weight[i] + bias[i];
            }
            means[b * T + t] = mean;
            rstds[b * T + t] = rstd_dev;
        }
    }
}
