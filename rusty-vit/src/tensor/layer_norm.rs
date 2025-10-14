#[allow(dead_code)]
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
    let (b, t, c) = (shape[0], shape[1], shape[2]);
    for _b in 0..b {
        for _t in 0..t {
            let idx = _b * t * c + _t * c;

            // Calculate the mean.
            let mut mean = 0.0;
            for i in 0..c {
                mean += input[i];
            }
            mean /= c as f32;

            // Calculate the variance.
            let mut var = 0.0;
            for i in 0..c {
                let shift = input[i] - mean;
                var += shift * shift;
            }
            var /= c as f32;

            // Calculate the reciprocal std deviation.
            let rstd_dev = 1.0 / f32::sqrt(var + 1.0e-5);

            for i in 0..c {
                // Normalise (with shift).
                let norm = rstd_dev * (input[idx] - mean);
                // Scale (weight) and shift (bias).
                out[idx] = norm * weight[i] + bias[i];
            }
            means[_b * t + _t] = mean;
            rstds[_b * t + _t] = rstd_dev;
        }
    }
}
