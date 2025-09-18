use rayon::prelude::*;

fn matmul(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    shape: &[usize; 4],
    out: &mut [f32],
) {
    let (_, T, C, OC) = (shape[0], shape[1], shape[2], shape[3]);

    let out_iter = out.par_chunks_mut(T * OC);
    let in_iter = input.par_chunks(T * C);

    out_iter
        .zip(in_iter)
        .enumerate()
        .for_each(|(_, (out_b, in_b))| {
            for t in 0..T {
                let in_bt = &in_b[t * C..(t + 1) * C];
                let out_bt = &mut out_b[t * OC..(t + 1) * OC];

                for o in 0..OC {
                    let mut val = bias.map_or(0.0, |b| b[o]);
                    let w_row = &weight[o * C..(o + 1) * C];
                    for i in 0..C {
                        val += in_bt[i] * w_row[i];
                    }
                    out_bt[o] = val;
                }
            }
        });
}
