use rayon::prelude::*;

#[allow(dead_code)]
fn matmul(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    shape: &[usize; 4],
    out: &mut [f32],
) {
    let (_, t, c, oc) = (shape[0], shape[1], shape[2], shape[3]);

    let out_iter = out.par_chunks_mut(t * oc);
    let in_iter = input.par_chunks(t * c);

    out_iter
        .zip(in_iter)
        .enumerate()
        .for_each(|(_, (out_b, in_b))| {
            for _t in 0..t {
                let in_bt = &in_b[_t * c..(_t + 1) * c];
                let out_bt = &mut out_b[_t * oc..(_t + 1) * oc];

                for o in 0..oc {
                    let mut val = bias.map_or(0.0, |b| b[o]);
                    let w_row = &weight[o * c..(o + 1) * c];
                    for i in 0..c {
                        val += in_bt[i] * w_row[i];
                    }
                    out_bt[o] = val;
                }
            }
        });
}
