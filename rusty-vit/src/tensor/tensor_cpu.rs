use crate::device::cpu::Cpu;
use crate::tensor::{FloatType, Tensor};

type TensorIndex = Vec<usize>;

fn to_tensor_idx(index: TensorIndex, strides: &[usize]) -> usize {
    strides.iter().zip(index.iter()).map(|(&s, &i)| s * i).sum()
}

impl<F: FloatType> std::ops::Index<TensorIndex> for Tensor<F, Cpu> {
    type Output = F;
    #[inline(always)]
    fn index(&self, index: TensorIndex) -> &Self::Output {
        let i = to_tensor_idx(index, &self.strides);
        &self.data[i]
    }
}

impl<F: FloatType> std::ops::IndexMut<TensorIndex> for Tensor<F, Cpu> {
    #[inline(always)]
    fn index_mut(&mut self, index: TensorIndex) -> &mut Self::Output {
        let i = to_tensor_idx(index, &self.strides);
        &mut self.data[i]
    }
}
