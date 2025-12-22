use crate::tensor::{Tensor, TensorType};
use rvit_core::element_traits::DataElem;
use rvit_device::Runtime;

type TensorIndex = Vec<usize>;

fn to_tensor_idx(index: TensorIndex, strides: &[usize]) -> usize {
    strides.iter().zip(index.iter()).map(|(&s, &i)| s * i).sum()
}

impl<F: TensorType + DataElem> std::ops::Index<TensorIndex> for Tensor<F, Runtime> {
    type Output = F;
    #[inline(always)]
    fn index(&self, index: TensorIndex) -> &Self::Output {
        let i = to_tensor_idx(index, &self.strides);
        &self.data.as_slice().unwrap()[i]
    }
}

impl<F: TensorType + DataElem> std::ops::IndexMut<TensorIndex> for Tensor<F, Runtime> {
    #[inline(always)]
    fn index_mut(&mut self, index: TensorIndex) -> &mut Self::Output {
        let i = to_tensor_idx(index, &self.strides);
        &mut self.data.as_mut_slice().unwrap()[i]
    }
}
