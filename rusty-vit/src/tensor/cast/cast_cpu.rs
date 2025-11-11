use crate::device::cpu::Cpu;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;

impl<T: FloatType, O: FloatType> super::CastKernel<T, O> for Cpu {
    fn cast(&mut self, lhs: &Tensor<T, Self>) -> Tensor<O, Self> {
        let mut out = Tensor::try_new(&lhs.shape, self).unwrap();
        out.data.data = lhs.data.iter().map(|x| O::from(*x).unwrap()).collect();
        out
    }
}
