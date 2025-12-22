use crate::Device;
use crate::float_ops::TensorFloatOps;
use crate::tensor::{Float, Tensor, TensorType};
use rvit_core::element_traits::{DataElem, FloatElem};
use rvit_device::Device;

pub trait TensorOps<T: TensorType, D: Device> {
    fn add(&mut self, rhs: &mut Tensor<T, D>) -> Tensor<T, D>;

    fn sub(&mut self, rhs: &mut Tensor<T, D>) -> Tensor<T, D>;

    fn mul(&mut self, rhs: &mut Tensor<T, D>) -> Tensor<T, D>;

    fn div(&mut self, rhs: &mut Tensor<T, D>) -> Tensor<T, D>;

    fn sqr(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D>;

    fn sqrt(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D>;

    fn exp(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D>;

    fn tanh(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D>;

    fn cos(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D>;

    fn sin(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D>;

    fn abs(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D>;

    fn log(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D>;

    fn floor(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D>;

    fn ceil(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D>;

    fn relu(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D>;

    fn gelu(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D>;

    fn matmul(&mut self, rhs: &mut Tensor<Float, D>) -> Tensor<Float, D>;

    fn cast<OE: DataElem, OT: TensorType>(&mut self, x: &Tensor<Float, D>) -> Tensor<OT, D>;
}

impl<D: Device> TensorOps<Float, D> for Tensor<Float, D>
where
    D: TensorFloatOps<D>,
{
    fn add(&mut self, rhs: &mut Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_add(self, rhs.data, rhs.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &rhs.device)
    }

    fn sub(&mut self, rhs: &mut Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_sub(self, rhs.data, rhs.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &rhs.device)
    }

    fn mul(&mut self, rhs: &mut Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_mul(self, rhs.data, rhs.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &rhs.device)
    }

    fn div(&mut self, rhs: &mut Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_div(self, rhs.data, rhs.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &rhs.device)
    }

    fn sqr(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_sqr(self, x.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &x.device)
    }

    fn sqrt(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_sqrt(self, x.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &x.device)
    }

    fn exp(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_exp(self, x.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &x.device)
    }

    fn tanh(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_tanh(self, x.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &x.device)
    }

    fn cos(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_cos(self, x.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &x.device)
    }

    fn sin(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_sin(self, x.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &x.device)
    }

    fn abs(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_abs(self, x.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &x.device)
    }

    fn log(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_log(self, x.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &x.device)
    }

    fn floor(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_floor(self, x.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &x.device)
    }

    fn ceil(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_ceil(self, x.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &x.device)
    }

    fn relu(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_relu(self, x.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &x.device)
    }

    fn gelu(&mut self, x: &Tensor<Float, D>) -> Tensor<Float, D> {
        let prim = D::float_gelu(self, x.device);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &x.device)
    }

    fn matmul(&mut self, rhs: &mut Tensor<Float, D>) -> Tensor<Float, D> {
        if self.shape.len() < 2 || rhs.shape.len() < 2 {
            panic!("Tensor must have at least two dimensions (for now).");
        }
        if self.shape.len() != rhs.shape.len() {
            panic!("Tensors must have the same dimensions.");
        }
        if self.shape.len() > 2 && (self.shape[0] != rhs.shape[0]) {
            panic!("Tensors must have the same batch size");
        }

        let prim = D::float_mul(self, rhs);
        Tensor::<Float, D>::from_parts(prim.data, &prim.shape, &prim.strides, &rhs.device)
    }

    fn cast<OE: DataElem, OT: TensorType>(&mut self, x: &Tensor<Float, D>) -> Tensor<OT, D> {
        let prim = D::float_cast::<OE>(self);
        Tensor::<OT, D>::from_parts(prim.data, &prim.shape, &prim.strides, &x.device)
    }
}
