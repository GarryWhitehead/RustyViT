use crate::tensor::{Integer, Tensor, TensorType};
use crate::{Float, Device};
use rvit_core::element_traits::{DataElem, FloatElem};
use rvit_core::memory::arena::ArenaError;
use rvit_core::memory::storage::DeviceStorage;
use rvit_device::Device;
use std::error::Error;
use std::marker::PhantomData;

pub trait BasicOps<S: Device> {
    type Elem: DataElem;

    fn try_new(shape: &[usize], dev: &mut S::Storage) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized;

    fn new(shape: &[usize], dev: &mut S::Storage) -> Self;

    fn zeros(shape: &[usize], dev: &mut S::Storage) -> Self;

    fn try_from_array(
        shape: &[usize],
        values: &[Self::Elem],
        dev: &mut S::Storage,
    ) -> Result<Self, ArenaError>
    where
        Self: Sized;

    fn from_array(shape: &[usize], values: &[Self::Elem], dev: &mut S::Storage) -> Self;

    fn from_array_convert<O: DataElem>(shape: &[usize], values: &[O], dev: &mut S::Storage)
    -> Self;

    fn from_data_shape1<O: DataElem, const A: usize>(data: &[O; A], dev: &mut S::Storage) -> Self;

    fn from_data_shape2<O: DataElem, const A: usize, const B: usize>(
        data: &[[O; B]; A],
        dev: &mut S::Storage,
    ) -> Self;

    fn from_data_shape3<O: DataElem, const A: usize, const B: usize, const C: usize>(
        data: &[[[O; C]; B]; A],
        dev: &mut S::Storage,
    ) -> Self;
    fn from_data_shape4<
        O: DataElem,
        const A: usize,
        const B: usize,
        const C: usize,
        const D: usize,
    >(
        data: &[[[[O; D]; C]; B]; A],
        dev: &mut S::Storage,
    ) -> Self;

    fn try_into_vec(&self) -> Result<Vec<Self::Elem>, ArenaError>;

    fn into_vec(&self) -> Vec<Self::Elem>;
}

impl<S: Device> BasicOps<S> for Tensor<Float, S> {
    type Elem = S::FloatElem;

    fn try_new(shape: &[usize], dev: &mut S::Storage) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized,
    {
        try_new::<Self::Elem, _, _>(shape, dev)
    }

    fn new(shape: &[usize], dev: &mut S::Storage) -> Self {
        Self::try_new(shape, dev).unwrap()
    }

    fn zeros(shape: &[usize], dev: &mut S::Storage) -> Self {
        // All device platforms initialise allocated memory to zero.
        Self::new(&shape, dev)
    }

    fn try_from_array(
        shape: &[usize],
        values: &[Self::Elem],
        dev: &mut S::Storage,
    ) -> Result<Self, ArenaError>
    where
        Self: Sized,
    {
        try_from_array::<Self::Elem, _, _>(shape, values, dev)
    }

    fn from_array(shape: &[usize], values: &[Self::Elem], dev: &mut S::Storage) -> Self
    where
        Self: Sized,
    {
        Self::try_from_array(shape, values, dev).unwrap()
    }

    fn from_array_convert<O: DataElem>(
        shape: &[usize],
        values: &[O],
        dev: &mut S::Storage,
    ) -> Self {
        from_array_convert::<Self::Elem, O, _, _>(shape, values, dev)
    }

    fn from_data_shape1<O: DataElem, const A: usize>(data: &[O; A], dev: &mut S::Storage) -> Self {
        from_data_shape1::<Self::Elem, O, A, _, _>(data, dev)
    }

    fn from_data_shape2<O: DataElem, const A: usize, const B: usize>(
        data: &[[O; B]; A],
        dev: &mut S::Storage,
    ) -> Self {
        from_data_shape2::<Self::Elem, O, A, B, _, _>(data, dev)
    }

    fn from_data_shape3<O: DataElem, const A: usize, const B: usize, const C: usize>(
        data: &[[[O; C]; B]; A],
        dev: &mut S::Storage,
    ) -> Self {
        from_data_shape3::<Self::Elem, O, A, B, C, _, _>(data, dev)
    }

    fn from_data_shape4<
        O: DataElem,
        const A: usize,
        const B: usize,
        const C: usize,
        const D: usize,
    >(
        data: &[[[[O; D]; C]; B]; A],
        dev: &mut S::Storage,
    ) -> Self {
        from_data_shape4::<Self::Elem, O, A, B, C, D, _, _>(data, dev)
    }

    fn try_into_vec(&self) -> Result<Vec<Self::Elem>, ArenaError> {
        try_into_vec(self)
    }

    fn into_vec(&self) -> Vec<Self::Elem> {
        Self::try_into_vec(self).unwrap()
    }
}

impl<S: Device> BasicOps<S> for Tensor<Integer, S> {
    type Elem = S::IntElem;

    fn try_new(shape: &[usize], dev: &mut S::Storage) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized,
    {
        try_new::<Self::Elem, _, _>(shape, dev)
    }

    fn new(shape: &[usize], dev: &mut S::Storage) -> Self {
        Self::try_new(shape, dev).unwrap()
    }

    fn zeros(shape: &[usize], dev: &mut S::Storage) -> Self {
        // All device platforms initialise allocated memory to zero.
        Self::new(&shape, dev)
    }

    fn try_from_array(
        shape: &[usize],
        values: &[Self::Elem],
        dev: &mut S::Storage,
    ) -> Result<Self, ArenaError>
    where
        Self: Sized,
    {
        try_from_array::<Self::Elem, _, _>(shape, values, dev)
    }

    fn from_array(shape: &[usize], values: &[Self::Elem], dev: &mut S::Storage) -> Self
    where
        Self: Sized,
    {
        Self::try_from_array(shape, values, dev).unwrap()
    }

    fn from_array_convert<O: DataElem>(
        shape: &[usize],
        values: &[O],
        dev: &mut S::Storage,
    ) -> Self {
        from_array_convert::<Self::Elem, O, _, _>(shape, values, dev)
    }

    fn from_data_shape1<O: DataElem, const A: usize>(data: &[O; A], dev: &mut S::Storage) -> Self {
        from_data_shape1::<Self::Elem, O, A, _, _>(data, dev)
    }

    fn from_data_shape2<O: DataElem, const A: usize, const B: usize>(
        data: &[[O; B]; A],
        dev: &mut S::Storage,
    ) -> Self {
        from_data_shape2::<Self::Elem, O, A, B, _, _>(data, dev)
    }

    fn from_data_shape3<O: DataElem, const A: usize, const B: usize, const C: usize>(
        data: &[[[O; C]; B]; A],
        dev: &mut S::Storage,
    ) -> Self {
        from_data_shape3::<Self::Elem, O, A, B, C, _, _>(data, dev)
    }

    fn from_data_shape4<
        O: DataElem,
        const A: usize,
        const B: usize,
        const C: usize,
        const D: usize,
    >(
        data: &[[[[O; D]; C]; B]; A],
        dev: &mut S::Storage,
    ) -> Self {
        from_data_shape4::<Self::Elem, O, A, B, C, D, _, _>(data, dev)
    }

    fn try_into_vec(&self) -> Result<Vec<Self::Elem>, ArenaError> {
        try_into_vec(self)
    }

    fn into_vec(&self) -> Vec<Self::Elem> {
        Self::try_into_vec(self).unwrap()
    }
}

fn try_new<E: DataElem, T: TensorType, S: Device>(
    shape: &[usize],
    dev: &mut S::Storage,
) -> Result<Tensor<T, S>, Box<dyn Error>> {
    Ok(Tensor {
        data: S::Storage::try_alloc(
            dev,
            Tensor::<T, S>::compute_memory_size(shape),
            S::FloatElem::DTYPE,
        )?,
        device: dev.clone(),
        shape: shape.to_vec(),
        strides: Tensor::<T, S>::compute_strides(shape),
        phantom_data: PhantomData,
    })
}

fn try_from_array<E: DataElem, T: TensorType, S: Device>(
    shape: &[usize],
    values: &[E],
    dev: &mut S::Storage,
) -> Result<Tensor<T, S>, ArenaError> {
    let sz = Tensor::<T, S>::compute_memory_size(shape);
    if values.len() != sz {
        return Err(ArenaError::InvalidParameters {
            reason: "Shape size doesn't match values length".into(),
        });
    }

    Ok(Tensor {
        data: dev.try_alloc_with_data(values)?,
        device: dev.clone(),
        shape: shape.to_vec(),
        strides: Tensor::<T, S>::compute_strides(shape),
        phantom_data: Default::default(),
    })
}

fn from_array_convert<E: DataElem, O: DataElem, T: TensorType, S: Device>(
    shape: &[usize],
    values: &[O],
    dev: &mut S::Storage,
) -> Tensor<T, S> {
    let mut data: Vec<E> = Vec::with_capacity(values.len());
    for val in values.iter() {
        data.push(num_traits::cast::<O, E>(*val).unwrap());
    }
    try_from_array::<E, _, _>(shape, &data, dev).unwrap()
}

pub fn try_into_vec<E: DataElem, T: TensorType, S: Device>(
    x: &Tensor<T, S>,
) -> Result<Vec<E>, ArenaError> {
    x.device.try_sync()?;
    x.device.try_into_vec(&x.data)
}

pub fn from_data_shape1<
    E: DataElem,
    O: DataElem,
    const A: usize,
    T: TensorType,
    S: Device,
>(
    data: &[O; A],
    dev: &mut S::Storage,
) -> Tensor<T, S> {
    let mut out: Vec<E> = Vec::with_capacity(data.len());
    for d in data.iter() {
        out.push(num_traits::cast::<O, E>(*d).unwrap());
    }
    try_from_array(&[data.len()], &out, dev).unwrap()
}

pub fn from_data_shape2<
    E: DataElem,
    O: DataElem,
    const A: usize,
    const B: usize,
    T: TensorType,
    S: Device,
>(
    data: &[[O; B]; A],
    dev: &mut S::Storage,
) -> Tensor<T, S> {
    let mut out: Vec<E> = Vec::with_capacity(A * B);
    for d in data.iter().take(A) {
        for d in d.iter().take(B) {
            out.push(num_traits::cast::<O, E>(*d).unwrap());
        }
    }
    try_from_array(&[A, B], &out, dev).unwrap()
}

pub fn from_data_shape3<
    E: DataElem,
    O: DataElem,
    const A: usize,
    const B: usize,
    const C: usize,
    T: TensorType,
    S: Device,
>(
    data: &[[[O; C]; B]; A],
    dev: &mut S::Storage,
) -> Tensor<T, S> {
    let mut out: Vec<E> = Vec::with_capacity(A * B * C);
    for d in data.iter().take(A) {
        for d in d.iter().take(B) {
            for d in d.iter().take(C) {
                out.push(num_traits::cast::<O, E>(*d).unwrap());
            }
        }
    }
    try_from_array(&[A, B, C], &out, dev).unwrap()
}

pub fn from_data_shape4<
    E: DataElem,
    O: DataElem,
    const A: usize,
    const B: usize,
    const C: usize,
    const D: usize,
    T: TensorType,
    S: Device,
>(
    data: &[[[[O; D]; C]; B]; A],
    dev: &mut S::Storage,
) -> Tensor<T, S> {
    let mut out: Vec<E> = Vec::with_capacity(A * B * C * D);
    for d in data.iter().take(A) {
        for d in d.iter().take(B) {
            for d in d.iter().take(C) {
                for d in d.iter().take(D) {
                    out.push(num_traits::cast::<O, E>(*d).unwrap());
                }
            }
        }
    }
    try_from_array(&[A, B, C, D], &out, dev).unwrap()
}
