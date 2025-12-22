use crate::element_traits::{FloatElem, IntElem};
use crate::memory::storage::DeviceStorage;

pub trait Device: Clone {
    type FloatElem: FloatElem;
    type IntElem: IntElem;
    type Storage: DeviceStorage;
}

pub type DAlloc<A> = <<A as Device>::Storage as DeviceStorage>::Alloc;
