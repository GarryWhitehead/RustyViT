use crate::resource_cache::{BufferHandle, TextureHandle};
use ash::vk;
use std::{
    marker::PhantomData,
    ops::{Bound, RangeBounds},
};

pub trait FormatToVk {
    fn to_vk(v: u32) -> vk::Format;
}
pub struct ImageFormat<T>(pub(crate) u32, PhantomData<T>);

impl FormatToVk for u8 {
    fn to_vk(v: u32) -> vk::Format {
        match v {
            0 => vk::Format::R8_UNORM,
            1 => vk::Format::R8G8B8_UNORM,
            2 => vk::Format::R8G8B8A8_UNORM,
            _ => vk::Format::UNDEFINED,
        }
    }
}
impl FormatToVk for u16 {
    fn to_vk(v: u32) -> vk::Format {
        match v {
            0 => vk::Format::R16_UNORM,
            1 => vk::Format::R16G16B16_UNORM,
            2 => vk::Format::R16G16B16A16_UNORM,
            _ => vk::Format::UNDEFINED,
        }
    }
}
impl FormatToVk for f32 {
    fn to_vk(v: u32) -> vk::Format {
        match v {
            0 => vk::Format::R32_SFLOAT,
            1 => vk::Format::R32G32B32_SFLOAT,
            2 => vk::Format::R32G32B32A32_SFLOAT,
            _ => vk::Format::UNDEFINED,
        }
    }
}

impl<T> ImageFormat<T> {
    pub const MONO: Self = Self(0, PhantomData);
    pub const RGB: Self = Self(1, PhantomData);
    pub const RGBA: Self = Self(2, PhantomData);
}
impl<T: FormatToVk> ImageFormat<T> {
    pub(crate) fn to_vk(&self) -> vk::Format {
        T::to_vk(self.0)
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum SamplerAddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
    MirrorClampToEdge,
}

impl SamplerAddressMode {
    pub fn to_vk(&self) -> vk::SamplerAddressMode {
        match self {
            SamplerAddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            SamplerAddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            SamplerAddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            SamplerAddressMode::MirrorClampToEdge => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
            SamplerAddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        }
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum SamplerFilter {
    Nearest,
    Linear,
    Cubic,
}

impl SamplerFilter {
    pub fn to_vk(&self) -> vk::Filter {
        match self {
            SamplerFilter::Nearest => vk::Filter::NEAREST,
            SamplerFilter::Linear => vk::Filter::LINEAR,
            SamplerFilter::Cubic => vk::Filter::CUBIC_EXT,
        }
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct SamplerInfo {
    pub min_filter: SamplerFilter,
    pub mag_filter: SamplerFilter,
    pub addr_mode_u: SamplerAddressMode,
    pub addr_mode_v: SamplerAddressMode,
    pub addr_mode_w: SamplerAddressMode,
    pub compare_op: vk::CompareOp,
    pub anisotropy: u32,
    pub mip_levels: u32,
    pub enable_compare: vk::Bool32,
    pub enable_anisotropy: vk::Bool32,
}

impl Default for SamplerInfo {
    fn default() -> Self {
        Self {
            min_filter: SamplerFilter::Linear,
            mag_filter: SamplerFilter::Linear,
            addr_mode_u: SamplerAddressMode::ClampToEdge,
            addr_mode_v: SamplerAddressMode::ClampToEdge,
            addr_mode_w: SamplerAddressMode::ClampToEdge,
            compare_op: vk::CompareOp::NEVER,
            anisotropy: 1,
            mip_levels: 1,
            enable_compare: vk::FALSE,
            enable_anisotropy: vk::TRUE,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DeviceType {
    InregratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
}

impl DeviceType {
    pub fn to_vk(self) -> vk::PhysicalDeviceType {
        match self {
            DeviceType::InregratedGpu => vk::PhysicalDeviceType::INTEGRATED_GPU,
            DeviceType::DiscreteGpu => vk::PhysicalDeviceType::DISCRETE_GPU,
            DeviceType::VirtualGpu => vk::PhysicalDeviceType::VIRTUAL_GPU,
            DeviceType::Cpu => vk::PhysicalDeviceType::CPU,
        }
    }
}

fn to_range(range: impl RangeBounds<usize>, size: usize) -> Option<(usize, usize)> {
    let start = match range.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&n) => n + 1,
        Bound::Excluded(&n) => n,
        Bound::Unbounded => size,
    };
    if end <= size {
        return Some((start, end));
    }
    None
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BufferView {
    pub(crate) buffer: BufferHandle,
    pub(crate) start: usize,
    pub(crate) end: usize,
}
impl BufferView {
    pub fn new(buffer: BufferHandle, start: usize, end: usize) -> Self {
        Self { buffer, start, end }
    }
}

#[derive(Debug, Default, Clone)]
pub struct TextureView {
    pub(crate) texture: TextureHandle,
}
impl TextureView {
    pub fn new(texture: TextureHandle) -> Self {
        Self { texture }
    }
}

#[derive(Copy, Clone)]
pub struct UniformBuffer {
    pub(crate) handle: BufferHandle,
    pub(crate) byte_size: vk::DeviceSize,
}

impl UniformBuffer {
    pub fn new(handle: BufferHandle, byte_size: vk::DeviceSize) -> UniformBuffer {
        UniformBuffer { handle, byte_size }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StorageBuffer<T> {
    pub(crate) handle: BufferHandle,
    pub(crate) elements: usize,
    pub(crate) phantom: PhantomData<T>,
}

impl<T> StorageBuffer<T> {
    pub fn new(handle: BufferHandle, elements: usize) -> Self {
        Self {
            handle,
            elements,
            phantom: PhantomData,
        }
    }

    pub fn element_count(&self) -> usize {
        self.elements
    }

    pub fn slice(&self, range: impl RangeBounds<usize>) -> Option<BufferView> {
        let t = to_range(range, self.elements);
        match t {
            Some((start, end)) => Some(BufferView::new(self.handle, start, end)),
            _ => None,
        }
    }
}

#[derive(Copy, Clone)]
pub struct VkTexture<T> {
    pub(crate) _handle: TextureHandle,
    pub(crate) phantom: PhantomData<T>,
}

impl<T> VkTexture<T> {
    pub fn new(handle: TextureHandle) -> Self {
        Self {
            _handle: handle,
            phantom: PhantomData,
        }
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct ComputeWork {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) z: u32,
}

impl ComputeWork {
    pub fn default() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }

    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}
