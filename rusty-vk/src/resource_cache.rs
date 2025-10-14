use crate::public_types::SamplerInfo;
use crate::sampler_cache::SamplerCache;
use crate::vk_buffer::{Buffer, BufferType};
use crate::vk_commands::MAX_CMD_BUFFER_IN_FLIGHT_COUNT;
use crate::vk_handle::Handle;
use crate::vk_texture::{Texture, TextureInfo};
use ash::vk;

pub type BufferHandle = Handle<Buffer>;
pub type TextureHandle = Handle<Texture>;

pub struct ResourceCache {
    textures: Vec<Texture>,
    buffers: Vec<Buffer>,
    _textures_gc: Vec<Texture>,
    _buffers_gc: Vec<Buffer>,
}

impl ResourceCache {
    pub fn new() -> Self {
        Self {
            textures: Vec::with_capacity(100),
            buffers: Vec::with_capacity(100),
            _textures_gc: Vec::with_capacity(100),
            _buffers_gc: Vec::with_capacity(100),
        }
    }

    pub fn create_texture2d(
        &mut self,
        info: &TextureInfo,
        usage_flags: vk::ImageUsageFlags,
        sampler_params: &SamplerInfo,
        vma_alloc: &vk_mem::Allocator,
        device: &ash::Device,
        sampler_cache: &mut SamplerCache,
    ) -> TextureHandle {
        let texture = Texture::new(
            info,
            usage_flags,
            vma_alloc,
            device,
            sampler_cache,
            sampler_params,
        );
        let handle = TextureHandle::new(self.textures.len());
        self.textures.push(texture);
        handle
    }

    pub fn get_texture2d(&self, handle: &TextureHandle) -> &Texture {
        if handle.get_id() >= self.textures.len() {
            panic!("Handle is out of bounds");
        }
        &self.textures[handle.get_id()]
    }

    pub fn create_buffer(
        &mut self,
        size: vk::DeviceSize,
        usage_flags: vk::BufferUsageFlags,
        vma_alloc: &vk_mem::Allocator,
        buffer_type: BufferType,
    ) -> BufferHandle {
        let buffer = Buffer::new(size, vma_alloc, usage_flags, buffer_type);
        let handle = BufferHandle::new(self.buffers.len());
        self.buffers.push(buffer);
        handle
    }

    pub fn get_buffer(&self, handle: &BufferHandle) -> Buffer {
        if handle.get_id() >= self.buffers.len() {
            panic!("Handle is out of bounds");
        }
        self.buffers[handle.get_id()].clone()
    }

    #[allow(dead_code)]
    pub fn destroy_texture2d(&mut self, handle: TextureHandle) {
        assert!(handle.is_valid());
        let mut texture = self.textures.remove(handle.get_id());
        // Begin the countdown before this texture is destroyed - this will ensure
        // that the texture has finished on whatever command buffer it was related
        // to and can be safely terminated.
        texture.frames_until_gc = MAX_CMD_BUFFER_IN_FLIGHT_COUNT as u32;
        self._textures_gc.push(texture);
    }

    #[allow(dead_code)]
    pub fn destroy_buffer(&mut self, handle: BufferHandle) {
        assert!(handle.is_valid());
        let mut buffer = self.buffers.remove(handle.get_id());
        buffer._frames_until_gc = MAX_CMD_BUFFER_IN_FLIGHT_COUNT as u32;
        self._buffers_gc.push(buffer);
    }

    #[allow(dead_code)]
    pub fn gc(&mut self, vma_alloc: &vk_mem::Allocator, device: &ash::Device) {
        self._buffers_gc.retain_mut(|buffer| {
            buffer._frames_until_gc -= 1;
            if buffer._frames_until_gc == 0 {
                buffer.destroy(vma_alloc);
                return false;
            }
            true
        });
        self._textures_gc.retain_mut(|texture| {
            texture.frames_until_gc -= 1;
            if texture.frames_until_gc == 0 {
                texture.destroy(vma_alloc, device);
                return false;
            }
            true
        });
    }

    pub fn destroy(&mut self, vma_alloc: &vk_mem::Allocator, device: &ash::Device) {
        self.textures.iter_mut().for_each(|texture| {
            unsafe { vma_alloc.free_memory(&mut texture.vma_alloc) };
        });
        self.buffers.iter_mut().for_each(|buffer| {
            unsafe { vma_alloc.free_memory(&mut buffer.memory) };
            unsafe { device.destroy_buffer(buffer.buffer, None) };
        });
    }
}
