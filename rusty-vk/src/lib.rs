use crate::descriptor_cache::{DescImage, DescriptorCache};
use crate::pipeline_cache::PipelineCache;
use crate::public_types::*;
use crate::resource_cache::ResourceCache;
use crate::sampler_cache::SamplerCache;
use crate::staging_pool::StagingPool;
use crate::vk_buffer::BufferType;
use crate::vk_commands::Commands;
use crate::vk_device::ContextDevice;
use crate::vk_instance::ContextInstance;
use crate::vk_shader::ShaderProgram;
use crate::vk_texture::{TextureInfo, TextureType};
use ash::vk;
use env_logger::Env;
use renderdoc::{CaptureOption, RenderDoc, V130};
use std::{error::Error, mem::ManuallyDrop, ptr::null};

mod descriptor_cache;
pub mod public_types;
mod resource_cache;
mod sampler_cache;
mod staging_pool;
mod vk_buffer;
mod vk_commands;
mod vk_device;
mod vk_handle;
mod vk_instance;
pub mod vk_shader;
mod vk_texture;

pub(crate) mod pipeline_cache;

pub struct Driver {
    pub(crate) device: ContextDevice,
    pub(crate) instance: ContextInstance,
    pub(crate) vma_allocator: ManuallyDrop<vk_mem::Allocator>,
    pub(crate) compute_commands: Commands,
    pub(crate) staging_pool: StagingPool,
    pub(crate) resource_cache: ResourceCache,
    pub(crate) descriptor_cache: DescriptorCache,
    pub(crate) pline_cache: PipelineCache,
    pub(crate) sampler_cache: SamplerCache,
    pub(crate) current_frame: u64,
    pub renderdoc: Option<RenderDoc<V130>>,
}

impl Driver {
    /// Create a new Vulkan driver instance based on the specified window.
    pub fn new(device_type: DeviceType) -> Result<Self, Box<dyn Error>> {
        let env = Env::default()
            .filter_or("MY_LOG_LEVEL", "trace")
            .write_style_or("MY_LOG_STYLE", "always");
        env_logger::init_from_env(env);

        let renderd: Result<RenderDoc<V130>, _> = RenderDoc::new();
        let rd = match renderd {
            Ok(mut r) => {
                r.set_capture_option_u32(CaptureOption::DebugOutputMute, 0);
                r.set_capture_option_u32(CaptureOption::ApiValidation, 1);
                Some(r)
            }
            Err(_) => None,
        };

        // Create the main vulkan instance for a given set of display extensions.
        let instance = ContextInstance::new()?;
        let device = ContextDevice::new(&instance, device_type.to_vk())?;

        // Create the VMA allocator.
        let mut create_info = vk_mem::AllocatorCreateInfo::new(
            &instance.instance,
            &device.device,
            device.physical_device,
        );
        create_info.vulkan_api_version = vk::make_api_version(0, 1, 3, 0);
        let vma_allocator = unsafe { ManuallyDrop::new(vk_mem::Allocator::new(create_info)?) };

        let compute_commands = Commands::new(
            device.compute_queue_idx,
            device.compute_queue,
            &device.device,
        );
        let descriptor_cache = DescriptorCache::new(&device.device);

        Ok(Self {
            device,
            instance,
            vma_allocator,
            compute_commands,
            staging_pool: StagingPool::new(),
            resource_cache: ResourceCache::new(),
            descriptor_cache,
            pline_cache: PipelineCache::new(),
            sampler_cache: SamplerCache::new(),
            current_frame: 0,
            renderdoc: rd,
        })
    }

    pub fn allocate_ubo(&mut self, size: usize) -> UniformBuffer {
        let handle = self.resource_cache.create_buffer(
            size as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &self.vma_allocator,
            BufferType::HostToGpu,
        );
        UniformBuffer::new(handle, size as vk::DeviceSize)
    }

    pub fn allocate_ssbo<T>(&mut self, size: usize) -> StorageBuffer<T> {
        let handle = self.resource_cache.create_buffer(
            (std::mem::size_of::<T>() * size) as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &self.vma_allocator,
            BufferType::GpuToHost,
        );
        StorageBuffer::new(handle, size)
    }

    pub fn allocate_image_buffer<T: FormatToVk>(
        &mut self,
        width: u32,
        height: u32,
        mips: u32,
        format: ImageFormat<T>,
        border_mode: SamplerAddressMode,
    ) -> VkTexture<T> {
        let info = TextureInfo {
            width,
            height,
            mip_levels: mips,
            array_layers: 1,
            format: format.to_vk(),
            texture_type: TextureType::Texture2d,
        };
        let sampler = SamplerInfo {
            addr_mode_u: border_mode,
            addr_mode_v: border_mode,
            addr_mode_w: border_mode,
            min_filter: SamplerFilter::Linear,
            mag_filter: SamplerFilter::Linear,
            ..SamplerInfo::default()
        };
        let handle = self.resource_cache.create_texture2d(
            &info,
            vk::ImageUsageFlags::empty(),
            &sampler,
            &self.vma_allocator,
            &self.device.device,
            &mut self.sampler_cache,
        );
        VkTexture::new(handle)
    }

    pub fn try_map_ssbo<T>(
        &self,
        data: &[T],
        ssbo: &StorageBuffer<T>,
    ) -> Result<(), Box<dyn Error>> {
        if data.len() > ssbo.elements {
            return Err("Out of bounds slice.".into());
        }
        let mut buffer = self.resource_cache.get_buffer(&ssbo.handle);
        let parts = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };

        buffer.map(
            parts.as_ptr(),
            parts.len() as vk::DeviceSize,
            0,
            &self.vma_allocator,
        );
        Ok(())
    }

    pub fn map_ubo(&self, data: &[u8], ubo: UniformBuffer) {
        let mut buffer = self.resource_cache.get_buffer(&ubo.handle);
        buffer.map(
            data.as_ptr(),
            data.len() as vk::DeviceSize,
            0,
            &self.vma_allocator,
        );
    }

    pub fn map_ssbo_to_host<T: num::Zero + Clone>(&mut self, ssbo: &StorageBuffer<T>) -> Vec<T> {
        let cmds = self.compute_commands.get(&self.device.device);
        let mut buffer = self.resource_cache.get_buffer(&ssbo.handle);
        buffer.map_to_host(&cmds, self)
    }

    pub fn map_ssbo_to_ssbo<T: num::Zero + Clone>(
        &mut self,
        src: &StorageBuffer<T>,
        dst: &StorageBuffer<T>,
    ) {
        let cmds = self.compute_commands.get(&self.device.device);
        let src_buffer = self.resource_cache.get_buffer(&src.handle).buffer;
        let dst_buffer = self.resource_cache.get_buffer(&dst.handle).buffer;

        let copy = vk::BufferCopy {
            size: (src.elements * std::mem::size_of::<T>()) as vk::DeviceSize,
            ..Default::default()
        };
        unsafe {
            self.device
                .device
                .cmd_copy_buffer(cmds.buffer, src_buffer, dst_buffer, &[copy])
        };
    }

    pub fn memset_zero<T>(&mut self, ssbo: &StorageBuffer<T>) {
        let buffer = self.resource_cache.get_buffer(&ssbo.handle);
        let cmds = self.compute_commands.get(&self.device.device);
        unsafe {
            self.device
                .device
                .cmd_fill_buffer(cmds.buffer, buffer.buffer, 0, vk::WHOLE_SIZE, 0)
        };
    }

    pub fn dispatch_compute(
        &mut self,
        program: &ShaderProgram,
        work: &ComputeWork,
    ) -> Result<(), Box<dyn Error>> {
        if let Some(rd) = &mut self.renderdoc {
            rd.start_frame_capture(null(), null());
        }

        let cmds = self.compute_commands.get(&self.device.device);

        self.pline_cache
            .bind_desc_layouts(&program.desc_set_layouts);
        let layout = self
            .pline_cache
            .get_layout(self.current_frame, &self.device.device);

        // Bind the valid storage images, ubo's and ssbo's to the descriptor cache.
        program.ssbos.iter().enumerate().for_each(|(idx, ssbo)| {
            if let Some(ssbo) = ssbo {
                let sz = (ssbo.end - ssbo.start) as vk::DeviceSize;
                let buffer = self.resource_cache.get_buffer(&ssbo.buffer);
                self.descriptor_cache.bind_ssbo(
                    idx,
                    buffer.buffer,
                    ssbo.start as vk::DeviceSize,
                    sz,
                );
            }
        });
        program.ubos.iter().enumerate().for_each(|(idx, ubo)| {
            if let Some(ubo) = ubo {
                let buffer = self.resource_cache.get_buffer(&ubo.buffer);
                self.descriptor_cache
                    .bind_ubo(idx, buffer.buffer, ubo.end as vk::DeviceSize, 0);
            }
        });
        program
            .storage_images
            .iter()
            .enumerate()
            .for_each(|(idx, image)| {
                if let Some(image) = image {
                    let image = self.resource_cache.get_texture2d(&image.texture);
                    self.descriptor_cache.bind_storage_image(
                        &DescImage::new(image.image_views[0], image.image_layout, image.sampler),
                        idx,
                    );
                }
            });
        self.descriptor_cache.bind_descriptors(
            program,
            &cmds.buffer,
            &layout.layout,
            &self.device.device,
            self.current_frame,
        )?;

        self.pline_cache.bind_shader_module(program.module);
        self.pline_cache.bind_layout(layout.layout);
        self.pline_cache
            .bind_pipeline(cmds.buffer, program, &self.device.device);

        unsafe {
            self.device
                .device
                .cmd_dispatch(cmds.buffer, work.x, work.y, work.z)
        };
        if let Some(rd) = &mut self.renderdoc {
            rd.end_frame_capture(null(), null());
        }
        Ok(())
    }

    pub fn sync_cmds(&mut self) {
        self.compute_commands.free_cmd_buffers(&self.device.device);
    }

    pub fn flush_cmds(&mut self) {
        self.compute_commands.flush(&self.device.device);
    }

    pub fn get_current_frame(&self) -> u64 {
        self.current_frame
    }

    pub fn write_read_barrier(&mut self) {
        let cmds = self.compute_commands.get(&self.device.device);
        let mem_barrier = vk::MemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            ..Default::default()
        };
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                cmds.buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[mem_barrier],
                &[],
                &[],
            )
        };
    }

    pub fn is_depth_format(format: &vk::Format) -> bool {
        let depth_formats = [
            vk::Format::D16_UNORM,
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
            vk::Format::D16_UNORM_S8_UINT,
            vk::Format::X8_D24_UNORM_PACK32,
        ];
        depth_formats.contains(format)
    }

    pub fn is_stencil_format(format: &vk::Format) -> bool {
        let stencil_formats = [
            vk::Format::S8_UINT,
            vk::Format::D16_UNORM_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
            vk::Format::D32_SFLOAT_S8_UINT,
        ];
        stencil_formats.contains(format)
    }

    pub fn destroy(&mut self) {
        // Manually destroy all objects as relying on RAII for this seems too risky.
        self.compute_commands.destroy(&self.device.device);
        self.sampler_cache.destroy(&self.device.device);
        self.staging_pool.destroy(&self.vma_allocator);
        self.resource_cache
            .destroy(&self.vma_allocator, &self.device.device);
        self.descriptor_cache.destroy(&self.device.device);
        self.pline_cache.destroy(&self.device.device);
        // Manually dropping the VMA allocator to ensure its lifetime outlives
        // that of the staging pool and resources.
        unsafe { ManuallyDrop::drop(&mut self.vma_allocator) };
        self.device.destroy();
        self.instance.destroy();
    }
}
