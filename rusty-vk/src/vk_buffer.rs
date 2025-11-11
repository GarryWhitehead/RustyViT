use crate::Driver;
use crate::vk_commands::CmdBuffer;
use ash::vk;
use vk_mem::Alloc;

pub enum BufferType {
    HostToGpu,
    GpuToHost,
    GpuOnly,
}

#[derive(Clone)]
pub struct Buffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk_mem::Allocation,
    pub(crate) size: vk::DeviceSize,
    pub(crate) _frames_until_gc: u32,
}

impl Buffer {
    pub fn new(
        size: vk::DeviceSize,
        allocator: &vk_mem::Allocator,
        usage: vk::BufferUsageFlags,
        buffer_type: BufferType,
    ) -> Self {
        let buffer_create_info = vk::BufferCreateInfo::default()
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | usage)
            .size(size);
        let mut alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::Auto,
            ..Default::default()
        };
        match buffer_type {
            BufferType::HostToGpu => {
                alloc_info.flags = vk_mem::AllocationCreateFlags::MAPPED
                    | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;
            }
            BufferType::GpuToHost => {
                alloc_info.flags = vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM
                    | vk_mem::AllocationCreateFlags::MAPPED;
                alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
            }
            BufferType::GpuOnly => alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice,
        }

        let (buffer, allocation) = unsafe {
            allocator
                .create_buffer(&buffer_create_info, &alloc_info)
                .unwrap()
        };

        Self {
            buffer,
            memory: allocation,
            size,
            _frames_until_gc: 0,
        }
    }

    pub fn map(
        &mut self,
        data: *const u8,
        data_size: vk::DeviceSize,
        offset: vk::DeviceSize,
        allocator: &vk_mem::Allocator,
    ) {
        assert!(data_size + offset <= self.size);
        let mapped = unsafe { allocator.map_memory(&mut self.memory).unwrap() };
        unsafe { mapped.copy_from(data, data_size as usize) };
        unsafe { allocator.unmap_memory(&mut self.memory) };
        allocator
            .flush_allocation(&self.memory, offset, data_size)
            .unwrap();
    }

    pub fn map_to_host<T: num::Zero + Clone>(
        &mut self,
        cmds: &CmdBuffer,
        driver: &mut Driver,
    ) -> Vec<T> {
        let barrier = vk::MemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::HOST_READ,
            ..Default::default()
        };
        unsafe {
            driver.device.device.cmd_pipeline_barrier(
                cmds.buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            )
        };
        driver.compute_commands.flush(&driver.device.device);
        unsafe {
            driver
                .device
                .device
                .wait_for_fences(&[cmds.fence], true, u64::MAX)
        }
        .unwrap();

        let host_vec: Vec<T> = vec![T::zero(); self.size as usize / size_of::<T>()];
        let mapped = unsafe { driver.vma_allocator.map_memory(&mut self.memory).unwrap() };

        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped,
                host_vec.as_ptr() as *mut u8,
                self.size as usize / size_of::<T>(),
            );
            driver.vma_allocator.unmap_memory(&mut self.memory);
            driver
                .vma_allocator
                .flush_allocation(&self.memory, 0, self.size)
                .unwrap()
        };
        host_vec
    }

    pub fn destroy(&mut self, allocator: &vk_mem::Allocator) {
        unsafe { allocator.destroy_buffer(self.buffer, &mut self.memory) };
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }
}
