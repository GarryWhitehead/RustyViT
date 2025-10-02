use crate::vk_shader::{
    IMAGE_STORAGE_SHADER_BINDING, SSBO_SHADER_BINDING, ShaderProgram, UBO_SHADER_BINDING,
};
use ash::vk;
use ash::vk::Handle;
use std::collections::HashMap;

pub const MAX_UBO_COUNT: usize = 8;
pub const MAX_SSBO_COUNT: usize = 8;
pub const MAX_STORAGE_IMAGE_COUNT: usize = 8;
pub const MAX_DESC_SET_COUNT: usize = 3;

#[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct DescImage {
    image_view: vk::ImageView,
    image_layout: vk::ImageLayout,
    sampler: vk::Sampler,
}

impl DescImage {
    pub fn new(
        image_view: vk::ImageView,
        image_layout: vk::ImageLayout,
        sampler: vk::Sampler,
    ) -> Self {
        Self {
            image_view,
            image_layout,
            sampler,
        }
    }
}

#[derive(Default, Hash, PartialEq, Eq, Clone, Debug)]
pub struct DescriptorKey {
    ubos: [vk::Buffer; MAX_UBO_COUNT],
    ssbos: [vk::Buffer; MAX_SSBO_COUNT],
    storage_images: [DescImage; MAX_STORAGE_IMAGE_COUNT],
    ubo_sizes: [vk::DeviceSize; MAX_UBO_COUNT],
    ssbo_sizes: [vk::DeviceSize; MAX_SSBO_COUNT],
    ubo_offsets: [vk::DeviceSize; MAX_UBO_COUNT],
    ssbo_offsets: [vk::DeviceSize; MAX_SSBO_COUNT],
}

#[derive(Default, Clone, Debug)]
struct DescSetInstance {
    sets: [vk::DescriptorSet; MAX_DESC_SET_COUNT],
    layouts: [vk::DescriptorSetLayout; MAX_DESC_SET_COUNT],
    last_frame_used: u64,
}

impl DescSetInstance {
    pub fn new() -> Self {
        Self {
            sets: Default::default(),
            layouts: Default::default(),
            last_frame_used: 0,
        }
    }
}

#[derive(Debug, Default)]
pub struct DescriptorCache {
    desc_sets: HashMap<DescriptorKey, DescSetInstance>,
    desc_pool: vk::DescriptorPool,
    desc_pool_sz: usize,
    desc_set_pool: Vec<vk::DescriptorPool>,
    desc_requires: DescriptorKey,
}

impl DescriptorCache {
    pub fn new(device: &ash::Device) -> Self {
        Self {
            desc_sets: HashMap::new(),
            desc_pool: Self::create_desc_pool(1000, device),
            desc_pool_sz: 1000,
            desc_set_pool: vec![],
            desc_requires: Default::default(),
        }
    }

    pub fn bind_descriptors(
        &mut self,
        program: &ShaderProgram,
        cmds: &vk::CommandBuffer,
        pl_layout: &vk::PipelineLayout,
        device: &ash::Device,
        current_frame: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let s = self.desc_sets.get_mut(&self.desc_requires);
        let desc_sets = match s {
            Some(s) => {
                s.last_frame_used = current_frame;
                s.sets
            }
            None => {
                let mut instance = self.create_desc_set(program, device);
                instance.last_frame_used = current_frame;
                if self
                    .desc_sets
                    .insert(self.desc_requires.clone(), instance.clone())
                    .is_some()
                {
                    return Err("Unable to insert desc into cache".into());
                }
                instance.sets
            }
        };

        unsafe {
            device.cmd_bind_descriptor_sets(
                *cmds,
                vk::PipelineBindPoint::COMPUTE,
                *pl_layout,
                0,
                &desc_sets,
                &[],
            )
        };
        Ok(())
    }

    fn create_desc_set(&self, program: &ShaderProgram, device: &ash::Device) -> DescSetInstance {
        let mut instance = DescSetInstance::default();
        instance.layouts = program.desc_set_layouts;

        let mut ai = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.desc_pool,
            descriptor_set_count: 1,
            ..Default::default()
        };
        for i in 0..MAX_DESC_SET_COUNT {
            ai.p_set_layouts = &instance.layouts[i];
            instance.sets[i] = unsafe { device.allocate_descriptor_sets(&ai).unwrap()[0] };
        }

        let mut write_desc_sets: Vec<vk::WriteDescriptorSet> = Vec::new();
        let mut buffer_infos = [vk::DescriptorBufferInfo::default(); MAX_UBO_COUNT];
        let mut ssbo_infos = [vk::DescriptorBufferInfo::default(); MAX_SSBO_COUNT];
        let mut storage_image_infos = [vk::DescriptorImageInfo::default(); MAX_STORAGE_IMAGE_COUNT];

        for i in 0..MAX_UBO_COUNT {
            if !self.desc_requires.ubos[i].is_null() {
                buffer_infos[i].buffer = self.desc_requires.ubos[i];
                buffer_infos[i].offset = 0;
                buffer_infos[i].range = self.desc_requires.ubo_sizes[i];

                let mut write = vk::WriteDescriptorSet::default();
                write.descriptor_count = 1;
                write.descriptor_type = vk::DescriptorType::UNIFORM_BUFFER;
                write.p_buffer_info = &buffer_infos[i];
                write.dst_binding = i as u32;
                write.dst_set = instance.sets[UBO_SHADER_BINDING];
                write_desc_sets.push(write);
            }
        }
        for i in 0..MAX_SSBO_COUNT {
            if !self.desc_requires.ssbos[i].is_null() {
                ssbo_infos[i].buffer = self.desc_requires.ssbos[i];
                ssbo_infos[i].offset = 0;
                ssbo_infos[i].range = self.desc_requires.ssbo_sizes[i];

                let mut write = vk::WriteDescriptorSet::default();
                write.descriptor_count = 1;
                write.descriptor_type = vk::DescriptorType::STORAGE_BUFFER;
                write.p_buffer_info = &ssbo_infos[i];
                write.dst_binding = i as u32;
                write.dst_set = instance.sets[SSBO_SHADER_BINDING];
                write_desc_sets.push(write);
            }
        }
        for i in 0..MAX_STORAGE_IMAGE_COUNT {
            if !self.desc_requires.storage_images[i].image_view.is_null() {
                storage_image_infos[i].image_layout =
                    self.desc_requires.storage_images[i].image_layout;
                storage_image_infos[i].image_view = self.desc_requires.storage_images[i].image_view;
                storage_image_infos[i].sampler = self.desc_requires.storage_images[i].sampler;

                let mut write = vk::WriteDescriptorSet::default();
                write.descriptor_count = 1;
                write.descriptor_type = vk::DescriptorType::STORAGE_IMAGE;
                write.p_image_info = &storage_image_infos[i];
                write.dst_binding = i as u32;
                write.dst_set = instance.sets[IMAGE_STORAGE_SHADER_BINDING];
                write_desc_sets.push(write);
            }
        }
        unsafe { device.update_descriptor_sets(&write_desc_sets, &[]) };
        instance
    }

    fn create_desc_pool(pool_size: usize, device: &ash::Device) -> vk::DescriptorPool {
        let mut pools = [vk::DescriptorPoolSize::default(); MAX_DESC_SET_COUNT];
        pools[0].ty = vk::DescriptorType::UNIFORM_BUFFER;
        pools[0].descriptor_count = (pool_size * MAX_UBO_COUNT) as u32;
        pools[1].ty = vk::DescriptorType::STORAGE_BUFFER;
        pools[1].descriptor_count = (pool_size * MAX_SSBO_COUNT) as u32;
        pools[2].ty = vk::DescriptorType::STORAGE_IMAGE;
        pools[2].descriptor_count = (pool_size * MAX_STORAGE_IMAGE_COUNT) as u32;

        let ci = vk::DescriptorPoolCreateInfo {
            flags: vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND
                | vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            max_sets: (pool_size * MAX_DESC_SET_COUNT) as u32,
            pool_size_count: MAX_DESC_SET_COUNT as u32,
            p_pool_sizes: pools.as_ptr(),
            ..Default::default()
        };
        unsafe { device.create_descriptor_pool(&ci, None).unwrap() }
    }

    pub fn bind_storage_image(&mut self, image: &DescImage, idx: usize) {
        self.desc_requires.storage_images[idx] = image.clone();
    }

    pub fn bind_ubo(
        &mut self,
        bind: usize,
        buffer: vk::Buffer,
        size: vk::DeviceSize,
        offset: vk::DeviceSize,
    ) {
        self.desc_requires.ubos[bind] = buffer;
        self.desc_requires.ubo_sizes[bind] = size;
        self.desc_requires.ubo_offsets[bind] = offset;
    }

    pub fn bind_ssbo(
        &mut self,
        bind: usize,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) {
        self.desc_requires.ssbos[bind] = buffer;
        self.desc_requires.ssbo_sizes[bind] = size;
        self.desc_requires.ssbo_offsets[bind] = offset;
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_descriptor_pool(self.desc_pool, None) };
    }
}
