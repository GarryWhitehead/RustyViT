use crate::Driver;
use crate::descriptor_cache::*;
use crate::public_types::{BufferView, TextureView, UniformBuffer};
use crate::resource_cache::TextureHandle;
use ash::vk;
use ash::vk::{DescriptorPool, ShaderModule};
use rspirv_reflect::Reflection;
use std::{collections::HashMap, error::Error, ops::Range};

pub const UBO_SHADER_BINDING: usize = 0;
pub const SSBO_SHADER_BINDING: usize = 1;
pub const IMAGE_STORAGE_SHADER_BINDING: usize = 2;

#[derive(Debug, Clone, Copy)]
pub struct WorkSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Default for WorkSize {
    fn default() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }
}

#[derive(Debug, Clone)]
pub struct BindInfo {
    binding: u32,
    desc_type: rspirv_reflect::DescriptorType,
}
impl BindInfo {
    pub fn new(binding: u32, desc_type: rspirv_reflect::DescriptorType) -> Self {
        Self { binding, desc_type }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SpecConstants {
    pub(crate) data: Vec<u8>,
    pub(crate) mappings: Vec<vk::SpecializationMapEntry>,
}

#[derive(Debug, Default, Clone)]
pub struct ShaderProgram {
    pub(crate) binding_map: HashMap<String, BindInfo>,
    pub(crate) desc_set_layouts: [vk::DescriptorSetLayout; MAX_DESC_SET_COUNT],
    pub(crate) module: vk::ShaderModule,
    pub(crate) ubos: [Option<BufferView>; MAX_UBO_COUNT],
    pub(crate) ssbos: [Option<BufferView>; MAX_SSBO_COUNT],
    pub(crate) storage_images: [Option<TextureView>; MAX_STORAGE_IMAGE_COUNT],
    pub(crate) spec_consts: SpecConstants,
    pub(crate) work_size: WorkSize,
}

impl<'a> ShaderProgram {
    pub fn try_new(spirv_bytes: &[u8], driver: &Driver) -> Result<Self, Box<dyn Error>> {
        let (layout_bindings, binding_map, work_size) = Self::reflect_spirv(spirv_bytes)?;
        let mut desc_set_layouts: [vk::DescriptorSetLayout; MAX_DESC_SET_COUNT] =
            [Default::default(); MAX_DESC_SET_COUNT];

        // Create the descriptor set layouts for use later by the descriptor cache.
        for idx in 0..MAX_DESC_SET_COUNT {
            let b = layout_bindings.get(&(idx as u32));
            let mut ci = vk::DescriptorSetLayoutCreateInfo::default();
            if let Some(binding) = b {
                ci.binding_count = binding.len() as u32;
                ci.p_bindings = binding.as_ptr()
            };
            let layout = unsafe {
                driver
                    .device
                    .device
                    .create_descriptor_set_layout(&ci, None)?
            };
            desc_set_layouts[idx] = layout;
        }

        let mut spirv_words: Vec<u32> = Vec::new();
        for i in (0..spirv_bytes.len()).step_by(4) {
            let word_slice: u32 = u32::from_le_bytes(spirv_bytes[i..i + 4].try_into().unwrap());
            unsafe { spirv_words.push(word_slice) };
        }
        let module = Self::create_shader_module(&spirv_words, &driver.device.device)?;
        Ok(ShaderProgram {
            binding_map,
            desc_set_layouts,
            module,
            ubos: [const { None }; MAX_UBO_COUNT],
            ssbos: [const { None }; MAX_SSBO_COUNT],
            storage_images: [const { None }; MAX_STORAGE_IMAGE_COUNT],
            spec_consts: SpecConstants::default(),
            work_size,
        })
    }

    pub fn reflect_spirv(
        data: &[u8],
    ) -> Result<
        (
            HashMap<u32, Vec<vk::DescriptorSetLayoutBinding<'a>>>,
            HashMap<String, BindInfo>,
            WorkSize,
        ),
        Box<dyn Error>,
    > {
        let reflect = Reflection::new_from_spirv(data)?;
        let (wx, wy, wz) = reflect.get_compute_group_size().unwrap();
        let work_size = WorkSize {
            x: wx,
            y: wy,
            z: wz,
        };

        let mut layout_bindings: HashMap<u32, Vec<vk::DescriptorSetLayoutBinding<'a>>> =
            HashMap::new();
        let mut binding_map: HashMap<String, BindInfo> = HashMap::new();
        for (set, binding) in reflect.get_descriptor_sets().unwrap() {
            for (bind, info) in binding {
                let mut layout = vk::DescriptorSetLayoutBinding {
                    binding: bind,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    descriptor_count: 1,
                    ..Default::default()
                };
                match info.ty {
                    rspirv_reflect::DescriptorType::STORAGE_BUFFER => {
                        if set != SSBO_SHADER_BINDING as u32 {
                            return Err("Storage buffer must be set 1.".into());
                        }
                        layout.descriptor_type = vk::DescriptorType::STORAGE_BUFFER
                    }
                    rspirv_reflect::DescriptorType::UNIFORM_BUFFER => {
                        if set != UBO_SHADER_BINDING as u32 {
                            return Err("Uniform buffer must be set 0.".into());
                        }
                        layout.descriptor_type = vk::DescriptorType::UNIFORM_BUFFER
                    }
                    rspirv_reflect::DescriptorType::STORAGE_IMAGE => {
                        if set != IMAGE_STORAGE_SHADER_BINDING as u32 {
                            return Err("Storage images must be set 2.".into());
                        }
                        layout.descriptor_type = vk::DescriptorType::STORAGE_IMAGE
                    }
                    _ => {
                        panic!("Unsupported descriptor type.");
                    }
                }
                let v = layout_bindings.get_mut(&set);
                match v {
                    Some(value) => {
                        value.push(layout);
                    }
                    None => {
                        layout_bindings.insert(set, vec![layout]);
                    }
                }
                if info.name.is_empty() {
                    panic!("Binding name is empty - Vulkan backend wont behave as expected.")
                }
                binding_map.insert(info.name.clone(), BindInfo::new(bind, info.ty));
            }
        }
        Ok((layout_bindings, binding_map, work_size))
    }

    pub fn create_shader_module(
        spirv_bytecode: &[u32],
        device: &ash::Device,
    ) -> Result<vk::ShaderModule, Box<dyn Error>> {
        let ci = vk::ShaderModuleCreateInfo {
            code_size: spirv_bytecode.len() * std::mem::size_of::<u32>(),
            p_code: spirv_bytecode.as_ptr(),
            ..Default::default()
        };
        let module = unsafe { device.create_shader_module(&ci, None)? };
        Ok(module)
    }

    pub fn try_bind_ubo(
        &mut self,
        ubo_name: &str,
        buffer: &UniformBuffer,
    ) -> Result<(), Box<dyn Error>> {
        let binding = self.binding_map.get(ubo_name);
        match binding {
            Some(info) => {
                if info.desc_type != rspirv_reflect::DescriptorType::UNIFORM_BUFFER {
                    return Err("Binding not registered as a UBO.".into());
                }
                self.ubos[info.binding as usize] =
                    Some(BufferView::new(buffer.handle, 0, buffer.byte_size as usize));
            }
            None => {
                return Err("UBO Binding not found.".into());
            }
        }
        Ok(())
    }

    pub fn try_bind_ssbo<T>(
        &mut self,
        ssbo_name: &str,
        view: BufferView,
    ) -> Result<(), Box<dyn Error>> {
        let binding = self.binding_map.get(ssbo_name);
        match binding {
            Some(info) => {
                if info.desc_type != rspirv_reflect::DescriptorType::STORAGE_BUFFER {
                    return Err("Binding not registered as a SSBO.".into());
                }
                self.ssbos[info.binding as usize] = Some(view);
            }
            None => {
                return Err("SSBO Binding not found.".into());
            }
        }
        Ok(())
    }

    pub fn try_bind_image_storage(
        &mut self,
        image_name: &str,
        texture: TextureHandle,
        size: vk::DeviceSize,
        range: Range<usize>,
    ) -> Result<(), Box<dyn Error>> {
        let binding = self.binding_map.get(image_name);
        match binding {
            Some(info) => {
                if info.desc_type != rspirv_reflect::DescriptorType::STORAGE_IMAGE {
                    return Err("Binding not registered as a image storage type.".into());
                }
                self.storage_images[info.binding as usize] = Some(TextureView::new(texture, size));
            }
            None => {
                return Err("UBO Binding not found.".into());
            }
        }
        Ok(())
    }

    pub fn bind_spec_constant<T>(&mut self, id: u32, data: &[T]) {
        let parts =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, size_of::<T>()) };
        let mapping = vk::SpecializationMapEntry {
            constant_id: id,
            offset: self.spec_consts.data.len() as u32,
            size: parts.len(),
        };
        self.spec_consts.data.extend_from_slice(parts);
        self.spec_consts.mappings.push(mapping);
    }

    pub fn destroy(&mut self, driver: &Driver) {
        let device = &driver.device.device;
        for layout in self.desc_set_layouts.iter() {
            unsafe { device.destroy_descriptor_set_layout(*layout, None) };
        }
        unsafe { device.destroy_shader_module(self.module, None) };
    }

    pub fn get_work_size(&self) -> WorkSize {
        self.work_size
    }
}
