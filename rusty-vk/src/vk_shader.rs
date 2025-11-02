use crate::Driver;
use crate::descriptor_cache::*;
use crate::public_types::{BufferView, TextureView, UniformBuffer};
use crate::resource_cache::TextureHandle;
use ash::vk;
use spirq::prelude::{ConstantValue, EntryPoint, Variable};
use spirq::ty::{AccessType, DescriptorType};
use spirq::*;
use std::{collections::HashMap, error::Error};

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
    desc_type: DescriptorType,
    access: Option<AccessType>,
}
impl BindInfo {
    pub fn new(binding: u32, desc_type: DescriptorType) -> Self {
        Self {
            binding,
            desc_type,
            access: None,
        }
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
        for (idx, ds_layout) in desc_set_layouts.iter_mut().enumerate() {
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
            *ds_layout = layout;
        }

        let mut spirv_words: Vec<u32> = Vec::new();
        for i in (0..spirv_bytes.len()).step_by(4) {
            let word_slice: u32 = u32::from_le_bytes(spirv_bytes[i..i + 4].try_into().unwrap());
            spirv_words.push(word_slice);
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

    #[allow(clippy::collapsible_if)]
    fn get_compute_group_size(entry_point: &EntryPoint) -> Option<(u32, u32, u32)> {
        for mode in entry_point.exec_modes.iter() {
            if mode.exec_mode == spirv::ExecutionMode::LocalSize
                || mode.exec_mode == spirv::ExecutionMode::LocalSizeHint
            {
                if let [
                    ConstantValue::U32(x),
                    ConstantValue::U32(y),
                    ConstantValue::U32(z),
                ] = mode.operands[..]
                    .iter()
                    .map(|op| op.value.clone())
                    .collect::<Vec<_>>()
                    .as_slice()
                {
                    return Some((*x, *y, *z));
                }
            }
        }
        None
    }

    #[allow(clippy::type_complexity)]
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
        let entry_points = ReflectConfig::new()
            .spv(data)
            .ref_all_rscs(true)
            .reflect()?;

        let mut layout_bindings: HashMap<u32, Vec<vk::DescriptorSetLayoutBinding<'a>>> =
            HashMap::new();
        let mut binding_map: HashMap<String, BindInfo> = HashMap::new();
        for entry_point in entry_points.iter() {
            for var in entry_point.vars.iter() {
                match var {
                    Variable::Descriptor {
                        name,
                        desc_bind,
                        desc_ty,
                        ty: _ty,
                        nbind: _nbind,
                    } => {
                        let mut layout = vk::DescriptorSetLayoutBinding {
                            binding: desc_bind.bind(),
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            descriptor_count: 1,
                            ..Default::default()
                        };
                        let mut bind_info = BindInfo::new(desc_bind.bind(), desc_ty.clone());
                        match desc_ty {
                            DescriptorType::StorageImage(s) => {
                                if desc_bind.set() != IMAGE_STORAGE_SHADER_BINDING as u32 {
                                    return Err("Storage images must be set 2.".into());
                                }
                                layout.descriptor_type = vk::DescriptorType::STORAGE_IMAGE;
                                bind_info.access = Some(*s);
                            }
                            DescriptorType::StorageBuffer(s) => {
                                if desc_bind.set() != SSBO_SHADER_BINDING as u32 {
                                    return Err("Storage buffer must be set 1.".into());
                                }
                                layout.descriptor_type = vk::DescriptorType::STORAGE_BUFFER;
                                bind_info.access = Some(*s);
                            }
                            DescriptorType::UniformBuffer() => {
                                if desc_bind.set() != UBO_SHADER_BINDING as u32 {
                                    return Err("Uniform buffer must be set 0.".into());
                                }
                                layout.descriptor_type = vk::DescriptorType::UNIFORM_BUFFER;
                            }
                            _ => {
                                panic!("Unsupported descriptor type.");
                            }
                        }
                        let v = layout_bindings.get_mut(&desc_bind.set());
                        match v {
                            Some(value) => {
                                value.push(layout);
                            }
                            None => {
                                layout_bindings.insert(desc_bind.set(), vec![layout]);
                            }
                        }
                        if name.is_none() {
                            panic!(
                                "Binding name is empty - Vulkan backend wont behave as expected."
                            )
                        }

                        binding_map.insert(name.clone().unwrap(), bind_info);
                    }
                    Variable::SpecConstant {
                        name: _name,
                        spec_id: _id,
                        ty: _ty,
                    } => {
                        // TODO: Use the info here to create the spec layout.
                    }
                    _ => {}
                }
            }
        }

        let mut work_size = WorkSize::default();
        for entry_point in entry_points.iter() {
            if let Some((x, y, z)) = Self::get_compute_group_size(entry_point) {
                work_size.x = x;
                work_size.y = y;
                work_size.z = z;
            };
        }
        Ok((layout_bindings, binding_map, work_size))
    }

    pub fn create_shader_module(
        spirv_bytecode: &[u32],
        device: &ash::Device,
    ) -> Result<vk::ShaderModule, Box<dyn Error>> {
        let ci = vk::ShaderModuleCreateInfo {
            code_size: std::mem::size_of_val(spirv_bytecode),
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
                if info.desc_type != DescriptorType::UniformBuffer() {
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
        view: &BufferView,
    ) -> Result<(), Box<dyn Error>> {
        let binding = self.binding_map.get(ssbo_name);
        match binding {
            Some(info) => {
                if info.desc_type != DescriptorType::StorageBuffer(info.access.unwrap()) {
                    return Err("Binding not registered as a SSBO.".into());
                }
                self.ssbos[info.binding as usize] = Some(*view);
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
        _size: vk::DeviceSize,
    ) -> Result<(), Box<dyn Error>> {
        let binding = self.binding_map.get(image_name);
        match binding {
            Some(info) => {
                if info.desc_type != DescriptorType::StorageImage(info.access.unwrap()) {
                    return Err("Binding not registered as a image storage type.".into());
                }
                self.storage_images[info.binding as usize] = Some(TextureView::new(texture));
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
