use crate::vk_instance::ContextInstance;
use ash::vk::Bool32;
use ash::{Instance, vk};
use log::info;
use std::error::Error;

#[derive(Clone)]
pub struct ContextDevice {
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub compute_queue_idx: u32,
    pub compute_queue: vk::Queue,
}

impl ContextDevice {
    pub fn new(
        c_instance: &ContextInstance,
        device_type: vk::PhysicalDeviceType,
    ) -> Result<Self, Box<dyn Error>> {
        let (physical_device, queue_family_idx) =
            find_physical_device(&c_instance.instance, device_type)?;

        let compute_queue_idx = queue_family_idx;

        let queue_priority = [1.0];
        // A compute queue is mandatory - obviously!!
        let queue_infos: Vec<vk::DeviceQueueCreateInfo> = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(compute_queue_idx)
                .queue_priorities(&queue_priority),
        ];

        let phys_features = unsafe {
            c_instance
                .instance
                .get_physical_device_features(physical_device)
        };
        let mut coop_mat_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR {
            cooperative_matrix: Bool32::from(true),
            ..Default::default()
        };
        let mut sg_features = vk::PhysicalDeviceSubgroupSizeControlFeatures {
            subgroup_size_control: Bool32::from(true),
            compute_full_subgroups: Bool32::from(true),
            ..Default::default()
        };
        let mut robust_info = vk::PhysicalDeviceImageRobustnessFeatures {
            robust_image_access: vk::TRUE,
            ..Default::default()
        };
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_sampled_image_array_non_uniform_indexing(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_variable_descriptor_count(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .shader_int8(true)
            .uniform_and_storage_buffer8_bit_access(true)
            .descriptor_indexing(true)
            .vulkan_memory_model(true)
            .shader_float16(true);

        let mut features11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(true)
            .uniform_and_storage_buffer16_bit_access(true);

        let phys_dev_features = vk::PhysicalDeviceFeatures {
            texture_compression_etc2: phys_features.texture_compression_etc2,
            texture_compression_bc: phys_features.texture_compression_bc,
            sampler_anisotropy: phys_features.sampler_anisotropy,
            shader_storage_image_extended_formats: phys_features
                .shader_storage_image_extended_formats,
            ..Default::default()
        };
        let mut required_features = vk::PhysicalDeviceFeatures2::default()
            .features(phys_dev_features)
            .push_next(&mut features11)
            .push_next(&mut features12)
            .push_next(&mut robust_info)
            .push_next(&mut coop_mat_features)
            .push_next(&mut sg_features);

        let device_extension_names_raw = [ash::ext::descriptor_indexing::NAME.as_ptr()];

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extension_names_raw)
            .push_next(&mut required_features);

        let device = unsafe {
            c_instance
                .instance
                .create_device(physical_device, &device_create_info, None)?
        };

        let compute_queue = unsafe { device.get_device_queue(compute_queue_idx, 0) };

        /*let coop_instance =
            ash::khr::cooperative_matrix::Instance::new(&c_instance.entry, &c_instance.instance);
        let coop_props = unsafe {
            coop_instance.get_physical_device_cooperative_matrix_properties(physical_device)
        };*/

        Ok(Self {
            device,
            physical_device,
            compute_queue_idx,
            compute_queue,
        })
    }

    pub fn destroy(&mut self) {
        unsafe { self.device.destroy_device(None) };
    }
}

fn find_physical_device(
    instance: &Instance,
    device_type: vk::PhysicalDeviceType,
) -> Result<(vk::PhysicalDevice, u32), Box<dyn Error>> {
    let phys_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Unable to find any physical devices.")
    };

    // Find an appropriate physical device.
    let (phys_device, queue_family_idx) = phys_devices
        .iter()
        .find_map(|phys_device| unsafe {
            instance
                .get_physical_device_queue_family_properties(*phys_device)
                .iter()
                .enumerate()
                .find_map(|(idx, info)| {
                    // Looking for a device with a compute queue - we don't care about graphics (well for now anyway!).
                    let gpu_props = instance.get_physical_device_properties(*phys_device);
                    if info.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && gpu_props.device_type == device_type
                    {
                        info!("{gpu_props:?}");
                        Some((*phys_device, idx))
                    } else {
                        None
                    }
                })
        })
        .expect("Unable to find a valid device.");

    Ok((phys_device, queue_family_idx as u32))
}
