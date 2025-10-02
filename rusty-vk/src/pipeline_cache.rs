use crate::descriptor_cache::MAX_DESC_SET_COUNT;
use ash::vk;
use ash::vk::Handle;
use std::{collections::HashMap, hash::Hash};

#[derive(Hash, Eq, PartialEq, Copy, Clone, Default)]
struct PipelineLayoutKey {
    desc_layouts: [vk::DescriptorSetLayout; MAX_DESC_SET_COUNT],
}

#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct PLayout {
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) frame_last_used: u64,
}

impl PLayout {
    fn new(layout: vk::PipelineLayout, current_frame: u64) -> Self {
        Self {
            layout: layout,
            frame_last_used: current_frame,
        }
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone, Default)]
struct PipelineKey {
    pub(crate) module: vk::ShaderModule,
    pub(crate) layout: vk::PipelineLayout,
}

#[derive(Copy, Clone, Debug)]
struct PLine {
    pipeline: vk::Pipeline,
    cache: vk::PipelineCache,
}

impl PLine {
    pub fn new(pipeline: vk::Pipeline, cache: vk::PipelineCache) -> Self {
        Self { pipeline, cache }
    }
}

pub struct PipelineCache {
    pub(crate) pipelines: HashMap<PipelineKey, PLine>,
    pipeline_requires: PipelineKey,
    pub(crate) playouts: HashMap<PipelineLayoutKey, PLayout>,
    layout_requires: PipelineLayoutKey,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            pipeline_requires: Default::default(),
            playouts: HashMap::new(),
            layout_requires: Default::default(),
        }
    }

    pub fn get_layout(&mut self, current_frame: u64, device: &ash::Device) -> PLayout {
        let pl = self.playouts.get_mut(&self.layout_requires);
        match pl {
            Some(layout) => {
                layout.frame_last_used = current_frame;
                *layout
            }
            None => {
                let ci = vk::PipelineLayoutCreateInfo {
                    set_layout_count: MAX_DESC_SET_COUNT as u32,
                    p_set_layouts: self.layout_requires.desc_layouts.as_ptr(),
                    ..Default::default()
                };
                let l = unsafe { device.create_pipeline_layout(&ci, None).unwrap() };
                let out = PLayout::new(l, current_frame);
                self.playouts.insert(self.layout_requires, out.clone());
                out
            }
        }
    }

    pub fn bind_pipeline(&mut self, cmds: vk::CommandBuffer, device: &ash::Device) {
        let p = self.pipelines.get_mut(&self.pipeline_requires);
        let pline = match p {
            Some(pline) => *pline,
            None => {
                let (cache, pipeline) = Self::create_compute_pline(
                    &self.pipeline_requires.layout,
                    &self.pipeline_requires.module,
                    device,
                );
                let pline = PLine::new(pipeline, cache);
                self.pipelines.insert(self.pipeline_requires, pline.clone());
                pline
            }
        };
        unsafe { device.cmd_bind_pipeline(cmds, vk::PipelineBindPoint::COMPUTE, pline.pipeline) };
    }

    fn create_compute_pline(
        layout: &vk::PipelineLayout,
        module: &vk::ShaderModule,
        device: &ash::Device,
    ) -> (vk::PipelineCache, vk::Pipeline) {
        assert!(!layout.is_null());

        let stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: *module,
            p_name: c"main".as_ptr(),
            ..Default::default()
        };
        let cache = vk::PipelineCacheCreateInfo {
            ..Default::default()
        };
        let ci = vk::ComputePipelineCreateInfo {
            layout: *layout,
            stage,
            ..Default::default()
        };
        let pline_cache = unsafe { device.create_pipeline_cache(&cache, None).unwrap() };
        let pline = unsafe {
            device
                .create_compute_pipelines(pline_cache, &[ci], None)
                .unwrap()[0]
        };
        (pline_cache, pline)
    }

    pub fn bind_shader_module(&mut self, module: vk::ShaderModule) {
        self.pipeline_requires.module = module;
    }

    pub fn bind_layout(&mut self, layout: vk::PipelineLayout) {
        self.pipeline_requires.layout = layout;
    }

    pub fn bind_desc_layouts(&mut self, layouts: &[vk::DescriptorSetLayout; MAX_DESC_SET_COUNT]) {
        self.layout_requires.desc_layouts = *layouts;
    }

    pub fn destroy(&self, device: &ash::Device) {
        for (_k, v) in self.pipelines.iter() {
            unsafe {
                device.destroy_pipeline(v.pipeline, None);
                device.destroy_pipeline_cache(v.cache, None)
            };
        }
        for (_k, v) in self.playouts.iter() {
            unsafe {
                device.destroy_pipeline_layout(v.layout, None);
            }
        }
    }
}
