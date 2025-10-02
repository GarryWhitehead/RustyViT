use crate::device::DeviceStorage;
use crate::type_traits::BType;
use rusty_vk::public_types::{DeviceType, StorageBuffer, UniformBuffer};
use rusty_vk::vk_shader::ShaderProgram;
use rusty_vk::*;
use std::fmt::Debug;
use std::ptr::null;
use std::{cell::RefCell, collections::HashMap, error::Error, ops::Deref, sync::Arc};

#[derive(Clone)]
pub struct Vulkan<'a> {
    pub(crate) driver: Arc<RefCell<Driver>>,
    pub(crate) modules: HashMap<String, ShaderProgram<'a>>,
}

impl<'a> Vulkan<'a> {
    pub fn new(device_type: DeviceType) -> Result<Vulkan<'a>, Box<dyn Error>> {
        let driver = Arc::new(RefCell::new(Driver::new(device_type)?));
        Ok(Vulkan {
            driver,
            modules: HashMap::new(),
        })
    }

    pub fn try_get_module(
        &mut self,
        path: &str,
        name: &str,
    ) -> Result<ShaderProgram, Box<dyn Error>> {
        let p = self.modules.get(name);
        match p {
            Some(prog) => Ok(prog.clone()),
            None => {
                let spirv_bytes = std::fs::read(format!("{}/{}", path, name))?;
                let program =
                    ShaderProgram::try_new(&spirv_bytes, self.driver.borrow_mut().deref()).unwrap();
                let r = self.modules.insert(name.to_string(), program.clone());
                if r.is_some() {
                    panic!("Internal error: duplicated module insertion.");
                }
                Ok(program)
            }
        }
    }

    pub fn alloc_ubo_from_slice<E>(&self, elements: &[E]) -> UniformBuffer {
        let parts =
            unsafe { std::slice::from_raw_parts(elements.as_ptr() as *const u8, elements.len()) };
        let ubo = self
            .driver
            .borrow_mut()
            .allocate_ubo(std::mem::size_of::<E>());
        self.driver.borrow().map_ubo(parts, ubo);
        ubo
    }
}

impl<'a> Drop for Vulkan<'a> {
    fn drop(&mut self) {
        for (_key, prog) in self.modules.iter_mut() {
            prog.destroy(&self.driver.borrow());
        }
        // self.driver.borrow_mut().destroy();
    }
}

impl<'a, T: num::Zero + Clone + Send + Sync + Debug + 'static> DeviceStorage<T> for Vulkan<'a> {
    type Vec = StorageBuffer<T>;

    fn try_alloc(&self, sz: usize) -> Result<Self::Vec, Box<dyn Error>> {
        let ssbo = self.driver.borrow_mut().allocate_ssbo(sz);
        self.driver.borrow_mut().memset_zero(&ssbo);
        Ok(ssbo)
    }

    fn try_alloc_with_slice(&self, slice: &[T]) -> Result<Self::Vec, Box<dyn Error>> {
        let ssbo = self.try_alloc(slice.len())?;
        self.driver.borrow().try_map_ssbo(slice, &ssbo)?;
        Ok(ssbo)
    }

    fn try_from_device_vec(&self, src: &Self::Vec) -> Result<Vec<T>, Box<dyn Error>> {
        if let Some(rd) = &mut self.driver.borrow_mut().renderdoc {
            println!("STARTING RENDERDOC CAPTURE....");
            rd.start_frame_capture(null(), null());
        }
        let v = self.driver.borrow_mut().map_ssbo_to_host(src);
        if let Some(rd) = &mut self.driver.borrow_mut().renderdoc {
            println!("ENDINGRENDERDOC CAPTURE....");
            rd.end_frame_capture(null(), null());
        }
        Ok((v))
    }

    fn len(v: &Self::Vec) -> usize {
        v.element_count()
    }

    fn try_sync_stream0(&self) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}
