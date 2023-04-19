use crate::{Error, Record};
use burn_tensor::backend::Backend;
use mmap_rs::{Mmap, MmapFlags, MmapOptions};
use safetensors::{
    tensor::{TensorInfo, TensorView},
    SafeTensorError, SafeTensors,
};
use std::{collections::HashMap, fmt::Debug, fs::File, path::Path, sync::Arc};

///
#[derive(Clone)]
pub struct Manifest<B: Backend> {
    st: Arc<MmapedSafeTensors>,
    name: String,
    overrides: HashMap<String, String>,
    device: Option<B::Device>,
    // quantizer settings?
}

// builder
impl<B: Backend> Manifest<B> {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let st = MmapedSafeTensors::open(path)?;
        Ok(Self {
            st: Arc::new(st),
            name: "".into(),
            overrides: Default::default(),
            device: None,
        })
    }

    /// Scopes the manifest's current `name`d record.
    pub fn scoped(&self, next: impl AsRef<str>) -> Self {
        let next = next.as_ref();
        let name = (&self.name == "")
            .then(|| next.into())
            .unwrap_or_else(|| format!("{}.{}", &self.name, next));
        Self {
            st: self.st.clone(),
            name,
            overrides: self.overrides.clone(),
            device: self.device.clone(),
        }
    }

    /// Scopes the manifest's current `name`d record, and configures overrides
    /// for the associated module's fields, such that their encoded names are
    /// overridden with that that match what was serialized in the underlying
    /// `SafeTensors` file.
    pub fn with_scoped_overrides<'a>(
        &self,
        next: impl AsRef<str>,
        overrides: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Self {
        let mut out = self.scoped(next);
        out.overrides = overrides
            .into_iter()
            .map(|(a, b)| {
                (
                    format!("{}.{}", &out.name, a),
                    format!("{}.{}", &out.name, b),
                )
            })
            .collect();
        out
    }

    /// Sets the target device for the loaded record(s).
    pub fn with_device(mut self, device: B::Device) -> Self {
        self.device = Some(device);
        self
    }
}

// getters
impl<B: Backend> Manifest<B> {
    /// Returns the names of all contained records.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.st.tensors.keys().map(|s| s.as_str())
    }

    /// Returns the currently scoped name.
    pub fn current_name(&self) -> &str {
        let name = self.name.as_str();
        self.overrides.get(name).map(|s| s.as_str()).unwrap_or(name)
    }

    /// Returns the device that any scoped tensors will be loaded onto, if any.
    pub fn device(&self) -> Option<B::Device> {
        self.device.clone()
    }

    /// Loads the [`SafeTensorsRecord`] from the manifest.
    pub fn load<T: Record<B>>(self) -> Result<T, T::Error> {
        T::load(self)
    }

    /// Returns a view of the tensor identified by the currently scoped name.
    pub(crate) fn current_tensor(&self) -> Result<TensorView, Error> {
        Ok(self.st.as_ref().tensor(self.current_name())?)
    }
}

struct MmapedSafeTensors {
    mmap: Mmap,
    byte_offset: usize,
    pub(crate) tensors: HashMap<String, TensorInfo>,
}

impl MmapedSafeTensors {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let mmap = Self::open_mmap(path)?;
        let (offset, metadata) = SafeTensors::read_metadata(mmap.as_slice())?;
        let tensors = metadata
            .tensors()
            .into_iter()
            .map(|(name, info)| (name, info.clone()))
            .collect();
        Ok(Self {
            mmap,
            byte_offset: offset,
            tensors,
        })
    }

    /// Allow the user to get a specific tensor within the SafeTensors.
    /// The tensor returned is merely a view and the data is not owned by this
    /// structure.
    fn tensor(&self, name: &str) -> Result<TensorView, Error> {
        self.tensors.get(name).map_or_else(
            || Err(Error::TensorNotFound(name.into())),
            |info| {
                Ok(TensorView::new(
                    info.dtype,
                    info.shape.clone(),
                    &self.data()[info.data_offsets.0..info.data_offsets.1],
                )?)
            },
        )
    }

    fn data(&self) -> &[u8] {
        &self.mmap.as_slice()[self.byte_offset + 8..]
    }

    fn open_mmap(path: impl AsRef<Path>) -> Result<mmap_rs::Mmap, Error> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(unsafe {
            MmapOptions::new(len as usize)?
                .with_file(file, 0)
                .with_flags(MmapFlags::NO_CORE_DUMP)
                .map()?
        })
    }
}
