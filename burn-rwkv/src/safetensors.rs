use crate::Error;
use burn_core::{
    module::{Module, Param, ParamId},
    nn::{
        conv::{Conv1d, Conv2d},
        Dropout, Embedding, EmbeddingConfig, EmbeddingRecord, LayerNorm, LayerNormConfig,
        LayerNormRecord, Linear, LinearConfig, LinearRecord,
    },
    record::Record,
};
use burn_tensor::{
    backend::Backend, BasicOps, Data, Element, ElementConversion, Shape, Tensor, TensorKind,
};
use half::{bf16, f16};
use safetensors::{tensor::TensorView, Dtype, SafeTensors, View};
use std::{collections::HashMap, fmt::Debug};

trait HasRecord<B: Backend> {
    type Record: Record;
}
impl<B: Backend, const D: usize, K: TensorKind<B>> HasRecord<B> for Tensor<B, D, K>
where
    Tensor<B, D, K>: Record,
{
    type Record = Self;
}
macro_rules! impl_has_record {
    ($($ty:ty,)*) => { $(
        impl<B: Backend> HasRecord<B> for $ty {
            type Record = <Self as Module<B>>::Record;
        }
    )* };
}
impl_has_record!(
    // BatchNorm<B>,
    Conv1d<B>,
    Conv2d<B>,
    Dropout,
    Embedding<B>,
    // GELU,
    Linear<B>,
    LayerNorm<B>,
    // ReLU,
);

/*
 * load records from safetensors
 */

pub trait LoadFromSafeTensors<B: Backend>: Sized {
    fn load<'data>(config: SafeTensorsConfig<B>, st: &SafeTensors<'data>) -> Result<Self, Error>;

    // #[cfg(feature = "std")]
    // fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
    //     let mmap = Self::open_mmap(path)?;
    //     let st = SafeTensors::deserialize(mmap.as_slice())?;
    //     Self::load(&st)
    // }
}

macro_rules! impl_load_record {
    ($(($($sig:tt)*),)*) => { $(
        $($sig)* {
            type Record = <Self as Module<B>>::Record;
        }
    )* };
}

#[derive(Debug, Clone, Default)]
pub struct SafeTensorsConfig<B: Backend> {
    pub name: Option<String>,
    pub overrides: Option<HashMap<String, String>>,
    pub device: Option<B::Device>,
    // quantizer settings?
}

impl<B: Backend> SafeTensorsConfig<B> {
    ///
    pub fn name(&self) -> &str {
        let orig = self.name.as_ref().unwrap();
        self.overrides
            .as_ref()
            .and_then(|o| o.get(orig))
            .unwrap_or(orig)
    }

    /// Scopes the config's current `name`
    pub fn scoped(&self, next: impl AsRef<str>) -> Self {
        let scope = self.name.as_ref().map_or_else(
            || next.as_ref().into(),
            |scope| format!("{}.{}", scope, next.as_ref()),
        );
        Self {
            name: Some(scope),
            overrides: self.overrides.clone(),
            device: self.device.clone(),
        }
    }

    /// Scopes the config's current `name`, and configures overrides for the
    /// associated module's fields, such that their names match what is
    /// serialized in the underlying `SafeTensors` file.
    pub fn with_scoped_overrides<'a>(
        &self,
        next: impl AsRef<str>,
        overrides: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Self {
        let mut out = self.scoped(next);
        let name = out.name();
        out.overrides = Some(
            overrides
                .into_iter()
                .map(|(a, b)| (format!("{}.{}", name, a), format!("{}.{}", name, b)))
                .collect(),
        );
        out
    }

    /// Sets the target device for the loaded tensors.
    pub fn with_device(mut self, device: B::Device) -> Self {
        self.device = Some(device);
        self
    }

    pub fn load<T: LoadFromSafeTensors<B>>(self, st: &SafeTensors) -> Result<T, Error> {
        T::load(self, &st)
    }

    #[cfg(feature = "std")]
    pub fn load_from_file<T: LoadFromSafeTensors<B>>(
        self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<T, Error> {
        let mmap = Self::open_mmap(path)?;
        let st = SafeTensors::deserialize(mmap.as_slice())?;
        T::load(self, &st)
    }

    fn open_mmap(path: impl AsRef<std::path::Path>) -> Result<mmap_rs::Mmap, Error> {
        use mmap_rs::{MmapFlags as Flags, MmapOptions as Opts};

        let f = std::fs::File::open(path)?;
        let len = f.metadata()?.len();
        Ok(unsafe {
            Opts::new(len as usize)?
                .with_file(f, 0)
                .with_flags(Flags::NO_CORE_DUMP)
                .map()?
        })
    }
}

/*
 * impls
 */

impl<B, const D: usize, K> LoadFromSafeTensors<B> for Tensor<B, D, K>
where
    B: Backend,
    K: TensorKind<B> + BasicOps<B>,
    K::Elem: Element,
{
    fn load<'data>(config: SafeTensorsConfig<B>, st: &SafeTensors<'data>) -> Result<Self, Error> {
        let name = config.name();
        let view = st.tensor(name)?;

        // FIXME: should we be squeezing? this assumes single dims are at the start
        let shape = {
            let i = view.shape().partition_point(|d| *d == 1);
            let dims = &view.shape()[i..];
            dims.try_into()
                .map(Shape::new)
                .map_err(|_| Error::mismatched_shape::<D>(name, dims))?
        };

        let dtype = view.dtype();
        debug_assert_eq!(
            (&view).data_len(),
            shape.num_elements() * dtype.size(),
            "unexpected tensor length",
        );

        let iter = view.data().chunks(dtype.size());
        let val = match dtype {
            Dtype::F16 => iter
                .map(|b| f16::from_le_bytes([b[0], b[1]]))
                .map(|b| b.elem())
                .collect(),
            Dtype::BF16 => iter
                .map(|b| bf16::from_le_bytes([b[0], b[1]]))
                .map(|b| b.elem())
                .collect(),
            _ => unimplemented!(),
        };

        let data = Data::new(val, shape);
        if let Some(device) = config.device {
            Ok(Tensor::from_data_device(data, &device))
        } else {
            Ok(Tensor::from_data(data))
        }
    }
}

impl<B, const D: usize, K> LoadFromSafeTensors<B> for Param<Tensor<B, D, K>>
where
    B: Backend,
    K: TensorKind<B> + BasicOps<B>,
    K::Elem: Element,
{
    fn load<'data>(config: SafeTensorsConfig<B>, st: &SafeTensors<'data>) -> Result<Self, Error> {
        let id = ParamId::from(config.name());
        Ok(Param::new(id, config.load(st)?))
    }
}

impl<B: Backend> LoadFromSafeTensors<B> for EmbeddingRecord<B> {
    fn load<'data>(config: SafeTensorsConfig<B>, st: &SafeTensors<'data>) -> Result<Self, Error> {
        let weight = config.scoped("weight").load::<Param<Tensor<B, 2>>>(st)?;
        // let [n_embedding, d_model] = weight.shape().dims;

        Ok(EmbeddingRecord { weight })
    }
}

// FIXME: better handling of missing bias
impl<B: Backend> LoadFromSafeTensors<B> for LinearRecord<B> {
    fn load<'data>(config: SafeTensorsConfig<B>, st: &SafeTensors<'data>) -> Result<Self, Error> {
        let weight = config.scoped("weight").load::<Param<Tensor<B, 2>>>(st)?;
        let bias = config.scoped("bias").load(st).ok();
        // let [d_input, d_output] = weight.shape().dims;

        Ok(LinearRecord { weight, bias })
    }
}

impl<B: Backend> LoadFromSafeTensors<B> for LayerNormRecord<B> {
    fn load<'data>(config: SafeTensorsConfig<B>, st: &SafeTensors<'data>) -> Result<Self, Error> {
        let gamma = config.scoped("gamma").load::<Param<Tensor<B, 1>>>(st)?;
        let beta = config.scoped("beta").load(st)?;

        Ok(LayerNormRecord {
            gamma,
            beta,
            epsilon: (),
        })
    }
}

/*
 * safetensors/burn tensor compatibility
 */

// struct ModuleLoader

// trait DataAdapter {
//     fn try_into_data<E: Element, const D: usize>(self) -> Result<Data<E, D>, Error>;
//     // fn into_tensor<B, const D: usize, K>(self, device: Option<B::Device>) -> Tensor<B, D, K>
//     // where
//     //     B: Backend,
//     //     K: TensorKind<B>,
//     // {
//     //     if let Some(device) = device {
//     //         Tensor::from_data_device(self.into_data(), &device)
//     //     } else {
//     //         Tensor::from_data(self.into_data())
//     //     }
//     // }
// }

// impl FileRecorder for SafeTensorRecorder {
//     // fn record(&mut self, tensor: &Tensor) {
//     //     let path = format!("tensor-{}.npy", tensor.name());
//     //     let mut file = File::create(path).unwrap();
//     //     tensor.save(&mut file).unwrap();
//     // }

//     fn file_extension() -> &'static str {
//         "safetensor"
//     }
// }

// impl Recorder for SafeTensorRecorder {
//     type RecordArgs = PathBuf;
//     type RecordOutput = ();
//     type LoadArgs = PathBuf;

//     fn record<Obj: Serialize + DeserializeOwned>(
//         obj: Obj,
//         mut dst: PathBuf,
//     ) -> Result<(), Error> {
//         todo!()
//     }

//     fn load<Obj: Serialize + DeserializeOwned>(src: PathBuf) -> Result<Obj, Error> {
//         todo!()
//     }
// }

// #[derive(Debug, Default)]
// pub struct SafeTensorSerializer<W>(W);
// impl<W: Write> Serializer for SafeTensorSerializer {
//     fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
//         todo!()
//     }
// }

// pub struct SafeTensorRecordSettings<'a>(PhantomData<&'a ()>);

// impl<'a> RecordSettings for SafeTensorRecordSettings<'a>
// {
//     type FloatElem = f16;
//     type IntElem = i16;
//     type Recorder = SafeTensorRecorder<R, W>;
// }

// pub trait MmapRecorder: Recorder<LoadArgs = Mmap, RecordOutput = (), RecordArgs = File>

// pub struct SafeTensorRecorder<R, W>(PhantomData<(R, W)>);

// impl<R, W> Debug for SafeTensorRecorder<R, W> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("SafeTensorRecorder").finish()
//     }
// }

// impl<R, W> Default for SafeTensorRecorder<R, W> {
//     fn default() -> Self {
//         Self(PhantomData)
//     }
// }

// impl<R: Read, W: Write> Recorder for SafeTensorRecorder<R, W>
// where
//     Self: Send + Sync,
// {
//     type RecordArgs = W;
//     type RecordOutput = ();
//     type LoadArgs = R;

//     fn record<Obj: Serialize + DeserializeOwned>(
//         obj: Obj,
//         mut dst: W,
//     ) -> Result<(), Error> {
//         todo!()
//     }

//     fn load<Obj: Serialize + DeserializeOwned>(mut src: R) -> Result<Obj, Error> {
//         todo!()
//     }
// }

// pub struct SafeTensorRecordSettings<R, W>(PhantomData<(R, W)>);

// impl<R, W> Debug for SafeTensorRecordSettings<R, W> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("SafeTensorRecordSettings").finish()
//     }
// }

// impl<R, W> Default for SafeTensorRecordSettings<R, W> {
//     fn default() -> Self {
//         Self(PhantomData)
//     }
// }

// impl<R: Read, W: Write> RecordSettings for SafeTensorRecordSettings<R, W>
// where
//     SafeTensorRecorder<R, W>: Send + Sync,
//     Self: Send + Sync,
// {
//     type FloatElem = f16;
//     type IntElem = i16;
//     type Recorder = SafeTensorRecorder<R, W>;
// }
