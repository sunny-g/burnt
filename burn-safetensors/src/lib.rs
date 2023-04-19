mod error;
mod manifest;

pub use error::Error;
pub use manifest::Manifest;
pub use safetensors::{Dtype, View};

use burn_core::{
    module::{Module, Param, ParamId},
    nn::{
        conv::{Conv1d, Conv2d},
        Dropout, EmbeddingRecord, LayerNormRecord, LinearRecord,
    },
};
use burn_tensor::{
    backend::Backend, BasicOps, Data, Element, ElementConversion, Shape, Tensor, TensorKind,
};
use half::{bf16, f16};

///
pub trait Record<B: Backend>: Sized {
    type Error: From<Error>;

    fn load(manifest: Manifest<B>) -> Result<Self, Self::Error>;

    // #[cfg(feature = "std")]
    // fn load_from_file(path: impl AsRef<std::path::Path>) -> Result<Self, Self::Error> {
    //     let mmap = Self::open_mmap(path)?;
    //     let st = SafeTensors::deserialize(mmap.as_slice())?;
    //     Self::load(&st)
    // }
}

/*
 * impls
 */

impl<B, const D: usize, K> Record<B> for Tensor<B, D, K>
where
    B: Backend,
    K: TensorKind<B> + BasicOps<B>,
    K::Elem: Element,
{
    type Error = Error;

    fn load(manifest: Manifest<B>) -> Result<Self, Self::Error> {
        let name = manifest.current_name();
        let view = manifest.current_tensor()?;

        // FIXME: should we be squeezing? this assumes single dims are at the start
        let shape = {
            let i = view.shape().partition_point(|d| *d == 1);
            let dims = &view.shape()[i..];
            dims.try_into()
                .map(Shape::new)
                .map_err(|_| Error::mismatched_shape::<D>(name, dims))?
        };

        let dtype = view.dtype();
        let dsize = dtype.size();
        debug_assert_eq!(
            shape.num_elements() * dsize,
            (&view).data_len(),
            "unexpected tensor length (bytes)",
        );

        let iter = view.data().chunks(dsize);
        // FIXME: assumes little-endian and probably other things about layout
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
        if let Some(device) = manifest.device() {
            Ok(Tensor::from_data_device(data, &device))
        } else {
            Ok(Tensor::from_data(data))
        }
    }
}

impl<B, const D: usize, K> Record<B> for Param<Tensor<B, D, K>>
where
    B: Backend,
    K: TensorKind<B> + BasicOps<B>,
    K::Elem: Element,
{
    type Error = Error;

    fn load(manifest: Manifest<B>) -> Result<Self, Self::Error> {
        let id = ParamId::from(manifest.current_name());
        Ok(Param::new(id, manifest.load()?))
    }
}

impl<B: Backend> Record<B> for EmbeddingRecord<B> {
    type Error = Error;

    fn load(manifest: Manifest<B>) -> Result<Self, Self::Error> {
        let weight = manifest.scoped("weight").load::<Param<Tensor<B, 2>>>()?;
        // let [n_embedding, d_model] = weight.shape().dims;

        Ok(EmbeddingRecord { weight })
    }
}

// FIXME: better handling of missing bias
impl<B: Backend> Record<B> for LinearRecord<B> {
    type Error = Error;

    fn load(manifest: Manifest<B>) -> Result<Self, Self::Error> {
        let weight = manifest.scoped("weight").load::<Param<Tensor<B, 2>>>()?;
        let bias = manifest.scoped("bias").load().ok();
        // let [d_input, d_output] = weight.shape().dims;

        Ok(Self { weight, bias })
    }
}

impl<B: Backend> Record<B> for LayerNormRecord<B> {
    type Error = Error;

    fn load(manifest: Manifest<B>) -> Result<Self, Self::Error> {
        let gamma = manifest.scoped("gamma").load::<Param<Tensor<B, 1>>>()?;
        let beta = manifest.scoped("beta").load()?;

        Ok(Self {
            gamma,
            beta,
            epsilon: (),
        })
    }
}

/*
 * safetensors/burn tensor compatibility
 */

// macro_rules! impl_load_record {
//     ($(($($sig:tt)*),)*) => { $(
//         $($sig)* {
//             type Record = <Self as Module<B>>::Record;
//         }
//     )* };
// }

// trait HasRecord<B: Backend> {
//     type Record: Record;
// }
// impl<B: Backend, const D: usize, K: TensorKind<B>> HasRecord<B> for Tensor<B, D, K>
// where
//     Tensor<B, D, K>: Record,
// {
//     type Record = Self;
// }
// macro_rules! impl_has_record {
//     ($($ty:ty,)*) => { $(
//         impl<B: Backend> HasRecord<B> for $ty {
//             type Record = <Self as Module<B>>::Record;
//         }
//     )* };
// }
// impl_has_record!(
//     // BatchNorm<B>,
//     Conv1d<B>,
//     Conv2d<B>,
//     Dropout,
//     Embedding<B>,
//     // GELU,
//     Linear<B>,
//     LayerNorm<B>,
//     // ReLU,
// );

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
