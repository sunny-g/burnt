use thiserror::Error;

///
#[derive(Debug, Error)]
pub enum Error {
    #[cfg(feature = "std")]
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("mmap error: {0}")]
    Mmap(#[from] mmap_rs::Error),

    #[error("SafeTensors error: {0}")]
    SafeTensors(#[from] safetensors::tensor::SafeTensorError),

    #[error("tokenizers error: {0}")]
    Tokenizers(#[from] tokenizers::tokenizer::Error),

    #[error("invalid format for tensor `{0}`: {1}")]
    InvalidFormat(String, String),
}

impl Error {
    pub fn invalid_format(name: impl ToString, msg: impl ToString) -> Self {
        Self::InvalidFormat(name.to_string(), msg.to_string())
    }

    pub fn mismatched_shape<const D: usize>(name: impl ToString, dims: &[usize]) -> Self {
        Self::invalid_format(
            name,
            format!(
                "mismatched dims; expected {} ndims, got shape {:?}",
                D, dims,
            ),
        )
    }
}

impl From<core::array::TryFromSliceError> for Error {
    fn from(error: core::array::TryFromSliceError) -> Self {
        Self::invalid_format("array", error)
    }
}

impl From<Error> for std::io::Error {
    fn from(error: Error) -> Self {
        match error {
            #[cfg(feature = "std")]
            Error::Io(e) => e,
            _ => Self::new(std::io::ErrorKind::Other, error),
        }
    }
}
