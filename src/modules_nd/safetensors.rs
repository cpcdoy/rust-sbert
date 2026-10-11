//! Minimal read-only safetensors parser (f32 tensors only).
//!
//! Used by [`crate::modules::Dense`] to load `2_Dense/weights.safetensors`
//! without a torch dependency: the Dense projection of a checkpoint lives
//! outside the ONNX graph. The format is trivial — 8 bytes little-endian
//! u64 header length, a JSON header (`{name: {dtype, shape,
//! data_offsets}}`), then the raw little-endian buffers — so a ~80-line
//! reader beats adding a dependency.
//!
//! See <https://github.com/huggingface/safetensors>.

use std::collections::HashMap;
use std::convert::TryInto;
use std::path::Path;

use ndarray::{ArrayD, IxDyn};
use serde::Deserialize;

use crate::Error;

#[derive(Debug, Deserialize)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    /// Byte range within the data section (start inclusive, end exclusive).
    data_offsets: Vec<u64>,
}

/// A parsed safetensors file. Data is kept as raw bytes; tensors are
/// materialized as owned f32 arrays on demand.
pub struct SafetensorsFile {
    header: HashMap<String, TensorMeta>,
    data: Vec<u8>,
    data_start: usize,
}

impl SafetensorsFile {
    pub fn read(path: &Path) -> Result<Self, Error> {
        let bytes = std::fs::read(path).map_err(|e| {
            log::error!("{} not readable: {}", path.display(), e);
            Error::Encoding("safetensors file not readable")
        })?;
        if bytes.len() < 8 {
            return Err(Error::Encoding(
                "safetensors file truncated (no header length)",
            ));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        if bytes.len() < 8 + header_len {
            return Err(Error::Encoding("safetensors file truncated (header)"));
        }
        // The header maps tensor names to descriptors, plus an optional
        // "__metadata__" entry ({format: "pt", ...}) that must be skipped.
        let raw: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..8 + header_len]).map_err(|e| {
                log::error!("invalid safetensors header in {}: {}", path.display(), e);
                Error::Encoding("invalid safetensors header")
            })?;
        let header: HashMap<String, TensorMeta> = raw
            .into_iter()
            .filter(|(name, _)| name != "__metadata__")
            .map(|(name, value)| {
                let meta: TensorMeta = serde_json::from_value(value).map_err(|e| {
                    log::error!("invalid safetensors tensor header for {:?}: {}", name, e);
                    Error::Encoding("invalid safetensors tensor header")
                })?;
                Ok((name, meta))
            })
            .collect::<Result<_, Error>>()?;

        Ok(SafetensorsFile {
            header,
            data: bytes,
            data_start: 8 + header_len,
        })
    }

    /// Materialize tensor `name` as an owned f32 array. Only `F32` tensors
    /// are supported — Dense weights of sentence-transformers checkpoints
    /// are f32, and anything else should fail loudly rather than silently
    /// round-trip through a lossy conversion.
    pub fn tensor_f32(&self, name: &str) -> Result<ArrayD<f32>, Error> {
        let meta = self.header.get(name).ok_or_else(|| {
            log::error!("tensor {:?} not found in safetensors file", name);
            Error::Encoding("tensor not found in safetensors file")
        })?;
        if meta.dtype != "F32" {
            return Err(Error::Encoding(
                "unsupported safetensors dtype (only F32 is supported)",
            ));
        }
        if meta.data_offsets.len() != 2 {
            return Err(Error::Encoding(
                "invalid data_offsets in safetensors header",
            ));
        }
        let (start, end) = (meta.data_offsets[0] as usize, meta.data_offsets[1] as usize);
        let expected_len: usize = meta.shape.iter().product();
        let range_end = self.data_start + end;
        if self.data.len() < range_end || end - start != expected_len * std::mem::size_of::<f32>() {
            return Err(Error::Encoding(
                "safetensors data section shorter than header claims",
            ));
        }
        let slice = &self.data[self.data_start + start..range_end];
        let values = slice
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect::<Vec<f32>>();
        ArrayD::from_shape_vec(IxDyn(&meta.shape), values)
            .map_err(|_| Error::Encoding("safetensors shape/len mismatch"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an in-memory safetensors file with two f32 tensors. The path
    /// is unique per caller so parallel tests never share (and unlink)
    /// the same file.
    fn build_file(name: &str) -> std::path::PathBuf {
        let t0 = [1.0f32, -2.0, 3.5];
        let t1 = [0.25f32; 4];
        let header = r#"{"t0":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},"t1":{"dtype":"F32","shape":[2,2],"data_offsets":[12,28]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        for v in t0.iter().chain(t1.iter()) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let path = std::env::temp_dir().join(format!("sbert-safetensors-test-{name}.st"));
        std::fs::write(&path, &bytes).unwrap();
        path
    }

    #[test]
    fn reads_f32_tensors() {
        let path = build_file("read");
        let f = SafetensorsFile::read(&path).unwrap();
        let t0 = f.tensor_f32("t0").unwrap();
        assert_eq!(t0.shape(), &[3]);
        assert_eq!(
            t0.iter().copied().collect::<Vec<f32>>(),
            vec![1.0, -2.0, 3.5]
        );
        let t1 = f.tensor_f32("t1").unwrap();
        assert_eq!(t1.shape(), &[2, 2]);
        assert!(t1.iter().all(|v| *v == 0.25));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_tensor_errors() {
        let path = build_file("missing");
        let f = SafetensorsFile::read(&path).unwrap();
        assert!(f.tensor_f32("nope").is_err());
        std::fs::remove_file(&path).ok();
    }
}
