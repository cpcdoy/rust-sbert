//! `sentence_transformers.models.Normalize` — L2-normalize the sentence
//! embedding. Has no config or weights; appears in `modules.json` as
//! `{"type": "sentence_transformers.models.Normalize"}` with no `path`.

use ndarray::Axis;

use crate::modules_nd::{Features, Module};
use crate::Error;

pub struct Normalize;

impl Normalize {
    pub fn new() -> Self {
        Normalize
    }
}

impl Default for Normalize {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Normalize {
    fn forward(&self, features: &mut Features) -> Result<(), Error> {
        let embedding = match features {
            Features::Sentence { embedding } => embedding,
            _ => {
                return Err(Error::Encoding("Normalize received non-sentence features"));
            }
        };
        // L2 norm along the hidden dim (axis 1 of [batch, hidden]), kept as
        // [batch, 1] so the broadcast divides cleanly. Clamp the denominator
        // to avoid div-by-zero on a zero vector.
        let norm = embedding
            .mapv(|v| v * v)
            .sum_axis(Axis(1))
            .mapv(|v| v.sqrt().max(1e-12))
            .insert_axis(Axis(1));
        let normalized = &*embedding / &norm;
        *embedding = normalized;
        Ok(())
    }
}
