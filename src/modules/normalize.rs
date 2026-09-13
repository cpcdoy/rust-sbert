//! `sentence_transformers.models.Normalize` — L2-normalize the sentence
//! embedding. Has no config or weights; appears in `modules.json` as
//! `{"type": "sentence_transformers.models.Normalize"}` with no `path`.

use tch::Tensor;

use crate::modules::{Features, Module};
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
                return Err(Error::Encoding(
                    "Normalize received non-sentence features",
                ));
            }
        };
        // L2 norm along the hidden dim (axis 1 of [batch, hidden]), keepdim
        // so the broadcast divides cleanly. Clamp the denominator to avoid
        // div-by-zero on a zero vector.
        //
        // Reborrow the &mut Tensor as &Tensor for the arithmetic (tch's Mul/Div
        // impls are on `&Tensor`/`Tensor`, not `&mut Tensor`).
        let emb: &Tensor = &*embedding;
        let norm = (emb * emb)
            .sum_dim_intlist(1, true, tch::Kind::Float)
            .sqrt()
            .clamp_min(1e-12);
        *embedding = emb / &norm;
        Ok(())
    }
}
