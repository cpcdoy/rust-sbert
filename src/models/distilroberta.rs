use std::mem;
use std::path::PathBuf;
use std::sync::Arc;

use rayon::prelude::*;
use rust_bert::bert::BertConfig;
use rust_bert::roberta::RobertaForSequenceClassification;
use rust_bert::Config;
use tch::{nn, Device, Tensor};

use crate::tokenizers::Tokenizer;
use crate::{Embeddings, Error};

pub struct DistilRobertaForSequenceClassification<T> {
    lm_model: RobertaForSequenceClassification,
    tokenizer: Arc<T>,
    device: Device,
}

impl<T> DistilRobertaForSequenceClassification<T>
where
    T: Tokenizer + Send + Sync,
{
    pub fn new<P>(root: P) -> Result<Self, Error>
    where
        P: Into<PathBuf>,
    {
        let root = root.into();

        let config_file = root.join("config.json");
        let weights_file = root.join("model.ot");

        // Set-up DistilRoBERTa model and tokenizer

        let config = BertConfig::from_file(&config_file);

        let device = Device::cuda_if_available();
        log::info!("Using device {:?}", device);

        let mut vs = nn::VarStore::new(device);

        let tokenizer = Arc::new(T::new(&root)?);
        let lm_model = RobertaForSequenceClassification::new(&vs.root(), &config);

        vs.load(weights_file)?;

        Ok(DistilRobertaForSequenceClassification {
            lm_model,
            tokenizer,
            device,
        })
    }

    pub fn forward<S, B>(&self, input: &[S], batch_size: B) -> Result<Vec<Embeddings>, Error>
    where
        S: AsRef<str>,
        B: Into<Option<usize>>,
    {
        let input = input.iter().map(AsRef::as_ref).collect::<Vec<&str>>();
        let batch_size = batch_size.into().unwrap_or_else(|| 64);

        let _guard = tch::no_grad_guard();

        let sorted_pad_input_idx = pad_sort(&input.iter().map(|s| s.len()).collect::<Vec<usize>>());
        let sorted_pad_input = sorted_pad_input_idx
            .iter()
            .map(|i| input[*i])
            .collect::<Vec<&str>>();

        let input_len = sorted_pad_input.len();
        let tokenizer = self.tokenizer.clone();
        let device = self.device;

        // Tokenize

        let tokenized_batches = (0..input_len)
            .into_par_iter()
            .step_by(batch_size)
            .map(|batch_i| {
                let max_range = std::cmp::min(batch_i + batch_size, input_len);
                let range = batch_i..max_range;

                log::info!(
                    "Batch {}/{}, size {}",
                    (batch_i as f64 / batch_size as f64).ceil() as usize + 1,
                    (input_len as f64 / batch_size as f64).ceil() as usize,
                    max_range - batch_i
                );

                let (tokenized_input, attention) = tokenizer.tokenize(&sorted_pad_input[range]);

                let batch_tensor = Tensor::stack(&tokenized_input, 0).to(device);
                let batch_attention = Tensor::stack(&attention, 0).to(device);

                (batch_tensor, batch_attention)
            })
            .collect::<Vec<(Tensor, Tensor)>>();

        // Embed

        let mut batch_tensors = Vec::<Embeddings>::with_capacity(input_len);

        for (batch_tensor, batch_attention) in tokenized_batches.into_iter() {
            let batch_attention_c = batch_attention.shallow_clone();

            let classification_logits = self
                .lm_model
                .forward_t(
                    Some(batch_tensor),
                    Some(batch_attention_c),
                    None,
                    None,
                    None,
                    false,
                )
                .logits;

            batch_tensors.extend(Vec::<Embeddings>::from(classification_logits));
        }

        // Sort results

        let sorted_pad_input_idx = pad_sort(&sorted_pad_input_idx);

        let batch_tensors = sorted_pad_input_idx
            .into_iter()
            .map(|i| mem::replace(&mut batch_tensors[i], vec![]))
            .collect::<Vec<_>>();

        Ok(batch_tensors)
    }

    pub fn tokenizer(&self) -> Arc<T> {
        self.tokenizer.clone()
    }
}

fn pad_sort<O: Ord>(arr: &[O]) -> Vec<usize> {
    let mut idx = (0..arr.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| arr[i].cmp(&arr[j]));
    idx
}
