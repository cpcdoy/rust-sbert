use std::convert::TryFrom;
use std::mem;
use std::path::PathBuf;
use std::sync::Arc;

use rayon::prelude::*;
use rust_bert::distilbert::{DistilBertConfig, DistilBertModel};
use rust_bert::Config;
use tch::{nn, Device, Tensor};

use crate::layers::{Dense, Pooling};
use crate::models::pad_sort;
use crate::tokenizers::Tokenizer;
use crate::{att, Attentions, Embeddings, Error};

pub struct SBert<T> {
    lm_model: DistilBertModel,
    nb_layers: usize,
    nb_heads: usize,
    pooling: Pooling,
    dense: Dense,
    tokenizer: Arc<T>,
    device: Device,
}

impl<T> SBert<T>
where
    T: Tokenizer + Send + Sync,
{
    pub fn new<P>(root: P, device: Option<Device>) -> Result<Self, Error>
    where
        P: Into<PathBuf>,
    {
        let root = root.into();
        let model_dir = root.join("0_DistilBERT");

        let config_file = model_dir.join("config.json");
        let weights_file = model_dir.join("model.ot");
        let vocab_file = model_dir.join("vocab.txt");

        // Set-up DistilBert model and tokenizer

        let config = DistilBertConfig::from_file(&config_file);
        let nb_layers = config.n_layers as usize;
        let nb_heads = config.n_heads as usize;

        let device = device.unwrap_or(Device::cuda_if_available());
        log::info!("Using device {:?}", device);

        let pooling = Pooling::new(root.clone());
        let dense = Dense::new(root, device)?;



        let mut vs = nn::VarStore::new(device);

        let tokenizer = Arc::new(T::new(&vocab_file)?);
        let lm_model = DistilBertModel::new(&vs.root(), &config);

        vs.load(weights_file)?;

        Ok(SBert {
            lm_model,
            nb_layers,
            nb_heads,
            pooling,
            dense,
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

            let embeddings = self
                .lm_model
                .forward_t(Some(&batch_tensor), Some(&batch_attention), None, false)?
                .hidden_state;

            let mean_pool = self.pooling.forward(&embeddings, &batch_attention_c);
            let linear_tanh = self.dense.forward(&mean_pool);

            batch_tensors.extend(Vec::<Embeddings>::try_from(linear_tanh).unwrap());
        }

        // Sort results

        let sorted_pad_input_idx = pad_sort(&sorted_pad_input_idx);

        let batch_tensors = sorted_pad_input_idx
            .into_iter()
            .map(|i| mem::replace(&mut batch_tensors[i], vec![]))
            .collect::<Vec<_>>();

        Ok(batch_tensors)
    }

    pub fn forward_with_attention<S, B>(
        &self,
        input: &[S],
        batch_size: B,
    ) -> Result<(Vec<Embeddings>, Attentions), Error>
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

                let batch_len = max_range - batch_i;
                let batch_tensor = Tensor::stack(&tokenized_input, 0).to(device);
                let batch_attention = Tensor::stack(&attention, 0).to(device);

                (batch_len, batch_tensor, batch_attention)
            })
            .collect::<Vec<(usize, Tensor, Tensor)>>();

        // Embed

        // type Attentions = Vec<Layers<Heads<Attention2D>>>>
        let mut batch_attention_tensors = Attentions::with_capacity(self.nb_layers);
        let mut batch_tensors = Vec::<Embeddings>::with_capacity(input_len);

        for (batch_len, batch_tensor, batch_attention) in tokenized_batches.into_iter() {
            let batch_attention_c = batch_attention.shallow_clone();

            let output = self.lm_model.forward_t(
                Some(&batch_tensor),
                Some(&batch_attention),
                None,
                false,
            )?;

            let embeddings = output.hidden_state;
            let attention = output.all_attentions;

            let mean_pool = self.pooling.forward(&embeddings, &batch_attention_c);
            let linear_tanh = self.dense.forward(&mean_pool);

            batch_tensors.extend(Vec::<Embeddings>::try_from(linear_tanh).unwrap());

            let attention = attention.ok_or_else(|| Error::Encoding("No attention"))?;
            for i in 0..batch_len as i64 {
                let mut layers_att = att::Layers::with_capacity(self.nb_layers);

                for layer in attention.iter() {
                    let mut heads_att = att::Heads::with_capacity(self.nb_heads);

                    for head in 0..self.nb_heads as usize {
                        let att_slice = layer
                            .slice(0, i, i + 1, 1)
                            .slice(1, head as i64, head as i64 + 1, 1)
                            .squeeze();

                        let head_att = att::Attention2D::try_from(att_slice).unwrap();
                        heads_att.push(head_att);
                    }
                    layers_att.push(heads_att);
                }
                batch_attention_tensors.push(layers_att);
            }
        }

        // Sort results

        let sorted_pad_input_idx = pad_sort(&sorted_pad_input_idx);

        let batch_tensors = sorted_pad_input_idx
            .iter()
            .map(|i| mem::replace(&mut batch_tensors[*i], vec![]))
            .collect::<Vec<_>>();

        let batch_attention_tensors = sorted_pad_input_idx
            .iter()
            .map(|i| mem::replace(&mut batch_attention_tensors[*i], vec![]))
            .collect::<Vec<_>>();

        Ok((batch_tensors, batch_attention_tensors))
    }

    pub fn tokenizer(&self) -> Arc<T> {
        self.tokenizer.clone()
    }
}
