use math::round::ceil;
use std::path::PathBuf;

use rust_bert::distilbert::{DistilBertConfig, DistilBertModel};
use rust_bert::Config;
use tch::{nn, Device, Tensor};

use crate::layers::{Dense, Pooling};
use crate::tokenizers::Tokenizer;
use crate::Error;

pub struct SBert<T: Tokenizer> {
    lm_model: DistilBertModel,
    pooling: Pooling,
    dense: Dense,
    tokenizer: T,
    device: Device,
}

impl<T: Tokenizer> SBert<T> {
    pub fn new<P: Into<PathBuf>>(root: P) -> Result<Self, Error> {
        let root = root.into();
        let model_dir = root.join("0_DistilBERT");

        let config_file = model_dir.join("config.json");
        let weights_file = model_dir.join("model.ot");
        let vocab_file = model_dir.join("vocab.txt");

        // Set-up DistilBert model and tokenizer

        let config = DistilBertConfig::from_file(&config_file);

        let pooling = Pooling::new(root.clone());
        let dense = Dense::new(root.clone())?;

        let device = Device::cuda_if_available();
        println!("Using device {:?}", device);

        let mut vs = nn::VarStore::new(device);

        let tokenizer = T::new(&vocab_file)?;
        let lm_model = DistilBertModel::new(&vs.root(), &config);

        vs.load(weights_file).map_err(|e| Error::VarStore(e))?;

        Ok(SBert {
            lm_model,
            pooling,
            dense,
            tokenizer,
            device,
        })
    }

    fn pad_sort<O: Ord>(&self, arr: &[O]) -> Vec<usize> {
        let mut idx = (0..arr.len()).collect::<Vec<_>>();
        idx.sort_unstable_by(|&i, &j| arr[i].cmp(&arr[j]));
        idx
    }

    pub fn encode<S: AsRef<str>, B: Into<Option<usize>>>(
        &self,
        input: &[S],
        batch_size: B,
    ) -> Result<Vec<Vec<f32>>, Error> {
        let batch_size = batch_size.into().unwrap_or_else(|| 64);

        let _guard = tch::no_grad_guard();

        let sorted_pad_input_idx =
            self.pad_sort(&input.iter().map(|e| e.as_ref().len()).collect::<Vec<_>>());
        let sorted_pad_input = sorted_pad_input_idx
            .iter()
            .map(|i| &input[*i])
            .collect::<Vec<_>>();

        let input_len = sorted_pad_input.len();
        let mut batch_tensors: Vec<Vec<f32>> = Vec::new();
        batch_tensors.reserve_exact(input_len);

        for batch_i in (0..input_len).step_by(batch_size) {
            let max_range = std::cmp::min(batch_i + batch_size, input_len);
            let range = batch_i..max_range;

            println!(
                "Batch {}/{}, size {}",
                ceil((batch_i as f64) / (batch_size as f64), 0) as usize + 1,
                ceil((input_len as f64) / (batch_size as f64), 0) as usize,
                max_range - batch_i
            );

            let (tokenized_input, attention) = self.tokenizer.tokenize(&sorted_pad_input[range]);

            let batch_attention = Tensor::stack(&attention, 0).to(self.device);
            let batch_attention_c = batch_attention.shallow_clone();
            let batch_tensor = Tensor::stack(&tokenized_input, 0).to(self.device);

            let (embeddings, _, _) = self
                .forward_t(Some(batch_tensor), Some(batch_attention))
                .map_err(Error::Encoding)?;

            let mean_pool = self.pooling.forward(&embeddings, &batch_attention_c);
            let linear_tanh = self.dense.forward(&mean_pool);

            batch_tensors.extend(Vec::<Vec<f32>>::from(linear_tanh));
        }

        let sorted_pad_input_idx = self.pad_sort(&sorted_pad_input_idx);
        let batch_tensors = sorted_pad_input_idx
            .into_iter()
            .map(|i| batch_tensors[i].clone())
            .collect::<Vec<_>>();

        //let stack_batches = Tensor::cat(&batch_tensors, 0);
        Ok(batch_tensors)
    }

    fn forward_t(
        &self,
        input: Option<Tensor>,
        mask: Option<Tensor>,
    ) -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (output, hidden_states, attention) =
            self.lm_model.forward_t(input, mask, None, false)?;
        Ok((output, hidden_states, attention))
    }
}
