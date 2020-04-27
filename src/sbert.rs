use std::path::PathBuf;

use rust_bert::distilbert::{DistilBertConfig, DistilBertModel};
use rust_bert::Config;
use rust_tokenizers::bert_tokenizer::BertTokenizer;
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::{Tokenizer, TruncationStrategy};
use tch::{nn, no_grad, Device, Tensor};

use crate::layers::{Dense, Pooling};
use crate::Error;

pub struct SBert {
    lm_model: DistilBertModel,
    pooling: Pooling,
    dense: Dense,
    tokenizer: BertTokenizer,
}

impl SBert {
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
        let mut vs = nn::VarStore::new(device);

        let tokenizer = BertTokenizer::from_file(&vocab_file.to_string_lossy(), false);
        let lm_model = DistilBertModel::new(&vs.root(), &config);

        vs.load(weights_file).map_err(|e| Error::VarStore(e))?;

        Ok(SBert {
            lm_model,
            pooling,
            dense,
            tokenizer,
        })
    }

    pub fn encode(&self, input: &[&str]) -> Result<Tensor, Error> {
        let tokenized_input =
            self.tokenizer
                .encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);

        let max_len = tokenized_input
            .iter()
            .map(|input| input.token_ids.len())
            .max()
            .unwrap_or_else(|| 0);

        let tokenized_input = tokenized_input
            .into_iter()
            .map(|input| input.token_ids)
            .map(|mut input| {
                input.extend(vec![0; max_len - input.len()]);
                input
            })
            .map(|input| Tensor::of_slice(&(input)))
            .collect::<Vec<_>>();

        let device = Device::cuda_if_available();
        let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

        let (output, _, _) = self
            .forward_t(Some(input_tensor), None)
            .map_err(Error::Encoding)?;
        let mean_pool = self.pooling.forward(&output);
        let linear_tanh = self.dense.forward(&mean_pool);

        Ok(linear_tanh)
    }

    pub fn forward_t(
        &self,
        input: Option<Tensor>,
        mask: Option<Tensor>,
    ) -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (output, hidden_states, attention) =
            self.lm_model.forward_t(input, mask, None, false)?;
        Ok(no_grad(|| (output, hidden_states, attention)))
    }
}
