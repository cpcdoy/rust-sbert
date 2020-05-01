use std::path::PathBuf;

use rust_bert::distilbert::{DistilBertConfig, DistilBertModel};
use rust_bert::Config;
use tch::{nn, no_grad, Device, Tensor};

use crate::layers::{Dense, Pooling};
use crate::tokenizers::{Tokenizer};
use crate::Error;

pub struct SBert<T: Tokenizer> {
    lm_model: DistilBertModel,
    pooling: Pooling,
    dense: Dense,
    tokenizer: T,
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
        })
    }

    pub fn encode<S: AsRef<str>>(&self, input: &[S]) -> Result<Tensor, Error> {
        let device = Device::cuda_if_available();

        let tokenized_input = self.tokenizer.tokenize(input);
        let input_tensor = Tensor::stack(&tokenized_input.as_slice(), 0).to(device);

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
