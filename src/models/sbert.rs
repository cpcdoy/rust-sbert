use rust_bert::distilbert::{DistilBertConfig, DistilBertModel};
use rust_bert::Config;
use rust_tokenizers::bert_tokenizer::BertTokenizer;
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::{Tokenizer, TruncationStrategy};

use std::path::PathBuf;

use tch::{nn, no_grad, Device, Tensor};

use crate::layers::dense::{Dense, DenseConfig};
use crate::layers::pooling::{Pooling, PoolingConfig};

use serde::{Deserialize, Serialize};

pub struct SBert<LM, Tok> {
    lm_model: LM,
    pooling: Pooling,
    dense: Dense,
    tokenizer: Tok,
    conf: DistilBertConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SBertConfig {
    pub distilbert_conf: DistilBertConfig,
}

impl Config<SBertConfig> for SBertConfig {}

impl SBert<DistilBertModel, BertTokenizer> {
    pub fn new(path: &str) -> SBert<DistilBertModel, BertTokenizer> {
        let device = Device::cuda_if_available();
        println!("Using device {:?}", device);
        
        let mut conf_copy = PathBuf::from(path);
        conf_copy.push("0_DistilBERT");
        println!("Loading conf {:?}", conf_copy);

        let config_path = &conf_copy.as_path().join("config.json");
        let vocab_path = &conf_copy.as_path().join("vocab.txt");
        let weights_path = &conf_copy.as_path().join("model.ot");
        
        assert_eq!(
            config_path.is_file() & vocab_path.is_file() & weights_path.is_file(),
            true
        );

        // Set-up DistilBert model and tokenizer

        let pooling = Pooling::new(path);
        let dense = Dense::new(path);
        
        let mut vs = nn::VarStore::new(device);
        let tokenizer: BertTokenizer =
            BertTokenizer::from_file(vocab_path.to_str().unwrap(), false);

        let conf = DistilBertConfig::from_file(config_path);
        let lm_model = DistilBertModel::new(&vs.root(), &conf);
        vs.load(weights_path).unwrap();

        SBert {
            lm_model,
            pooling,
            dense,
            tokenizer,
            conf
        }
    }

    pub fn encode(
        &self,
        input: Vec<&str>,
    ) -> Result<Tensor, &'static str> {
        
        let tokenized_input =
            self.tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);

        let max_len = tokenized_input
            .iter()
            .map(|input| input.token_ids.len())
            .max()
            .unwrap();
        
        let tokenized_input = tokenized_input
            .iter()
            .map(|input| input.token_ids.clone())
            .map(|mut input| {
                input.extend(vec![0; max_len - input.len()]);
                input
            })
            .collect::<Vec<_>>();

        let tokenized_input = tokenized_input
            .iter()
            .map(|input| Tensor::of_slice(&(input)))
            .collect::<Vec<_>>();
        
        let device = Device::cuda_if_available();

        let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

        let (output, _, _) = self.forward_t(Some(input_tensor), None).unwrap();
        let mean_pool = self.pooling.forward(&output);
        let linear_tanh = self.dense.forward(&mean_pool);
        
        Ok(linear_tanh)
    }

    pub fn forward_t(
        &self,
        input: Option<Tensor>,
        mask: Option<Tensor>,
    ) -> Result<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (output, hidden_states, attention) = no_grad(|| {
            (&self.lm_model)
                .forward_t(input, mask, None, false)
                .unwrap()
        });
        Ok((output, hidden_states, attention))
    }
}
