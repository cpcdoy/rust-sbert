use crossbeam;
use math::round::ceil;
use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};

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

unsafe impl std::marker::Sync for SafeTensor {}
unsafe impl std::marker::Send for SafeTensor {}

pub struct SafeTensor {
    tensor: Arc<Mutex<Tensor>>,
}

pub struct SafeSBert<T: Tokenizer> {
    sbert: SBert<T>,
    tokenizer: T,
    device: Device,
}

unsafe impl<T: Tokenizer> std::marker::Sync for SafeSBert<T> {}
unsafe impl<T: Tokenizer> std::marker::Send for SafeSBert<T> {}

impl<T: Tokenizer> SafeSBert<T> {
    pub fn new<P: Into<PathBuf>>(root: P) -> Result<Self, Error> {
        let root = root.into();
        let root_c = root.clone();
        let model_dir = root_c.join("0_DistilBERT");
        let vocab_file = model_dir.join("vocab.txt");
        let tokenizer = T::new(&vocab_file)?;

        let device = Device::cuda_if_available();

        Ok(SafeSBert {
            sbert: SBert::new(root).unwrap(),
            tokenizer,
            device,
        })
    }

    pub fn tokenize_batch<S: AsRef<str>>(
        &self,
        sorted_pad_input: &[S],
    ) -> (SafeTensor, SafeTensor) {
        let (tokenized_input, attention) = self.tokenizer.tokenize(sorted_pad_input);

        let attention = Tensor::stack(&attention, 0).pin_memory();
        let batch_attention = SafeTensor {
            tensor: Arc::new(Mutex::new(attention.to4(
                self.device,
                attention.kind(),
                true,
                false,
            ))),
        };
        let tokens = Tensor::stack(&tokenized_input, 0).pin_memory();
        let batch_tensor = SafeTensor {
            tensor: Arc::new(Mutex::new(tokens.to4(
                self.device,
                tokens.kind(),
                true,
                false,
            ))),
        };

        (batch_attention, batch_tensor)
    }

    pub fn forward_batch(&self, batch_attention: SafeTensor, batch_tensor: SafeTensor) -> Tensor {
        let batch_attention_c = (*batch_attention.tensor).lock().unwrap().shallow_clone();

        let (embeddings, _, _) = self
            .sbert
            .forward_t(
                Some((*batch_tensor.tensor.lock().unwrap()).shallow_clone()),
                Some((*batch_attention.tensor.lock().unwrap()).shallow_clone()),
            )
            .map_err(Error::Encoding)
            .unwrap();

        let mean_pool = self.sbert.pooling.forward(&embeddings, &batch_attention_c);
        let linear_tanh = self.sbert.dense.forward(&mean_pool);

        linear_tanh
    }

    pub fn encode<S: AsRef<str>, B: Into<Option<usize>>>(
        &self,
        input: &[S],
        batch_size: B,
    ) -> Result<Vec<Vec<f32>>, Error> {
        self.sbert.encode(input, batch_size)
    }

    pub fn par_encode<S: AsRef<str> + Send + Sync, B: Into<Option<usize>>>(
        &self,
        input: &[S],
        batch_size: B,
    ) -> Result<Vec<Vec<f32>>, Error> {
        let batch_size = batch_size.into().unwrap_or_else(|| 64);

        let _guard = tch::no_grad_guard();

        let sorted_pad_input_idx = self
            .sbert
            .pad_sort(&input.iter().map(|e| e.as_ref().len()).collect::<Vec<_>>());
        let sorted_pad_input = sorted_pad_input_idx
            .iter()
            .map(|i| &input[*i])
            .collect::<Vec<_>>();

        let input_len = sorted_pad_input.len();

        let (tx_tok, rx_model) = channel::<(SafeTensor, SafeTensor)>();
        let (tx_model, rx_gather) = channel::<SafeTensor>();

        let batch_tensors = crossbeam::scope(|scope| {
            let tok = scope.spawn(move || {
                for batch_i in (0..input_len).step_by(batch_size) {
                    let max_range = std::cmp::min(batch_i + batch_size, input_len);
                    let range = batch_i..max_range;

                    log::info!(
                        "Scheduled batch {}/{}, size {}",
                        ceil((batch_i as f64) / (batch_size as f64), 0) as usize + 1,
                        ceil((input_len as f64) / (batch_size as f64), 0) as usize,
                        max_range - batch_i
                    );

                    let batch = self.tokenize_batch(&sorted_pad_input[range]);

                    tx_tok
                        .send(batch)
                        .expect("Unable to send on tokens through channel");
                }
            });

            let gather = scope.spawn(move || {
                let mut batch_tensors: Vec<Vec<f32>> = Vec::new();
                batch_tensors.reserve_exact(input_len);

                let embs_rx = rx_gather.iter();
                for emb in embs_rx {
                    let emb = (*emb.tensor.lock().unwrap()).shallow_clone();
                    let embeddings_cpu = emb.to4(self.device, emb.kind(), true, false);
                    batch_tensors.extend(Vec::<Vec<f32>>::from(embeddings_cpu));
                }
                batch_tensors
            });

            let batches_rx = rx_model.iter();
            for b in batches_rx {
                let (batch_attention, batch_tensor) = b;

                let embeddings = self.forward_batch(batch_attention, batch_tensor);
                let safe_embeddings = SafeTensor {
                    tensor: Arc::new(Mutex::new(embeddings)),
                };

                tx_model
                    .send(safe_embeddings)
                    .expect("Unable to send embeddings through channel");
            }
            drop(tx_model);

            tok.join();
            log::info!("Gathering batches...");
            gather.join()
        });

        let sorted_pad_input_idx = self.sbert.pad_sort(&sorted_pad_input_idx);
        let batch_tensors = sorted_pad_input_idx
            .into_iter()
            .map(|i| batch_tensors[i].clone())
            .collect::<Vec<_>>();

        Ok(batch_tensors)
    }
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
        //let device = Device::Cpu;
        log::info!("Using device {:?}", device);

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

    pub fn pad_sort<O: Ord>(&self, arr: &[O]) -> Vec<usize> {
        let mut idx = (0..arr.len()).collect::<Vec<_>>();
        idx.sort_unstable_by(|&i, &j| arr[i].cmp(&arr[j]));
        idx
    }

    pub fn tokenize_batch<S: AsRef<str>>(
        &self,
        sorted_pad_input: &[S],
    ) -> (SafeTensor, SafeTensor) {
        let (tokenized_input, attention) = self.tokenizer.tokenize(sorted_pad_input);

        let attention = Tensor::stack(&attention, 0).pin_memory();
        let batch_attention = SafeTensor {
            tensor: Arc::new(Mutex::new(attention.to4(
                self.device,
                attention.kind(),
                true,
                false,
            ))),
        };
        let tokens = Tensor::stack(&tokenized_input, 0).pin_memory();
        let batch_tensor = SafeTensor {
            tensor: Arc::new(Mutex::new(tokens.to4(
                self.device,
                tokens.kind(),
                true,
                false,
            ))),
        };

        (batch_attention, batch_tensor)
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

            log::info!(
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
