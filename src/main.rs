extern crate dirs;
extern crate rust_bert;
extern crate tch;

use tch::{nn, no_grad, Device, Tensor, Kind};

use failure::err_msg;
use rust_bert::distilbert::{DistilBertConfig, DistilBertModel};
use rust_bert::Config;
use rust_tokenizers::bert_tokenizer::BertTokenizer;
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::{Tokenizer, TruncationStrategy};

use std::path::PathBuf;
use std::env;

fn mean_pooling(token_embeddings: Tensor) -> Tensor {

    let attention_mask = token_embeddings.ones_like();
    let input_mask_expanded = attention_mask.expand_as(&token_embeddings);

    let mut output_vectors = Vec::new();
    let dim = [1];
    let mut sum_mask = input_mask_expanded.copy();
    sum_mask = sum_mask.sum1(&dim, false, Kind::Float);
    let sum_embeddings = (token_embeddings * input_mask_expanded).sum1(&dim, false, Kind::Float);

    println!("token emb size {:?}", sum_embeddings.size());
    println!("token emb sum dim1 {:?}", sum_embeddings);
    println!("token emb sum dim1 {:?}", sum_embeddings.get(0).get(0));
    println!("token emb sum dim1 {:?}", sum_embeddings.get(0).get(1));
    println!("token emb sum dim1 {:?}", sum_embeddings.get(0).get(2));

    output_vectors.push(sum_embeddings / sum_mask);

    let output_vector = Tensor::cat(&output_vectors, 1);
    
    output_vector
}

fn main() -> failure::Fallible<()> {
    let device = Device::cuda_if_available();
    println!("device {:?}", device);

    let mut home: PathBuf = env::current_dir().unwrap();
    home.push("models");
    home.push("distiluse-base-multilingual-cased");
    home.push("0_DistilBERT");
    println!("Path {:?}", home);
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let weights_path = &home.as_path().join("model.ot");

    if !config_path.is_file() | !vocab_path.is_file() | !weights_path.is_file() {
        return Err(err_msg(
            "Could not find required resources to run example. \
                          Please run ../utils/prepare_distilbert.py \
                          in a Python environment with dependencies listed in ../requirements.txt",
        ));
    }

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), false);
    let config = DistilBertConfig::from_file(config_path);
    let distil_bert_model = DistilBertModel::new(&vs.root(), &config);
    vs.load(weights_path)?;

    // Dense
    let mut vs_dense = nn::VarStore::new(device);
    let cfg = nn::LinearConfig {
        ws_init: nn::Init::Const(0.),
        bs_init: Some(nn::Init::Const(0.)),
    };
    home.pop();
    home.push("2_Dense");
    let config_path = &home.as_path().join("config.json");
    let weights_path = &home.as_path().join("model.ot");
    let linear = nn::linear(&vs_dense.root(), 768, 512, cfg);
    vs_dense.load(weights_path)?;

    //    Define input
    let input = [
        "TTThis player needs tp be reported lolz."
    ];
    let tokenized_input =
        tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
    
    println!("tokenized {:?}", tokenized_input);
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
    println!("tokenized {:?}", tokenized_input);
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    println!("tensor {:?}", input_tensor);
    //    Forward pass
    let (mut output, hidden_states, attention) = no_grad(|| {
        distil_bert_model
            .forward_t(Some(input_tensor), None, None, false)
            .unwrap()
    });

    println!("output distilbert {:?}", output);
    println!("output distilbert {:?}", output.get(0).get(0).get(0));
    println!("output distilbert {:?}", output.get(0).get(0).get(1));
    println!("output distilbert {:?}", output.get(0).get(0).get(2));

    println!("hidden {:?}", hidden_states);
    println!("attention {:?}", attention);

    // Good until here
    output = mean_pooling(output);
    println!("sentence emb mean pool {:?}", output);
    println!("sentence emb mean pool {:?}", output.get(0).get(0));
    println!("sentence emb mean pool {:?}", output.get(0).get(1));
    println!("sentence emb mean pool {:?}", output.get(0).get(2));
    output = output.apply(&linear);
    output = output.tanh();

    println!("embeddings: {:?}", output);
    println!("embeddings: {:?}", output.get(0).get(0));
    println!("embeddings: {:?}", output.get(0).get(1));
    println!("embeddings: {:?}", output.get(0).get(2));

    Ok(())
}