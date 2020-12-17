use std::borrow::Borrow;
use std::collections::HashMap;

use rust_bert::bert::BertEncoder;
use rust_bert::bert::BertPooler;
use rust_bert::bert::{BertConfig};
use rust_bert::{Config, RustBertError};
use serde::{Deserialize, Serialize};
use tch::{nn, Kind, Tensor};

////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Serialize, Deserialize)]
pub struct CharacterBertConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f32,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub layer_norm_eps: f64,
    pub num_labels: i64,
}

impl Config<CharacterBertConfig> for CharacterBertConfig {}

impl CharacterBertConfig {
    pub fn as_bert_config(&self) -> BertConfig {
        BertConfig {
            hidden_act: self.hidden_act,
            attention_probs_dropout_prob: self.attention_probs_dropout_prob,
            hidden_dropout_prob: self.hidden_dropout_prob,
            hidden_size: self.hidden_size,
            initializer_range: self.initializer_range,
            intermediate_size: self.intermediate_size,
            max_position_embeddings: self.max_position_embeddings,
            num_attention_heads: self.num_attention_heads,
            num_hidden_layers: self.num_hidden_layers,
            type_vocab_size: self.type_vocab_size,
            vocab_size: self.vocab_size,
            output_attentions: self.output_attentions,
            output_hidden_states: self.output_hidden_states,
            is_decoder: self.is_decoder,
            id2label: self.id2label.clone(),
            label2id: self.label2id.clone(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////

pub struct BertCharacterEmbeddings {
    word_embeddings: CharacterCNN,
    position_embeddings: nn::Embedding,
    token_type_embeddings: nn::Embedding,
    layer_norm: nn::LayerNorm,
    dropout: u32,
}

impl BertCharacterEmbeddings {
    
}

////////////////////////////////////////////////////////////////////////////////////////

pub struct CharacterBertModelOutput {
    pub sequence_output: Tensor,
    pub pooled_output: Tensor,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}

pub struct CharacterBertModel {
    is_decoder: bool,

    embeddings: BertCharacterEmbeddings,
    encoder: BertEncoder,
    pooler: BertPooler,
}

impl CharacterBertModel {
    pub fn new<'p, P>(p: P, conf: &CharacterBertConfig) -> CharacterBertModel
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow() / "roberta";
        let bert_conf = conf.as_bert_config();

        let is_decoder = conf.is_decoder.unwrap_or(false);

        let embeddings = BertCharacterEmbeddings::new(&p / "embeddings", conf);
        let encoder = BertEncoder::new(&p / "encoder", conf);
        let pooler = BertPooler::new(&p / "poolr", &bert_conf);

        CharacterBertModel {
            is_decoder,
            embeddings,
            encoder,
            pooler,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<Tensor>,
        attention_mask: Option<Tensor>,
        token_type_ids: Option<Tensor>,
        position_ids: Option<Tensor>,
        input_embeds: Option<Tensor>,
        encoder_hidden_states: &Option<Tensor>,
        encoder_attention_mask: &Option<Tensor>,
        train: bool,
    ) -> Result<CharacterBertModelOutput, RustBertError> {
        let (input_shape, device) = match (&input_ids, &input_embeds) {
            (Some(_), Some(_)) => Err(RustBertError::ValueError(
                "You cannot specify both input_ds and inputs_embeds at the same time".into(),
            ))?,
            (Some(input_value), None) => (input_value.size(), input_value.device()),
            (None, Some(embeds)) => (vec![embeds.size()[0], embeds.size()[1]], embeds.device()),
            (None, None) => Err(RustBertError::ValueError(
                "You have to specify either input_ids or inputs_embeds".into(),
            ))?,
        };

        let attention_mask =
            attention_mask.unwrap_or_else(|| Tensor::ones(&input_shape, (Kind::Int64, device)));
        let token_type_ids =
            token_type_ids.unwrap_or_else(|| Tensor::zeros(&input_shape, (Kind::Int64, device)));

        let extended_attention_mask = match attention_mask.dim() {
            3 => attention_mask.unsqueeze(1),
            2 => {
                if self.is_decoder {
                    let (batch_size, seq_length) = (input_shape[0], input_shape[1]);
                    let seq_ids = Tensor::arange(seq_length, (Kind::Float, device));
                    let causal_mask = seq_ids
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(&[batch_size, seq_length, 1]);
                    let causal_mask = causal_mask.le1(&seq_ids.unsqueeze(0).unsqueeze(-1));
                    causal_mask = causal_mask.to_kind(attention_mask.kind());

                    causal_mask * attention_mask.unsqueeze(1).unsqueeze(2)
                } else {
                    attention_mask.unsqueeze(1).unsqueeze(2)
                }
            }
            _ => Err(RustBertError::ValueError(
                "Invalid attention mask or input_ids dimension, must be 2 or 3".into(),
            ))?,
        };

        // Should do this for fp16 compat if needed
        //let extended_attention_mask = extended_attention_mask.to_kind(self.parameters.kind());
        let extended_attention_mask: Tensor = (1.0 - extended_attention_mask) * -10000.0;
        // (extended_attention_mask.ones_like() - extended_attention_mask) * -10000.0;

        // Make broadcastable to [batch_size, num_heads, seq_length, seq_length] for decoder cross_attention
        let encoder_extended_attention_mask: Option<Tensor> =
            if self.is_decoder && encoder_hidden_states.is_some() {
                let encoder_hidden_states = encoder_hidden_states.as_ref().unwrap();
                let encoder_hidden_states_shape = encoder_hidden_states.size();
                let (encoder_batch_size, encoder_sequence_length) = (
                    encoder_hidden_states_shape[0],
                    encoder_hidden_states_shape[1],
                );

                //let encoder_hidden_shape = [encoder_batch_size, encoder_sequence_length];

                let encoder_attention_mask = match encoder_attention_mask {
                    Some(value) => value.copy(),
                    None => Tensor::ones(
                        &[encoder_batch_size, encoder_sequence_length],
                        (Kind::Int64, device),
                    ),
                };

                let encoder_extended_attention_mask = match encoder_attention_mask.dim() {
                    2 => encoder_attention_mask.unsqueeze(1).unsqueeze(2),
                    3 => encoder_attention_mask.unsqueeze(1),
                    _ => Err(RustBertError::ValueError(
                        "Invalid attention mask dimension, must be 2 or 3".into(),
                    ))?,
                };

                let encoder_extended_attention_mask =
                    (1.0 - encoder_extended_attention_mask) * -10000.0;

                Some(encoder_extended_attention_mask)
            } else {
                None
            };

        let embedding_output =
            self.embeddings
                .forward_t(input_ids, position_ids, token_type_ids)?;

        let CharacterBertEncoderOutput {
            sequence_output,
            hidden_states,
            attentions,
        } = self.encoder.forward_t(
            &embedding_output,
            &Some(extended_attention_mask),
            encoder_hidden_states,
            &encoder_extended_attention_mask,
            train,
        );

        let pooled_output = self.pooler.forward(&sequence_output);

        Ok(CharacterBertModelOutput {
            sequence_output,
            pooled_output,
            hidden_states,
            attentions,
        })
    }
}
