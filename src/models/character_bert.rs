use std::borrow::Borrow;
use std::collections::HashMap;

use rust_bert::bert::{BertConfig, BertEncoder, BertEncoderOutput, BertPooler};
use rust_bert::{Activation, Config, RustBertError};
use serde::{Deserialize, Serialize};
use tch::nn::Module;
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

pub struct Highway {
    layers: Vec<nn::Linear>,
    activation: Activation,
}

impl Highway {
    pub fn new<'p, P>(p: P, input_dim: i64, num_layers: i64, activation: Activation) -> Self
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow() / "_layers";
        let mut layers = Vec::<nn::Linear>::with_capacity(num_layers as usize);
        for i in 0..num_layers {
            let lin_conf = nn::LinearConfig::default();
            let linear = nn::linear(
                p.borrow() / i.to_string().as_str(),
                input_dim,
                input_dim * 2,
                lin_conf,
            );

            layers.push(linear);
        }

        Self { layers, activation }
    }

    pub fn forward(&self, inputs: Tensor) -> Result<Tensor, RustBertError> {
        let mut current_input = inputs;
        for layer in &self.layers {
            let projected_input = layer.forward(&current_input);
            let linear_part = current_input.shallow_clone();

            let linear_part_chunk = projected_input.chunk(2, -1);
            let (nonlinear_part, gate) = (
                linear_part_chunk[0].shallow_clone(),
                linear_part_chunk[1].shallow_clone(),
            );
            let nonlinear_part = match self.activation {
                Activation::relu => nonlinear_part.relu(),
                Activation::tanh => nonlinear_part.tanh(),
                _ => nonlinear_part.relu(),
            };

            let gate = gate.sigmoid();
            current_input = gate.shallow_clone() * linear_part + (1 - gate) * nonlinear_part;
        }

        Ok(current_input)
    }
}

////////////////////////////////////////////////////////////////////////////////////////

pub struct CharacterCNNOptions {
    activation: Activation,
    filters: [[i64; 2]; 7],
    n_highways: i64,
    embedding_dim: i64,
    n_characters: i64,
    max_characters_per_token: i64,
}

pub struct CharacterCNN {
    options: CharacterCNNOptions,
    char_embeddings: Tensor,
    char_embeddings_conf: nn::EmbeddingConfig,
    convolutions: Vec<nn::Conv1D>,
    highway: Highway,
    projection: nn::Linear,
    //
    // Defined in the original CharacterCNN but not here
    //requires_grad: bool,
    //beginning_of_sentence_characters: Tensor
    //end_of_sentence_characters: Tensor,
}

impl CharacterCNN {
    pub fn new<'p, P>(p: P, conf: &CharacterBertConfig) -> Self
    where
        P: Borrow<nn::Path<'p>>,
    {
        let options = CharacterCNNOptions {
            activation: Activation::relu,
            filters: [
                [1, 32],
                [2, 32],
                [3, 64],
                [4, 128],
                [5, 256],
                [6, 512],
                [7, 1024],
            ],
            n_highways: 2,
            embedding_dim: 16,
            n_characters: 65536,
            max_characters_per_token: 50,
        };

        //let output_dim = conf.hidden_size;

        // Init char embeddings
        let char_embeddings_conf = nn::EmbeddingConfig::default();
        // let char_embeddings = nn::embedding(
        //     p.borrow() / "_char_embedding_weights",
        //     options.n_characters + 1,
        //     options.embedding_dim,
        //     emb_conf,
        // );

        let char_embeddings = p.borrow().var(
            "_char_embedding_weights",
            &[options.n_characters + 1, options.embedding_dim],
            char_embeddings_conf.ws_init,
        );

        // Init CNNs weights
        let mut convolutions = Vec::<nn::Conv1D>::with_capacity(options.filters.len());
        for (i, [width, num]) in options.filters.iter().enumerate() {
            let mut conv_conf = nn::ConvConfig::default();
            conv_conf.bias = true;

            let conv = nn::conv1d(
                p.borrow() / format!("char_conv_{}", i).as_str(),
                options.embedding_dim,
                *num,
                *width,
                conv_conf,
            );

            convolutions.push(conv);
        }

        // Init highway
        let n_filters = options.filters.iter().fold(0, |sum, i| sum + i[1]);
        let highway = Highway::new(
            p.borrow() / "_highways",
            n_filters,
            options.n_highways,
            Activation::relu,
        );

        // Init projection
        let lin_conf = nn::LinearConfig::default();
        let projection = nn::linear(
            p.borrow() / "_projection",
            n_filters,
            conf.hidden_size,
            lin_conf,
        );

        Self {
            options,
            char_embeddings,
            char_embeddings_conf,
            convolutions,
            highway,
            projection,
        }
    }

    pub fn forward(&self, inputs: Tensor) -> Result<Tensor, RustBertError> {
        let mask = (inputs.ge(0))
            .to_kind(Kind::Int64)
            .sum1(&[-1], false, Kind::Int64);

        let (character_ids_with_bos_eos, _mask_with_bos_eos) = (inputs, mask);

        let max_chars_per_token = self.options.max_characters_per_token;

        // let character_embedding = self
        //     .char_embeddings
        //     .forward(&character_ids_with_bos_eos.view([-1, max_chars_per_token]));

        let character_embedding = Tensor::embedding(
            &self.char_embeddings,
            &character_ids_with_bos_eos.view([-1, max_chars_per_token]),
            self.char_embeddings_conf.padding_idx,
            self.char_embeddings_conf.scale_grad_by_freq,
            self.char_embeddings_conf.sparse,
        );

        let character_embedding = character_embedding.transpose(1, 2);
        let mut convs = Vec::<Tensor>::with_capacity(self.convolutions.len());
        for conv in self.convolutions.iter() {
            let convolved = conv.forward(&character_embedding);

            let (convolved, _) = convolved.max2(-1, false);
            let convolved = match self.options.activation {
                Activation::relu => convolved.relu(),
                Activation::tanh => convolved.tanh(),
                _ => convolved.relu(),
            };

            convs.push(convolved);
        }

        let mut token_embedding = Tensor::cat(&convs, -1);
        token_embedding = self.highway.forward(token_embedding).unwrap();
        token_embedding = self.projection.forward(&token_embedding);

        let (batch_size, sequence_length) = (
            character_ids_with_bos_eos.size()[0],
            character_ids_with_bos_eos.size()[1],
        );

        Ok(token_embedding.view([batch_size, sequence_length, -1]))
    }
}

////////////////////////////////////////////////////////////////////////////////////////

pub struct BertCharacterEmbeddings {
    word_embeddings: CharacterCNN,
    position_embeddings: nn::Embedding,
    token_type_embeddings: nn::Embedding,
    layer_norm: nn::LayerNorm,
    hidden_dropout_prob: f64,
}

impl BertCharacterEmbeddings {
    pub fn new<'p, P>(p: P, conf: &CharacterBertConfig) -> Self
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let CharacterBertConfig {
            hidden_size,
            max_position_embeddings,
            type_vocab_size,
            hidden_dropout_prob,
            ..
        } = *conf;

        let word_embeddings = CharacterCNN::new(p / "word_embeddings", conf);

        let emb_conf = nn::EmbeddingConfig::default();
        let position_embeddings = nn::embedding(
            p / "position_embeddings",
            max_position_embeddings,
            hidden_size,
            emb_conf,
        );

        let emb_conf = nn::EmbeddingConfig::default();
        let token_type_embeddings = nn::embedding(
            p / "token_type_embeddings",
            type_vocab_size,
            hidden_size,
            emb_conf,
        );

        let mut layer_norm_conf = nn::LayerNormConfig::default();
        layer_norm_conf.eps = 1e-12;

        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![hidden_size], layer_norm_conf.clone());

        Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            hidden_dropout_prob,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        position_ids: Option<Tensor>,
        train: bool,
    ) -> Result<Tensor, RustBertError> {
        let seq_length = input_ids.size()[1];

        let position_ids = position_ids.unwrap_or_else(|| {
            let position_ids = Tensor::arange(seq_length, (Kind::Int64, input_ids.device()));
            // println!("position_ids: {:?}", position_ids.size());
            // println!("input_ids: {:?}", input_ids.size());
            position_ids
                .unsqueeze(0)
                .expand_as(&input_ids.slice(2, 0, 1, 1).squeeze1(-1))
        });

        // println!("-1");

        // In original implem, but redundant...
        // let token_type_ids =
        //     token_type_ids.unwrap_or_else(|| input_ids.slice(2, 0, 1, 1).zeros_like());

        let word_embeddings = self.word_embeddings.forward(input_ids).unwrap();
        // println!("-2");
        let position_embeddings = self.position_embeddings.forward(&position_ids);
        // println!("-3");
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids);
        // println!("-4");
        let mut embeddings = word_embeddings + position_embeddings + token_type_embeddings;
        embeddings = self.layer_norm.forward(&embeddings);
        // println!("-5");
        embeddings = embeddings.dropout(self.hidden_dropout_prob, train);
        // println!("-6");

        Ok(embeddings)
    }
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
        let p = p.borrow();
        let bert_conf = conf.as_bert_config();

        let is_decoder = conf.is_decoder.unwrap_or(false);

        let embeddings = BertCharacterEmbeddings::new(p / "embeddings", conf);
        let encoder = BertEncoder::new(p / "encoder", &bert_conf);
        let pooler = BertPooler::new(p / "pooler", &bert_conf);

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
        // println!("1");
        let (input_shape, device) = match (&input_ids, &input_embeds) {
            (Some(_), Some(_)) => Err(RustBertError::ValueError(
                "You cannot specify both input_ds and inputs_embeds at the same time".into(),
            ))?,
            (Some(input_value), None) => (
                vec![input_value.size()[0], input_value.size()[1]],
                input_value.device(),
            ),
            (None, Some(embeds)) => (vec![embeds.size()[0], embeds.size()[1]], embeds.device()),
            (None, None) => Err(RustBertError::ValueError(
                "You have to specify either input_ids or inputs_embeds".into(),
            ))?,
        };

        let attention_mask =
            attention_mask.unwrap_or_else(|| Tensor::ones(&input_shape, (Kind::Int64, device)));
        let token_type_ids =
            token_type_ids.unwrap_or_else(|| Tensor::zeros(&input_shape, (Kind::Int64, device)));

        // println!("2");
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
                    let causal_mask = causal_mask.to_kind(attention_mask.kind());

                    causal_mask * attention_mask.unsqueeze(1).unsqueeze(2)
                } else {
                    attention_mask.unsqueeze(1).unsqueeze(2)
                }
            }
            _ => Err(RustBertError::ValueError(
                "Invalid attention mask or input_ids dimension, must be 2 or 3".into(),
            ))?,
        };

        // println!("3");
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

        let input_ids = input_ids.unwrap();

        // println!("4");
        let embedding_output =
            self.embeddings
                .forward_t(input_ids, token_type_ids, position_ids, train)?;

        // println!("5");
        let BertEncoderOutput {
            hidden_state,
            all_hidden_states,
            all_attentions,
        } = self.encoder.forward_t(
            &embedding_output,
            &Some(extended_attention_mask),
            encoder_hidden_states,
            &encoder_extended_attention_mask,
            train,
        );

        let sequence_output = hidden_state;
        let hidden_states = all_hidden_states;
        let attentions = all_attentions;

        // println!("6");
        let pooled_output = self.pooler.forward(&sequence_output);

        // println!("7");
        Ok(CharacterBertModelOutput {
            sequence_output,
            pooled_output,
            hidden_states,
            attentions,
        })
    }
}

pub struct CharacterBertClassificationHead {
    dense: nn::Linear,
    hidden_dropout_prob: f64,
    out_proj: nn::Linear,
}

impl CharacterBertClassificationHead {
    pub fn new<'p, P>(p: P, conf: &CharacterBertConfig) -> Self
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let lin_conf = nn::LinearConfig::default();
        let dense = nn::linear(p / "dense", conf.hidden_size, conf.hidden_size, lin_conf);

        let hidden_dropout_prob = conf.hidden_dropout_prob;

        let lin_conf = nn::LinearConfig::default();
        let out_proj = nn::linear(p / "out_proj", conf.hidden_size, conf.num_labels, lin_conf);

        Self {
            dense,
            hidden_dropout_prob,
            out_proj,
        }
    }

    pub fn forward_t(&self, features: &Tensor, train: bool) -> Tensor {
        let x = features.slice(1, 0, 1, 1);
        let x = x.dropout(self.hidden_dropout_prob, train);
        let x = self.dense.forward(&x);
        let x = x.tanh();
        let x = x.dropout(self.hidden_dropout_prob, train);
        let x = self.out_proj.forward(&x);

        x
    }
}

pub struct CharacterBertForSequenceClassification {
    character_bert: CharacterBertModel,
    classifier: CharacterBertClassificationHead,
}

impl CharacterBertForSequenceClassification {
    pub fn new<'p, P>(p: P, config: &CharacterBertConfig) -> Self
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let character_bert = CharacterBertModel::new(p / "roberta", config);
        let classifier = CharacterBertClassificationHead::new(p / "classifier", config);

        Self {
            character_bert,
            classifier,
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
    ) -> Result<CharacterBertSequenceClassificationOutput, RustBertError> {
        let character_bert_output = self.character_bert.forward_t(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            input_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            train,
        )?;

        let logits = self
            .classifier
            .forward_t(&character_bert_output.sequence_output, train);

        Ok(CharacterBertSequenceClassificationOutput {
            logits,
            character_bert_output,
        })
    }
}

pub struct CharacterBertSequenceClassificationOutput {
    pub logits: Tensor,
    pub character_bert_output: CharacterBertModelOutput,
}
