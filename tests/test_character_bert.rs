#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;
    use std::time::Instant;

    use torch_sys::dummy_cuda_dependency;

    use std::error::Error;

    use rust_bert::Config;
    use tch::{nn, Device, Tensor, Kind};

    use sbert::{CharacterBertForSequenceClassification, CharacterBertConfig};

    const BATCH_SIZE: usize = 64;

    #[test]
    fn test_character_bert_model() {
        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distilcharacteroberta");

        let device = Device::cuda_if_available();

        let config_file = home.join("config.json");
        let weights_file = home.join("model.ot");

        let mut vs = nn::VarStore::new(device);
        let config = CharacterBertConfig::from_file(config_file);
        let model = CharacterBertForSequenceClassification::new(&vs.root(), &config);
        vs.variables()["classifier.out_proj.bias"].print();
        // println!("Varstore content before: {:?}", vs.variables());
        vs.load(weights_file).unwrap();
//        let partials = vs.load_partial(weights_file);
//        println!("partials: {:?}", partials);

        println!("Varstore len: {:?}", vs.len());
        println!("Varstore content: {:?}", vs.variables());
        vs.variables()["classifier.out_proj.bias"].print();

        let _guard = tch::no_grad_guard();

        let input_shape = [1, 2, 50];
        let r = model.forward_t(Some(Tensor::ones(&input_shape, (Kind::Int64, device))), None, None, None, None, &None, &None, false).unwrap();

        vs.variables()["classifier.out_proj.bias"].print();

        r.logits.softmax(2, Kind::Float).print();
    }
}