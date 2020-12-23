#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;
    use std::time::Instant;

    use rust_bert::Config;
    use tch::{nn, Device, Kind, Tensor};

    use sbert::{CharacterBertConfig, CharacterBertForSequenceClassification};

    #[test]
    fn test_character_bert_model() {
        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distilcharacteroberta");

        let device = Device::cuda_if_available();

        println!("Running on device {:?}", device);

        let config_file = home.join("config.json");
        let weights_file = home.join("model.ot");

        let mut vs = nn::VarStore::new(device);
        let config = CharacterBertConfig::from_file(config_file);
        let model = CharacterBertForSequenceClassification::new(&vs.root(), &config);
        //vs.variables()["classifier.out_proj.bias"].print();
        // println!("Varstore content before: {:?}", vs.variables());
        vs.load(weights_file).unwrap();
        //        let partials = vs.load_partial(weights_file);
        //        println!("partials: {:?}", partials);

        // println!("Varstore len: {:?}", vs.len());
        // println!("Varstore content: {:?}", vs.variables());
        // vs.variables()["classifier.out_proj.bias"].print();

        let _guard = tch::no_grad_guard();

        let input_shape = [1, 2, 50];
        let input_tensor = Tensor::cat(
            &[
                Tensor::zeros(&input_shape, (Kind::Int64, device)),
                Tensor::ones(&input_shape, (Kind::Int64, device)) * 3,
            ],
            0,
        );

        println!("Input tensor shape: {:?}", input_tensor.size());

        let before = Instant::now();
        let r = model
            .forward_t(
                Some(input_tensor),
                None,
                None,
                None,
                None,
                &None,
                &None,
                false,
            )
            .unwrap();

        println!("Elapsed time: {:?}ms", before.elapsed().as_millis() / 10);

        let r = r.logits.softmax(2, Kind::Float).squeeze1(1);

        let r = Vec::<Vec<f32>>::from((r * 10000.0).round() / 10000.0);
        assert_eq!(r, [[0.0491, 0.9509], [0.0076, 0.9924]]);
    }
}
