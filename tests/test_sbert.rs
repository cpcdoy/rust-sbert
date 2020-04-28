#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;
    use std::time::Instant;

    use sbert_rs::SBert;

    #[test]
    fn test_sbert() {
        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distiluse-base-multilingual-cased");

        println!("Loading sbert_rs ...");
        let before = Instant::now();
        let sbert_model = SBert::new(home).unwrap();
        println!("Elapsed time: {:.2?}", before.elapsed());

        let texts = vec!["TTThis player needs tp be reported lolz."; 100];

        println!("Encoding {:?}...", texts[0]);
        let mut t = 0;
        let n = 10;
        let mut output = sbert_model.encode(&texts).unwrap();
        for _i in 0..n {
            let before = Instant::now();
            output = sbert_model.encode(&texts).unwrap();
            t = t + before.elapsed().as_millis();
        }
        println!("Elapsed time: {:?}ms", t / n);

        let r = output.get(0).slice(0, 0, 5, 1);
        r.print();

        let v = (r / 0.01)
            .iter::<f64>()
            .unwrap()
            .map(|f| (f * 10000.0).round() / 10000.0)
            .collect::<Vec<_>>();
        assert_eq!(v, [-2.2717, -0.6020, 5.5196, 1.8546, -7.5385]);
    }
}
