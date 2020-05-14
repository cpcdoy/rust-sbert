use service::embedder_client::EmbedderClient;

use std::time::Instant;

pub mod service {
    tonic::include_proto!("services.embedder");
}

use rand::random;
fn rand_string() -> String {
    (0..(random::<f32>() * 100.0) as usize)
        .map(|_| (0x20u8 + (random::<f32>() * 96.0) as u8) as char)
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbedderClient::connect("http://[::1]:50050").await?;

    let now = Instant::now();
    for i in 1..10 {
        let mut texts = Vec::new();
        for _ in 0..1000 {
            texts.push(rand_string());
        }
        //println!("Texts: {:?}", texts);
        println!("Batch {}", i);
        let request = tonic::Request::new(service::Query {
            //texts: vec!["TTThis player needs tp be reported lolz.".to_string(), "a".to_string(), "b".to_string(), "c".to_string()],
            texts: texts,
        });

        let response = client.vectorize(request).await?;

        for i in response.into_inner().vecs[..5].iter() {
            println!(
                "Response: [{:?}, {:?}, {:?}, {:?}, {:?}]",
                i.v[0], i.v[1], i.v[2], i.v[3], i.v[4]
            );
        }
    }

    println!("Time elapsed: {}ms", now.elapsed().as_millis());

    Ok(())
}
