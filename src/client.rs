use service::embedder_client::EmbedderClient;

pub mod service {
    tonic::include_proto!("services.embedder");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbedderClient::connect("http://[::1]:50050").await?;

    for i in 1..100 {
        let request = tonic::Request::new(service::Query {
            texts: vec!["TTThis player needs tp be reported lolz.".to_string(); i * 2],
        });

        println!("Request: {}", i * 2);
        let response = client.vectorize(request).await?;

        println!("Response: {:?}", response.into_inner().vecs.len());
    }

    Ok(())
}
