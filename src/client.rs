use service::embedder_client::{EmbedderClient};

pub mod service {
    tonic::include_proto!("services.embedder");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbedderClient::connect("http://[::1]:50050").await?;

    let request = tonic::Request::new(service::Query {
        texts: vec!["test".to_string(); 1],
    });

    let response = client.vectorize(request).await?;

    println!("RESPONSE={:?}", response);

    Ok(())
}