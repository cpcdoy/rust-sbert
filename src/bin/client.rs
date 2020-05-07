use service::embedder_client::EmbedderClient;

pub mod service {
    tonic::include_proto!("services.embedder");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EmbedderClient::connect("http://[::1]:50050").await?;

    // for i in 1..100 {
    let request = tonic::Request::new(service::Query {
        texts: vec!["TTThis player needs tp be reported lolz.".to_string(), "a".to_string(), "b".to_string(), "c".to_string()],
    });

    let response = client.vectorize(request).await?;

    for i in response.into_inner().vecs.iter() {
        println!("Response: [{:?}, {:?}, {:?}, {:?}, {:?}]", i.v[0], i.v[1], i.v[2], i.v[3], i.v[4]);
    }
    //println!("Len: {:?}", response.into_inner().vecs.len());
    //}

    Ok(())
}
