use std::env;
use std::path::PathBuf;
use std::sync::Mutex;

use sbert_rs::Error;

use tonic::{transport::Server, Request, Response, Status};

use service::embedder_server::{Embedder, EmbedderServer};

pub mod service {
    tonic::include_proto!("services.embedder");
}

pub struct SBert {
    model: sbert_rs::SBert,
}

impl SBert {
    pub fn new() -> Result<Self, Error> {
        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distiluse-base-multilingual-cased");

        println!("Loading sbert_rs ...");
        let model = sbert_rs::SBert::new(home).unwrap();

        Ok(SBert { model })
    }
}

unsafe impl Send for SBert {}

struct SBertSync(Mutex<SBert>);

#[tonic::async_trait]
impl Embedder for SBertSync {
    async fn vectorize(
        &self,
        query: Request<service::Query>,
    ) -> Result<Response<service::Response>, Status> {
        let texts = vec!["TTThis player needs tp be reported lolz."; 100];

        println!("Encoding {:?}...", texts[0]);
        let output = self.0.lock().unwrap().model.encode(&texts).unwrap();

        let r = output.get(0).slice(0, 0, 5, 1);
        r.print();

        let vector = service::Vector { v: vec![0.0; 10] };
        let vec_vectors = vec![vector; 10];

        let reply = service::Response { vecs: vec_vectors };
        println!("reply: {:?}", reply);

        Ok(Response::new(reply))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50050".parse()?;
    println!("Starting SBert server on {}", addr);
    let embedder = SBert::new()?;
    let embedder = SBertSync(Mutex::new(embedder));

    Server::builder()
        .add_service(EmbedderServer::new(embedder))
        .serve(addr)
        .await?;

    Ok(())
}
