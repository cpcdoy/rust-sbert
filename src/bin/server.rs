use std::env;
use std::path::PathBuf;
use tokio::sync::Mutex;

use sbert_rs::{Error, SBertHF};

use tonic::{transport::Server, Request, Response, Status};

use service::embedder_server::{Embedder, EmbedderServer};

//Windows Hack
use torch_sys::dummy_cuda_dependency;

pub mod service {
    tonic::include_proto!("services.embedder");
}

pub struct SBert {
    model: SBertHF,
}

impl SBert {
    pub fn new() -> Result<Self, Error> {
        unsafe {
            dummy_cuda_dependency();
        } //Windows Hack
        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distiluse-base-multilingual-cased");

        println!("Loading sbert_rs ...");
        let model = SBertHF::new(home).unwrap();

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
        let texts = Vec::from(query.into_inner().texts);

        println!("Encoding {:?}", texts.len());

        let output = &self.0.lock().await.model.encode(&texts, None).unwrap();

        //let r = Vec::<Vec<f32>>::from(output);
        let vecs = output
            .iter()
            .map(|v| service::Vector { v: v.clone() })
            .collect::<Vec<_>>();

        let reply = service::Response { vecs: vecs };

        Ok(Response::new(reply))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = match env::var("EMBEDDER_PORT") {
        Ok(val) => val.parse()?,
        Err(_e) => "[::1]:50050".parse()?,
    };

    println!("Starting SBert server on {}", addr);
    let embedder = SBert::new()?;
    let embedder = SBertSync(Mutex::new(embedder));

    Server::builder()
        .add_service(EmbedderServer::new(embedder))
        .serve(addr)
        .await?;

    Ok(())
}
