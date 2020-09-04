ARG RUST_VER=1.43.1
ARG TARGET=x86_64-unknown-linux-gnu
ARG TORCH_VER=1.6.0

FROM python:3.8-slim

ARG HTTP_PROXY
ARG RUST_VER
ARG TARGET
ARG TORCH_VER

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTP_PROXY

RUN apt-get update && apt-get install -y \
    clang \
    curl \
    unzip \
    libssl-dev \
    pkg-config \
    make

ENV PATH=/root/.cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y \
    --default-toolchain $RUST_VER-x86_64-unknown-linux-gnu && \
    rustup target add $TARGET

RUN pip install torch==${TORCH_VER}+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY Cargo.toml Cargo.lock ./
COPY src/bin ./src/bin

RUN touch src/lib.rs && cargo build --bin=convert-tensor

COPY utils/prepare_distilbert.py ./
