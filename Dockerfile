ARG HTTP_PROXY
ARG RUST_VER=1.43.0
ARG CUDA_VER=10.2
ARG TARGET=x86_64-unknown-linux-gnu

# -----------------
# Cargo Build Stage
# -----------------

FROM nvidia/cuda:$CUDA_VER-cudnn7-runtime-ubuntu18.04 AS build

ARG RUST_VER
ARG TARGET
ARG CUDA_VER

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTP_PROXY
ENV TARGET=$TARGET

RUN apt-get update && apt-get install -y \
    clang \
    curl \
    unzip \
    libgomp1

ENV PATH=/root/.cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y \
    --default-toolchain $RUST_VER-x86_64-unknown-linux-gnu && \
    rustup target add $TARGET

RUN rustup target add $TARGET && \
    rustup component add rustfmt --toolchain $RUST_VER-x86_64-unknown-linux-gnu

WORKDIR /src
RUN USER=root cargo new embedder-rs

WORKDIR /src/embedder-rs
COPY Cargo.toml Cargo.lock ./

RUN apt-get install -y libssl-dev pkg-config make

ENV OPENSSL_LIB_DIR="/usr/lib/x86_64-linux-gnu" 
ENV OPENSSL_INCLUDE_DIR="/usr/include/openssl" 
ENV TORCH_CUDA_VERSION=$CUDA_VER

RUN cargo build --target $TARGET --release && \
    cargo build --target $TARGET --tests

COPY build.rs ./
COPY proto ./proto
RUN cargo run --bin force-build --features build_deps --target $TARGET && \
    cargo run --bin force-build --features build_deps --target $TARGET --release

RUN rm src/*.rs && \
    rm target/$TARGET/release/deps/sbert_rs* && \
    rm target/$TARGET/debug/deps/sbert_rs*

COPY src ./src
RUN cargo build --bin server --target $TARGET --release

RUN curl -o libtorch.zip https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.5.0.zip && \
    unzip libtorch.zip

CMD ["sh", "-c", "echo ${TARGET}"]

# -----------------
# Final Stage
# -----------------

FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04 AS final
ARG TARGET
ARG CUDA_VER

ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTP_PROXY
ENV TORCH_CUDA_VERSION=$CUDA_VER

COPY --from=build /src/embedder-rs/target/$TARGET/release/server embedder-rs

COPY --from=build /src/embedder-rs/libtorch/lib/ libtorch/
COPY --from=build /usr/lib/x86_64-linux-gnu/libgomp.so.1 libtorch/

COPY ./models/distiluse-base-multilingual-cased ./models/distiluse-base-multilingual-cased

ENV LIBTORCH=/libtorch
ENV LD_LIBRARY_PATH=${LIBTORCH}:$LD_LIBRARY_PATH

ENV EMBEDDER_PORT=0.0.0.0:50051

USER 1000
CMD ["./embedder-rs"]