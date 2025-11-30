FROM postgres:18-bookworm

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    postgresql-server-dev-18 \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN cd /tmp && \
    git clone https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install && \
    rm -rf /tmp/pgvector

WORKDIR /usr/src/pg_gembed
COPY . .

RUN make install

USER postgres
