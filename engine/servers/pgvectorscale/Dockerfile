FROM pgvector/pgvector:0.7.2-pg16

# Update new packages
RUN apt-get update

# Get Ubuntu packages
RUN apt-get install -y \
    build-essential \
    curl \
	git \
	pkg-config \
	libssl-dev \
	libclang-dev \
	postgresql-server-dev-16 \
	libopenblas-dev


# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

RUN cargo install --locked cargo-pgrx
RUN cargo pgrx init --pg16 pg_config


RUN cd /tmp &&\
    git clone --branch 0.2.0 https://github.com/timescale/pgvectorscale &&\
    cd /tmp/pgvectorscale/pgvectorscale &&\
    cargo pgrx install --release

# make the "en_US.UTF-8" locale so postgres will be utf-8 enabled by default
RUN set -eux; \
	apt-get update; apt-get install -y --no-install-recommends locales; rm -rf /var/lib/apt/lists/*; \
	localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8
RUN localedef -i en_US -f UTF-8 en_US.UTF-8
