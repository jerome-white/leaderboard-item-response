# syntax=docker/dockerfile:1
#
# Build context is the repo root (docker build -f aws/Dockerfile .), not
# aws/ itself, since the model source lives outside this directory.

FROM ghcr.io/prefix-dev/pixi:0.81.0 AS builder
WORKDIR /workspace

COPY aws/pixi.toml aws/pixi.lock aws/
RUN cd aws && pixi install --locked

COPY src/model/model.stan src/model/model.stan
RUN cd aws && pixi run bash -c '\
    cd $CMDSTAN && make --jobs=$(nproc) /workspace/src/model/model \
'

# cmdstan links the model binary dynamically against libtbb (and the
# toolchain's libc++/libstdc++), via an rpath into the pixi env - that
# env isn't present in the slim runtime image, so collect the binary's
# actual runtime dependencies (via ldd) into a self-contained bundle and
# rewrite the rpath to point at it relatively ($ORIGIN/lib), rather than
# shipping the whole multi-GB build tree.
RUN cd aws && pixi run bash -c '\
    set -eu; \
    mkdir -p /out/lib; \
    cp /workspace/src/model/model /out/model; \
    cp $CMDSTAN/bin/stansummary /out/stansummary; \
    for bin in /out/model /out/stansummary; do \
        for lib in $(ldd "$bin" | awk "\$3 ~ /^\\// {print \$3}"); do \
            cp -n "$lib" /out/lib/; \
        done; \
        patchelf --set-rpath "\$ORIGIN/lib" "$bin"; \
    done \
'

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl unzip xz-utils ca-certificates \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws \
    && apt-get purge -y unzip && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/model /app/model
COPY --from=builder /out/stansummary /app/stansummary
COPY --from=builder /out/lib/ /app/lib/
COPY aws/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
