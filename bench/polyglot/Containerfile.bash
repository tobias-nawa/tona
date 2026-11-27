# Sandbox for the polyglot exercises (bash): bats-core 1.10.0 (the version the Exercism Bash track's CI installs, checksum
# pinned) on Debian's bash, plus the usual tools exercise solutions call (bc, gawk). Exercises are mounted at run time,
# no network then.
FROM docker.io/library/debian:trixie-slim@sha256:7378260c0a098d27ac206225db5e7d92b8c7c84ea7afccc86eeb513756bbd392
ARG BATS_SHA256=a1a9f7875aa4b6a9480ca384d5865f1ccf1b0b1faead6b47aa47d79709a5c5fd
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl bc gawk \
    && curl -fsSL -o /tmp/bats.tar.gz https://codeload.github.com/bats-core/bats-core/tar.gz/refs/tags/v1.10.0 \
    && echo "$BATS_SHA256  /tmp/bats.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/bats.tar.gz -C /tmp && /tmp/bats-core-1.10.0/install.sh /usr/local \
    && rm -rf /tmp/bats* /var/lib/apt/lists/* && apt-get purge -y curl && apt-get autoremove -y
