# Sandbox for the polyglot exercises (javascript): jest and babel installed once at build time (network), into
# /opt/node_modules; every exercise shares the same dev dependencies. Exercises are mounted at run time, no network then.
FROM docker.io/library/node:22-slim
WORKDIR /opt
COPY package.json /opt/package.json
RUN npm install --no-audit --no-fund
