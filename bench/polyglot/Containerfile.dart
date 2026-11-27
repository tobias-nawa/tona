# Sandbox for the polyglot exercises (dart, no Flutter): Dart SDK and the `test` package resolved once at build time
# into /opt/pub-cache (network); at run time `dart pub get --offline` uses it, no network. Exercises are mounted then.
FROM docker.io/library/dart:3.10@sha256:7eed31e0fe401422fa253a3af7e51e4c4e09c04d7cafd29627ec7a9975516db2
RUN mkdir /opt/prep && cd /opt/prep \
    && printf "name: prep\nenvironment:\n  sdk: '>=3.2.0 <4.0.0'\ndev_dependencies:\n  test: '<2.0.0'\n" > pubspec.yaml \
    && PUB_CACHE=/opt/pub-cache dart pub get && rm -rf /opt/prep
