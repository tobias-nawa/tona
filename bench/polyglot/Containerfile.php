# Sandbox for the polyglot exercises (php): PHP 8.4 CLI and the PHPUnit phar the Exercism PHP track documents (11.5,
# version and checksum pinned). Exercises are mounted at run time, no network then.
FROM docker.io/library/php:8.4-cli@sha256:37d80ce5e2b44a1a33e4d4aa4cf8e0aaf90532143803e70c3223bd2c1d2b062a
ARG PHPUNIT_SHA256=9d488cd5e62b8eea20dd50ea1e2b65ff6e3837183922f152c7264407f82c368e
RUN curl -fsSL -o /usr/local/bin/phpunit https://phar.phpunit.de/phpunit-11.5.58.phar \
    && echo "$PHPUNIT_SHA256  /usr/local/bin/phpunit" | sha256sum -c - && chmod +x /usr/local/bin/phpunit
