# Sandbox for the polyglot exercises (java): JDK 21 as in Aider's benchmark image, Gradle 8.7 (the version every
# exercise's wrapper asks for) and the JUnit/AssertJ/rxjava/json dependencies resolved once at build time (network);
# the tests run with --network=none and `gradle --offline`. Exercises are mounted at run time.
FROM docker.io/library/eclipse-temurin:21-jdk@sha256:c775c38419b7decaeb04d1bad0d43724eb9b9da04a81d68d25a0327b45e4598b
ARG GRADLE_SHA256=544c35d6bd849ae8a5ed0bcea39ba677dc40f49df7d1835561582da2009b961d
RUN curl -fsSL -o /tmp/gradle.zip https://services.gradle.org/distributions/gradle-8.7-bin.zip \
    && echo "$GRADLE_SHA256  /tmp/gradle.zip" | sha256sum -c - \
    && cd /opt && jar xf /tmp/gradle.zip && mv gradle-8.7 gradle \
    && chmod +x gradle/bin/gradle && rm /tmp/gradle.zip
ENV PATH=/opt/gradle/bin:$PATH
COPY java-prep/build.gradle /opt/prep/build.gradle
# Warm the dependency cache into /opt/gradle-home; at run time it is the read-only cache (GRADLE_RO_DEP_CACHE).
RUN mkdir -p /opt/prep/src/test/java && printf 'class T {}\n' > /opt/prep/src/test/java/T.java \
    && cd /opt/prep && GRADLE_USER_HOME=/opt/gradle-home gradle --no-daemon -q test compileJava \
    && rm -rf /opt/prep/.gradle /opt/prep/build
