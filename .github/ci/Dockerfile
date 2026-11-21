FROM ubuntu:24.04 AS base

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git pkg-config \
        libprotobuf-dev protobuf-compiler \
        python3-dev python3-numpy pybind11-dev \
        libgtk-3-dev libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

FROM base AS dependencies
WORKDIR /src
COPY 3rd-party/ 3rd-party/
COPY cmake/ cmake/
COPY .github/ci/CMakeLists.txt .github/ci/CMakeLists.txt
RUN cmake -S .github/ci -B /build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build /build --parallel 4 \
    && cmake --install /build

FROM base AS ci
ARG SOURCE_URL
LABEL org.opencontainers.image.source="${SOURCE_URL}"
COPY --from=dependencies /usr/local/ /usr/local/
RUN ldconfig
WORKDIR /workspace
