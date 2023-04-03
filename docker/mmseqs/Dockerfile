FROM ubuntu:18.04 AS builder

# install dependecies and setup
RUN apt-get update
RUN apt-get install -y build-essential cmake git
RUN mkdir /app
WORKDIR /app

# build mmseqs2
RUN git clone https://github.com/neftlon/MMseqs2.git 
RUN cd MMseqs2 && mkdir build
RUN cd /app/MMseqs2/build && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=. -DNATIVE_ARCH=0 -DHAVE_AVX2=0 -DHAVE_SSE4_1=1 ..
RUN cd /app/MMseqs2/build && \
    make -j `nproc` && \
    make install

FROM ubuntu:18.04

RUN mkdir /app
WORKDIR /app

# TODO: we should not have to install `build-essential` here, but mmseqs
# depends on some dyanmically linked libraries. these are included in 
# `build-essential`. however, this should not pose a problem other than
# creating a larger container.
RUN apt update && apt install -y build-essential gdb python3

# copy mmseqs executable from builder to container
COPY --from=builder /app/MMseqs2/build/bin/mmseqs /mmseqs

# copy scripts to container
COPY ./scripts/neff_mmseqs.py /neff_mmseqs.py
