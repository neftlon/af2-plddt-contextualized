FROM pytorch/pytorch:latest

RUN mkdir /app
WORKDIR /app
ENV PATH="${PATH}:/app"

# TODO: this is copied because there is no installer available yet, otherwise the next two lines could be replaced
RUN pip3 install matplotlib pandas seaborn tqdm numpy biopython
COPY ./af22c /app/af22c/

COPY ./scripts/neff_gpu.py /app/neff_gpu.py

