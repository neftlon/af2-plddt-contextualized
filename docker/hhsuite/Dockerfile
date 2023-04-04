FROM soedinglab/hh-suite

# HACK: https://serverfault.com/questions/1074688/security-debian-org-does-not-have-a-release-file-on-with-debian-docker-images
RUN sed -i 's/stable\/updates/stable-security\/updates/' /etc/apt/sources.list
RUN apt update && apt install -y python3

RUN mkdir /app
WORKDIR /app

# copy scripts to container
COPY ./scripts/neff_hhsuite.py /app/neff_hhsuite.py

ENV PATH="${PATH}:/app"
