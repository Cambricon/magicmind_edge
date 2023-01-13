FROM python:3.7-slim

RUN apt update && \
    apt install -y \
    curl \
    cppcheck \
    git

RUN pip install --upgrade python-gitlab sh

ADD typos-v1.12.14-x86_64-unknown-linux-musl.tar.gz /usr/bin
