FROM datamechanics/spark:3.1-latest

ARG PROJECT_NAME

USER root

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -U pip && \
    cat /tmp/requirements.txt | xargs -L 1 pip3 install && \
    rm /tmp/requirements.txt

RUN mkdir -p /$PROJECT_NAME

WORKDIR /$PROJECT_NAME
