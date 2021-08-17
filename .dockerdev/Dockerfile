ARG UBUNTU_VERSION
FROM ubuntu:$UBUNTU_VERSION

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

ARG PROJECT_NAME

RUN apt-get update -qq && apt-get upgrade -yq \
  && apt-get install -yq --no-install-recommends \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    openjdk-11-jdk \
    curl \
  && apt-get clean \
  && rm -rf /var/cache/apt/archives/* \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && truncate -s 0 /var/log/*log

ARG SPARK=spark-3.1.2-bin-hadoop2.7
RUN cd /home && \
  curl https://downloads.apache.org/spark/spark-3.1.2/$SPARK.tgz -o $SPARK.tgz && \
  tar xf $SPARK.tgz && \
  rm $SPARK.tgz && \
  mv $SPARK /usr/local/share

RUN ln -s /usr/bin/python3.8 /usr/bin/python

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -U pip && \
  cat /tmp/requirements.txt | xargs -L 1 pip3 install && \
  rm /tmp/requirements.txt

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV SPARK_HOME=/usr/local/share/spark-3.1.2-bin-hadoop2.7
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PATH=${PATH}:${JAVA_HOME}:${SPARK_HOME}/bin:${PYSPARK_PYTHON}

RUN mkdir -p /$PROJECT_NAME

WORKDIR /$PROJECT_NAME
