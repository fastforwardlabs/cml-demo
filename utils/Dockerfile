FROM docker.repository.cloudera.com/cdsw/engine:9
  WORKDIR /tmp
  COPY ./requirements3.txt ./install.R ./rserver.conf ./rstudio-cdsw /tmp/
  RUN /usr/local/bin/pip3 install -r /tmp/requirements3.txt && \
    /usr/local/bin/python3 -m spacy download en && \
    apt-get update && apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
      libclang-dev \
      lsb-release \
      psmisc \
      sudo && \
    wget https://download2.rstudio.org/server/trusty/amd64/rstudio-server-1.2.1335-amd64.deb && \
    dpkg -i rstudio-server-1.2.1335-amd64.deb && \
    mv /tmp/rserver.conf /etc/rstudio/rserver.conf && \
    mv /tmp/rstudio-cdsw /usr/local/bin/rstudio-cdsw  && \
    chmod +x /usr/local/bin/rstudio-cdsw && \
    /usr/local/bin/Rscript install.R
