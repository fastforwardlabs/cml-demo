FROM docker.repository.cloudera.com/cdsw/engine:8
  ADD ./requirements.txt requirements.txt
  ADD ./install.R install.R
  RUN /usr/local/bin/pip3 install -r requirements.txt
  RUN /usr/local/bin/python3 -m spacy download en
  RUN /usr/local/bin/Rscript install.R

WORKDIR /tmp

RUN apt-get update && apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
    libclang-dev \
    lsb-release \
    psmisc \
    sudo

RUN wget https://download2.rstudio.org/server/trusty/amd64/rstudio-server-1.2.1335-amd64.deb && \
    dpkg -i rstudio-server-1.2.1335-amd64.deb

COPY rserver.conf /etc/rstudio/rserver.conf

COPY rstudio-cdsw /usr/local/bin/rstudio-cdsw

RUN chmod +x /usr/local/bin/rstudio-cdsw

RUN rm rstudio-server-1.2.1335-amd64.deb

