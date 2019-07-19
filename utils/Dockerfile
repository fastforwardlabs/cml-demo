FROM docker.repository.cloudera.com/cdsw/engine:8
  ADD ./requirements.txt requirements.txt
  ADD ./install.R install.R
  RUN /usr/local/bin/pip3 install -r requirements.txt
  RUN /usr/local/bin/python3 -m spacy download en
  RUN /usr/local/bin/Rscript install.R
