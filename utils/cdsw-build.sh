pip3 install -r /home/cdsw/utils/requirements3.txt
python3 -m spacy download en

if [[ ! -d /home/cdsw/R ]]; then mkdir -m 755 /home/cdsw/R; fi
