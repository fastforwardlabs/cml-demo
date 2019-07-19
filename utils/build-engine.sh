#!/bin/sh

CLUSTER_NAME=`hostname -s | rev | cut -d'-' -f 2- | rev`-
WORKERS=`echo {5..8}`


#sudo yum install -y git
#git clone https://github.com/fastforwardlabs/airline-sentiment
#cd airline-sentiment

sudo docker build --network=host -t spacy:1 . -f Dockerfile
sudo docker image save -o ./spacy.tar spacy:1
sudo chmod 755 ./spacy.tar

for i in $WORKERS
do
 scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no spacy.tar $CLUSTER_NAME$i:~
 ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no $CLUSTER_NAME$i "sudo docker load --input ~/spacy.tar"
done

#Jupyter : /usr/local/bin/jupyter notebook --no-browser --port=$CDSW_APP_PORT --ip=127.0.0.1 --NotebookApp.token='' --NotebookApp.allow_remote_access=True

#Rstudio : /usr/sbin/rstudio-server start
