#!/bin/sh

CLUSTER_NAME=`hostname -s | rev | cut -d'-' -f 2- | rev`-
WORKERS=`echo {5..6}`
ENGINE_NAME='spacy'
ENGINE_FILE={$ENGINE_NAME}.tar
ENGINE_INSTANCE='latest'

#sudo yum install -y git
#git clone https://github.com/fastforwardlabs/airline-sentiment
#cd airline-sentiment/utils

sudo docker build --network=host -t $ENGINE_NAME:$ENGINE_INSTANCE . -f Dockerfile
sudo docker image save -o ./$ENGINE_FILE $ENGINE_NAME:$ENGINE_INSTANCE
sudo chmod 755 ./$ENGINE_FILE

for i in $WORKERS
do
 scp -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no $ENGINE_FILE $CLUSTER_NAME$i:~
 ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no $CLUSTER_NAME$i "sudo docker load --input ~/$ENGINE_FILE"
done

#Jupyter : /usr/local/bin/jupyter notebook --no-browser --port=$CDSW_APP_PORT --ip=127.0.0.1 --NotebookApp.token='' --NotebookApp.allow_remote_access=True

#Rstudio : /usr/sbin/rstudio-server start
