#!/bin/sh

ENGINE_NAME='cml-demo'
ENGINE_FILE=$ENGINE_NAME.tar
ENGINE_INSTANCE='latest'

sudo docker build --network=host -t $ENGINE_NAME:$ENGINE_INSTANCE . -f Dockerfile
sudo docker image save -o ./$ENGINE_FILE $ENGINE_NAME:$ENGINE_INSTANCE
sudo chmod 755 ./$ENGINE_FILE
