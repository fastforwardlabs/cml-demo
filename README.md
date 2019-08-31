# PROJECT: Sentiment Analysis Pilot 

Setup: 
Admin->Engine 
Engine Profile: 2 CPU / 8GB Memory (Add)
docker.repository.cloudera.com/cdsw/engine:9-cml1.0-demo
RStudio			/usr/sbin/rstudio-server start
Jupyter Notebook	/usr/local/bin/jupyter-notebook --no-browser --ip=127.0.0.1 --port=${CDSW_APP_PORT} --NotebookApp.token= --NotebookApp.allow_remote_access=True --log-level=ERROR

Milestone 1: Batch scoring of images with TF on spark

Milestone 2: Sentiment Analysis of Twitter feeds

Milestone 3: Rstudio analysis of airline data from CDW or HMS/external tables and iDbroker. https://blog.cloudera.com/blog/2017/02/analyzing-us-flight-data-on-amazon-s3-with-sparklyr-and-apache-spark-2-0/

wget https://airlines-orc.s3-us-west-2.amazonaws.com/hive3-standalone-jdbc/hive-jdbc-3.1.0-SNAPSHOT-standalone.jar

jdbc:hive2://hs2-default-vw.env-tqqtbg.dwx.cloudera.site/default;transportMode=http;httpPath=cliservice;ssl=true;retries=3

