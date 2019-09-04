# Cloudera Machine Learning Demonstration

Setup:  
Admin->Engine  
Engine Profile: 2 CPU / 8GB Memory (Add)  
>>>>>contact Cloudera for a custom docker engine link<<<<<<< . 
RStudio			/usr/sbin/rstudio-server start  
Jupyter Notebook	/usr/local/bin/jupyter-notebook --no-browser --ip=127.0.0.1 --port=${CDSW_APP_PORT} --NotebookApp.token= --NotebookApp.allow_remote_access=True --log-level=ERROR  


Due to DSE-7019, if using pyspark from Jupyter you need to set the following in the Project  
Settings -> Engine -> Environment Variables  
-Name: PYSPARK_DRIVER_PYTHON  
-Value: ipython3  
-Name: PYSPARK_PYTHON  
-Value: /usr/local/bin/python3  


Demos  
0. Setup CDP Environment and CML Workspace  
1. Batch and online scoring of images with TensorFlow on spark  
2. Experiment Tracking and NLP model training for sentiment analysis of Twitter feeds  
3. Rstudio analysis of airline data from CDW or HMS/external tables and iDbroker   

wget https://airlines-orc.s3-us-west-2.amazonaws.com/hive3-standalone-jdbc/hive-jdbc-3.1.0-SNAPSHOT-standalone.jar

jdbc:hive2://hs2-default-vw.env-tqqtbg.dwx.cloudera.site/default;transportMode=http;httpPath=cliservice;ssl=true;retries=3

