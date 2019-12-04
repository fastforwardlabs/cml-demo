# Cloudera Machine Learning Demonstration

Setup:  
Admin->Engine  
Engine Profile: 2 CPU / 8GB Memory (Add)  


Start a Python3 Session (atleast 8gb memory) and run utils/setup.py  
This will install all requirements for the code below. 


Alternatively you can create a custom docker engine using the utils/Dockerfile.  
For the CML trials contact Cloudera for a link to the engine image.  
Create a new engine as a CML admin with the following editors  


-RStudio			/usr/sbin/rstudio-server start  


-Jupyter Notebook	/usr/local/bin/jupyter-notebook --no-browser --ip=127.0.0.1 --port=${CDSW_APP_PORT} --NotebookApp.token= --NotebookApp.allow_remote_access=True --log-level=ERROR  


Due to DSE-7019, if using pyspark from Jupyter or sparklyR from RStudio you need to set the following in the Project ->
Settings -> Engine -> Environment Variables  
-PYSPARK_DRIVER_PYTHON : ipython3  
-PYSPARK_PYTHON : /usr/local/bin/python3  
-SPARK_HOME : /etc/spark/  


Demos  
0. Setup CDP Environment and CML Workspace  
1. Batch and online scoring of images with TensorFlow on spark  
2. Experiment Tracking and NLP model training for sentiment analysis of Twitter feeds  
3. Rstudio analysis of airline data from CDW or HMS/external tables and iDbroker   
