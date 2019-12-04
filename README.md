# Cloudera Machine Learning Demonstration

Setup:  
Admin->Engine  
Engine Profile: 2 CPU / 8GB Memory (Add)  


Start a Python3 Session (at least 8gb memory) and run utils/setup.py  
This will install all requirements for the code below. 


Alternatively you can create a custom docker engine using the utils/Dockerfile.  
For the CML trials contact Cloudera for a link to the engine image.  
Create a new engine as a CML admin with the following editors  

-RStudio			/usr/sbin/rstudio-server start  

-Jupyter Notebook	/usr/local/bin/jupyter-notebook --no-browser --ip=127.0.0.1 --port=${CDSW_APP_PORT} --NotebookApp.token= --NotebookApp.allow_remote_access=True --log-level=ERROR  


Demos  
0. Setup CDP Environment and CML Workspace  
1. Batch and online scoring of images with TensorFlow on spark  
2. Experiment Tracking and NLP model training for sentiment analysis of Twitter feeds  
3. Rstudio analysis of airline data from CDW or HMS/external tables and iDbroker   
