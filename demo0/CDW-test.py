# # Spark-SQL from PySpark
# 
# This example shows how to send SQL queries to Spark.

from __future__ import print_function
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType


#NOTE: In CDP find the HMS external table location by browsing to:
# Environment -> <env name> ->  Data Lake Cluster -> Cloud Storage
# copy and paste the external location to the config setting below.


spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://<******BUCKET*****>/<*****DATALAKE*****/*****OBJECT*****")\
    .getOrCreate()

spark.sql("SHOW databases").show()
spark.sql("USE retail_clickstream")
spark.sql("SHOW tables").show()
spark.sql("SELECT * FROM customers LIMIT 10").show()

    
spark.stop()

