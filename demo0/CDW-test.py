# # Spark-SQL from PySpark
# 
# This example shows how to send SQL queries to Spark.


#NOTE: In CDP find the HMS warehouse directory and external table directory by browsing to:
# Environment -> <env name> ->  Data Lake Cluster -> Cloud Storage
# copy and paste the external location to the config setting below.

#Temporary workaround for MLX-975
#In utils/hive-site.xml edit hive.metastore.warehouse.dir and hive.metastore.warehouse.external.dir based on settings in CDP Data Lake -> Cloud Storage
!cp /home/cdsw/utils/hive-site.xml /etc/hadoop/conf/


from __future__ import print_function
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType




spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://prod-cdptrialuser11-trycdp-com/cdp-lake")\
    .getOrCreate()

spark.sql("SHOW databases").show()
spark.sql("USE retail_clickstream")
spark.sql("SHOW tables").show()
spark.sql("SELECT * FROM customers LIMIT 10").show()

    
spark.stop()

