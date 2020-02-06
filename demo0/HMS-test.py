# # Spark-SQL from PySpark
# 
# This example shows how to send SQL queries to Spark.

# Running from CDP environment S3 bucket
#---------------------------------------
#NOTE: In CDP find the HMS warehouse directory and external table directory by browsing to:
# Environment -> <env name> ->  Data Lake Cluster -> Cloud Storage
#Data taken from http://stat-computing.org/dataexpo/2009/the-data.html
#!for i in `seq 1987 2008`; do wget http://stat-computing.org/dataexpo/2009/$i.csv.bz2; bunzip2 $i.csv.bz2; sed -i '1d' $i.csv; aws s3 cp $i.csv s3://ml-field/demo/flight-analysis/data/flights_csv/; rm $i.csv; done

#------------------------------------------------
# New in FEB 2020
# Running directly from external public S3 bucket
#------------------------------------------------
#s3://harshalpatilpublic-s3/flights_csv/ has 2007.csv and 2008.csv 
#s3://harshalpatilpublic-s3/airports_csv/
#s3://harshalpatilpublic-s3/airports_csv_extended/
#we will use this flight dataset for the demo


from __future__ import print_function
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType

# spark on kubernetes session
spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .config("spark.executor.memory", "4g")\
    .config("spark.executor.instances", 2)\
    .config("spark.yarn.access.hadoopFileSystems","s3a://harshalpatilpublic-s3/flights_csv")\
    .config("spark.driver.maxResultSize","4g")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region", "ap-southeast-1")\
    .getOrCreate()

#create database
statement = """
CREATE DATABASE IF NOT EXISTS flight_demo_public
"""    
spark.sql(statement)     

spark.sql("SHOW databases").show()

    
# create tables    
statement = '''
CREATE EXTERNAL TABLE IF NOT EXISTS `flight_demo_public`.`flight_table_public` (
`Year` int , 
`Month` int , 
`DayofMonth` int , 
`DayOfWeek` int , 
`DepTime` int , 
`CRSDepTime` int , 
`ArrTime` int , 
`CRSArrTime` int , 
`UniqueCarrier` string , 
`FlightNum` int , 
`TailNum` string , 
`ActualElapsedTime` int , 
`CRSElapsedTime` int , 
`AirTime` int , 
`ArrDelay` int , 
`DepDelay` int , 
`Origin` string , 
`Dest` string , 
`Distance` int , 
`TaxiIn` int , 
`TaxiOut` int , 
`Cancelled` int , 
`CancellationCode` string , 
`Diverted` string , 
`CarrierDelay` int , 
`WeatherDelay` int , 
`NASDelay` int , 
`SecurityDelay` int , 
`LateAircraftDelay` int )
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' 
STORED AS TextFile 
LOCATION 's3a://harshalpatilpublic-s3/flights_csv/'
'''
spark.sql(statement) 

spark.sql("USE flight_demo_public")
spark.sql("SHOW tables").show()
    
statement = '''
CREATE EXTERNAL TABLE IF NOT EXISTS `flight_demo_public`.`airport_table_public` (
`iata` string , 
`airport` string ,
`city` string ,
`state` string ,
`country` string ,
`lat` double ,
`long` double )
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' 
STORED AS TextFile 
LOCATION 's3a://harshalpatilpublic-s3/airports_csv/'
'''
spark.sql(statement) 

spark.sql("SHOW tables").show()
    
statement = '''
CREATE EXTERNAL TABLE IF NOT EXISTS `flight_demo_public`.`airport_extended_public` (
`ident` string , 
`type` string ,
`name` string ,
`elevation_ft` string ,
`continent` string ,
`iso_country` string ,
`iso_region` string ,
`municipality` string ,
`gps_code` string ,
`iata_code` string ,
`local_code` string ,
`coordinates` string )
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' 
STORED AS TextFile 
LOCATION 's3a://harshalpatilpublic-s3/airports_extended_csv/'
'''
spark.sql(statement)     
    
spark.sql("SHOW tables").show()    


#check data counts in these tables
spark.sql("SELECT count(*) FROM `flight_demo_public`.`flight_table_public`").show()


# check data in these tables
spark.sql("SELECT * FROM `flight_demo_public`.`flight_table_public` LIMIT 10").toPandas()
spark.sql("SELECT DepDelay FROM `flight_demo_public`.`flight_table_public` WHERE DepDelay > 0.0 LIMIT 10").toPandas()
spark.sql("SELECT * FROM `flight_demo_public`.`airport_table_public` LIMIT 10").toPandas()
spark.sql("SELECT * FROM `flight_demo_public`.`airport_extended_public` LIMIT 10").toPandas()

# STOP spark on Kubernetes session
spark.stop()



