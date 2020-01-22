# # Spark-SQL from PySpark
# 
# This example shows how to send SQL queries to Spark.


#NOTE: In CDP find the HMS warehouse directory and external table directory by browsing to:
# Environment -> <env name> ->  Data Lake Cluster -> Cloud Storage

#Data taken from http://stat-computing.org/dataexpo/2009/the-data.html
#!for i in `seq 1987 2008`; do wget http://stat-computing.org/dataexpo/2009/$i.csv.bz2; bunzip2 $i.csv.bz2; sed -i '1d' $i.csv; aws s3 cp $i.csv s3://ml-field/demo/flight-analysis/data/flights_csv/; rm $i.csv; done


from __future__ import print_function
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import Row, StructField, StructType, StringType, IntegerType

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .config("spark.executor.memory", "4g")\
    .config("spark.executor.instances", 2)\
    .config("spark.yarn.access.hadoopFileSystems","s3a://ml-field/demo/flight-analysis/data/")\
    .config("spark.driver.maxResultSize","4g")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region", "us-west-2")\
    .getOrCreate()

spark.sql("SHOW databases").show()
spark.sql("USE default")
spark.sql("SHOW tables").show()

#spark.sql("SELECT COUNT(*) FROM `default`.`flights`").show()
spark.sql("SELECT * FROM `default`.`flights` LIMIT 10").take(5)
spark.sql("SELECT DepDelay FROM `default`.`flights` WHERE DepDelay > 0.0").take(5)

#spark.sql("SELECT COUNT(*) FROM `default`.`airports`").show()
spark.sql("SELECT * FROM `default`.`airports` LIMIT 10").show()









#spark.sql("DROP TABLE flights").show()
statement = '''
CREATE EXTERNAL TABLE IF NOT EXISTS `default`.`flights` (
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
LOCATION 's3a://ml-field/demo/flight-analysis/data/flights_csv/'
'''
#spark.sql(statement) 

#spark.sql("DROP TABLE airports").show()
statement = '''
CREATE EXTERNAL TABLE IF NOT EXISTS `default`.`airports` (
`iata` string , 
`airport` string ,
`city` string ,
`state` string ,
`country` string ,
`lat` double ,
`long` double )
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' 
STORED AS TextFile 
LOCATION 's3a://ml-field/demo/flight-analysis/data/airports_csv/'
'''
#spark.sql(statement) 


#spark.sql("DROP TABLE airports_extended").show()
statement = '''
CREATE EXTERNAL TABLE IF NOT EXISTS `default`.`airports_extended` (
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
LOCATION 's3a://ml-field/demo/flight-analysis/data/airports-extended/'
'''
#spark.sql(statement) 

    
#spark.stop()
