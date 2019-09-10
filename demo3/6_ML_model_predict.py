from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel


## Note this a local Spark instance running in the engine
spark = SparkSession.builder \
      .appName("Flight Predictor") \
      .master("local[*]") \
      .config("spark.driver.memory","4g")\
      .config("spark.hadoop.fs.s3a.aws.credentials.provider","org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")\
      .config("spark.hadoop.fs.s3a.metadatastore.impl","org.apache.hadoop.fs.s3a.s3guard.NullMetadataStore")\
      .config("spark.hadoop.fs.s3a.delegation.token.binding","")\
      .getOrCreate()
  
model = PipelineModel.load("s3a://ml-field/demo/flight-analysis/data/models/lr_model") 

from pyspark.sql.types import *

feature_schema = StructType([StructField("OP_CARRIER", StringType(), True),
StructField("ORIGIN", StringType(), True),
StructField("DEST", StringType(), True),
StructField("CRS_DEP_TIME", StringType(), True),
StructField("CRS_ELAPSED_TIME", DoubleType(), True),
StructField("DISTANCE", DoubleType(), True)])

from pyspark.sql.types import StringType
from pyspark.sql.functions import udf,substring

convert_time_to_hour = udf(lambda x: x if len(x) == 4 else "0{}".format(x),StringType())

#args = {"feature":"AA,ICT,DFW,1135,85,328"}

def predict(args):
  flight_features = args["feature"].split(",")
  features = spark.createDataFrame([
  (
    flight_features[0],
    flight_features[1],
    flight_features[2],
    flight_features[3],
    float(flight_features[4]),
    float(flight_features[5]))],schema=feature_schema)
  features = features.withColumn('CRS_DEP_HOUR', substring(convert_time_to_hour("CRS_DEP_TIME"),0,2))
  result = model.transform(features).collect()[0].prediction
  return {"result" : result}



