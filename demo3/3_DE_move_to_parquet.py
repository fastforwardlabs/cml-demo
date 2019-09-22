from pyspark.sql import SparkSession
import os


spark = SparkSession\
    .builder\
    .appName("Airline")\
    .config("spark.hadoop.fs.s3a.aws.credentials.provider","org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")\
    .config("spark.executor.memory","8g")\
    .config("spark.executor.cores","4")\
    .config("spark.driver.memory","6g")\
    .config("spark.executor.instances","3")\
    .config("spark.dynamicAllocation.enabled","false")\
    .getOrCreate()
#    .config("spark.hadoop.fs.s3a.access.key",os.getenv("AWS_ACCESS_KEY"))\
#    .config("spark.hadoop.fs.s3a.secret.key",os.getenv("AWS_SECRET_KEY"))\
#    .getOrCreate()
    #.config("spark.hadoop.fs.s3a.metadatastore.impl","org.apache.hadoop.fs.s3a.s3guard.NullMetadataStore")\
    #.config("spark.hadoop.fs.s3a.delegation.token.binding","")\

from pyspark.sql.types import *


from IPython.core.display import HTML
HTML('<a href="http://spark-{}.{}">Spark UI</a>'.format(os.getenv("CDSW_ENGINE_ID"),os.getenv("CDSW_DOMAIN")))

HTML("<script src='https://d3js.org/d3.v5.min.js'></script>")

schema = StructType([StructField("FL_DATE", TimestampType(), True),
StructField("OP_CARRIER", StringType(), True),
StructField("OP_CARRIER_FL_NUM", StringType(), True),
StructField("ORIGIN", StringType(), True),
StructField("DEST", StringType(), True),
StructField("CRS_DEP_TIME", StringType(), True),
StructField("DEP_TIME", StringType(), True),
StructField("DEP_DELAY", DoubleType(), True),
StructField("TAXI_OUT", DoubleType(), True),
StructField("WHEELS_OFF", StringType(), True),
StructField("WHEELS_ON", StringType(), True),
StructField("TAXI_IN", DoubleType(), True),
StructField("CRS_ARR_TIME", StringType(), True),
StructField("ARR_TIME", StringType(), True),
StructField("ARR_DELAY", DoubleType(), True),
StructField("CANCELLED", DoubleType(), True),
StructField("CANCELLATION_CODE", StringType(), True),
StructField("DIVERTED", DoubleType(), True),
StructField("CRS_ELAPSED_TIME", DoubleType(), True),
StructField("ACTUAL_ELAPSED_TIME", DoubleType(), True),
StructField("AIR_TIME", DoubleType(), True),
StructField("DISTANCE", DoubleType(), True),
StructField("CARRIER_DELAY", DoubleType(), True),
StructField("WEATHER_DELAY", DoubleType(), True),
StructField("NAS_DELAY", DoubleType(), True),
StructField("SECURITY_DELAY", DoubleType(), True),
StructField("LATE_AIRCRAFT_DELAY", DoubleType(), True)])


df=spark.read.csv(
  path="s3a://ml-field/demo/flight-analysis/data/airlines_csv/*",header=True,
  schema=schema)

from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

udf1 = udf(lambda x: x if len(x) == 4 else "0{}".format(x),StringType())
#df.withColumn('COLUMN_NAME_fix',udf1('COLUMN_NAME')).show()

df.select("CRS_DEP_TIME").withColumn('pad_time', udf1("CRS_DEP_TIME")).show()

smaller_data_set = df.select("FL_DATE","OP_CARRIER","OP_CARRIER_FL_NUM","ORIGIN","DEST","CRS_DEP_TIME","CRS_ARR_TIME","CANCELLED","CRS_ELAPSED_TIME","DISTANCE")

smaller_data_set.show()

#smaller_data_set.write.parquet(path="s3a://ml-field/demo/flight-analysis/data/airline_parquet_2/",compression="snappy")