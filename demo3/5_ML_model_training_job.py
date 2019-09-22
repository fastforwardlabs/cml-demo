from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("Airline ML")\
    .config("spark.hadoop.fs.s3a.aws.credentials.provider","org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")\
    .config("spark.executor.memory","8g")\
    .config("spark.executor.cores","4")\
    .config("spark.driver.memory","6g")\
    .config("spark.executor.instances","3")\
    .config("spark.hadoop.fs.s3a.metadatastore.impl","org.apache.hadoop.fs.s3a.s3guard.NullMetadataStore")\
    .config("spark.hadoop.fs.s3a.delegation.token.binding","")\
    .getOrCreate()
#    .config("spark.hadoop.fs.s3a.access.key",os.getenv("AWS_ACCESS_KEY"))\
#    .config("spark.hadoop.fs.s3a.secret.key",os.getenv("AWS_SECRET_KEY"))\

    
flight_df=spark.read.parquet(
  "s3a://ml-field/demo/flight-analysis/data/airline_parquet_2/",
)

flight_df = flight_df.na.drop() #.limit(1000000) #this limit is here for the demo

from pyspark.sql.types import StringType
from pyspark.sql.functions import udf,substring

convert_time_to_hour = udf(lambda x: x if len(x) == 4 else "0{}".format(x),StringType())
#df.withColumn('COLUMN_NAME_fix',udf1('COLUMN_NAME')).show()

flight_df = flight_df.withColumn('CRS_DEP_HOUR', substring(convert_time_to_hour("CRS_DEP_TIME"),0,2))
#flight_df = flight_df.withColumn('CRS_ARR_HOUR', substring(convert_time_to_hour("CRS_ARR_TIME"),0,2))

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

numeric_cols = ["CRS_ELAPSED_TIME","DISTANCE"]

op_carrier_indexer = StringIndexer(inputCol ='OP_CARRIER', outputCol = 'OP_CARRIER_INDEXED',handleInvalid="keep")
op_carrier_encoder = OneHotEncoder(inputCol ='OP_CARRIER_INDEXED', outputCol='OP_CARRIER_ENCODED')

origin_indexer = StringIndexer(inputCol ='ORIGIN', outputCol = 'ORIGIN_INDEXED',handleInvalid="keep")
origin_encoder = OneHotEncoder(inputCol ='ORIGIN_INDEXED', outputCol='ORIGIN_ENCODED')

dest_indexer = StringIndexer(inputCol ='DEST', outputCol = 'DEST_INDEXED',handleInvalid="keep")
dest_encoder = OneHotEncoder(inputCol ='DEST_INDEXED', outputCol='DEST_ENCODED')

crs_dep_hour_indexer = StringIndexer(inputCol ='CRS_DEP_HOUR', outputCol = 'CRS_DEP_HOUR_INDEXED',handleInvalid="keep")
crs_dep_hour_encoder = OneHotEncoder(inputCol ='CRS_DEP_HOUR_INDEXED', outputCol='CRS_DEP_HOUR_ENCODED')

input_cols=[
    'OP_CARRIER_ENCODED',
    'ORIGIN_ENCODED',
    'DEST_ENCODED',
    'CRS_DEP_HOUR_ENCODED'] + numeric_cols


assembler = VectorAssembler(
    inputCols = input_cols,
    outputCol = 'features')

from pyspark.ml import Pipeline


from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'CANCELLED', maxIter=100)

pipeline = Pipeline(stages=[op_carrier_indexer, 
                            op_carrier_encoder, 
                            origin_indexer,
                            origin_encoder,
                            dest_indexer,
                            dest_encoder,
                            crs_dep_hour_indexer,
                            crs_dep_hour_encoder,
                            assembler,
                            lr])

(train, test) = flight_df.randomSplit([0.7, 0.3])

lrModel = pipeline.fit(train)

from pyspark.ml.evaluation import BinaryClassificationEvaluator

predictionslr = lrModel.transform(test)
evaluator = BinaryClassificationEvaluator(labelCol="CANCELLED",metricName="areaUnderROC")
evaluator.evaluate(predictionslr)

#lrModel.write().overwrite().save("s3a://ml-field/demo/flight-analysis/data/models/lr_model")