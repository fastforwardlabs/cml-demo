from pyspark.sql import SparkSession
import os

spark = SparkSession\
    .builder\
    .appName("Airline")\
    .config("spark.hadoop.fs.s3a.aws.credentials.provider","org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")\
    .getOrCreate()
        
from pyspark.sql.types import *

tinydf = spark.read.parquet("s3a://ml-field/demo/flight-analysis/data/airline_parquet_2/")

from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols = [
        'number_customer_service_calls', \
        'total_night_minutes', \
        'total_day_minutes', \
        'total_eve_minutes', \
        'account_length'],
    outputCol = 'features')

# Transform labels
from pyspark.ml.feature import StringIndexer

label_indexer = StringIndexer(inputCol = 'churned', outputCol = 'label')
# Fit the model
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier

classifier = RandomForestClassifier(labelCol = 'label', featuresCol = 'features')

pipeline = Pipeline(stages=[assembler, label_indexer, classifier])

(train, test) = df.randomSplit([0.7, 0.3])
model = pipeline.fit(train)


from pyspark.ml.evaluation import BinaryClassificationEvaluator

predictions = model.transform(train)
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)