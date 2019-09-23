from pyspark.sql import SparkSession
from pyspark.sql.functions import split, regexp_extract, regexp_replace, col
import sys
import joblib


spark = SparkSession \
    .builder \
    .appName("Pyspark Tokenize") \
    .enableHiveSupport() \
    .getOrCreate()


input_path = '/home/cdsw/access.log.2'
base_df=spark.read.text(input_path)

split_df = base_df.select(regexp_extract('value', r'([^ ]*)', 1).alias('ip'),
                          regexp_extract('value', r'(\d\d\/\w{3}\/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})', 1).alias('date_logged'),
                          regexp_extract('value', r'^(?:[^ ]*\ ){6}([^ ]*)', 1).alias('url'),
                          regexp_extract('value', r'(?<=product\/).*?(?=\s|\/)', 0).alias('productstring')
                         )


filtered_products_df = split_df.filter("productstring != ''")

cleansed_products_df=filtered_products_df.select(regexp_replace("productstring", "%20", " ").alias('product'), "ip", "date_logged", "url")

local_df = cleansed_products_df.toPandas()

joblib.dump(local_df, 'clickstream.pkl')