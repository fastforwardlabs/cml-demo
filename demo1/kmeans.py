# # K-Means
#
# The K-means algorithm written from scratch against PySpark. In practice,
# one may prefer to use the KMeans algorithm in ML, as shown in
# [this example](https://github.com/apache/spark/blob/master/examples/src/main/python/ml/kmeans_example.py).
# 
# This example requires [NumPy](http://www.numpy.org/).

from __future__ import print_function
import sys
import numpy as np
from pyspark.sql import SparkSession

dataFile = 's3a://ml-field/demo/spark-mllib/data/kmeans_data.txt'

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex

spark = SparkSession\
    .builder\
    .appName("PythonKMeans")\
    .config("spark.executor.memory","1g")\
    .config("spark.executor.cores","1")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://ml-field/demo/")\
    .getOrCreate()

#    .config("spark.jars","file:/home/cdsw/jars/org.apache.hadoop_hadoop-aws-3.2.0.jar,file:/home/cdsw/jars/com.amazonaws_aws-java-sdk-bundle-1.11.375.jar")\    
#.config("spark.jars.packages","org.apache.hadoop:hadoop-aws:3.2.0")\

    
# Add the data file to hdfs.
#!hdfs dfs -put resources/data/mllib/kmeans_data.txt /tmp

#testDF=spark.read.csv(dataFile).collect()

lines = spark.read.text(dataFile).rdd.map(lambda r: r[0])
data = lines.map(parseVector).cache()
K = 2
convergeDist = 0.1

kPoints = data.takeSample(False, K, 1)
tempDist = 1.0

while tempDist > convergeDist:
    closest = data.map(
        lambda p: (closestPoint(p, kPoints), (p, 1)))
    pointStats = closest.reduceByKey(
        lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
    newPoints = pointStats.map(
        lambda st: (st[0], st[1][0] / st[1][1])).collect()

    tempDist = sum(np.sum((kPoints[iK] - p) ** 2) for (iK, p) in newPoints)

    for (iK, p) in newPoints:
        kPoints[iK] = p

print("Final centers: " + str(kPoints))

spark.stop()
