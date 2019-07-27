library (DBI)
library (sparklyr)

## Configure cluster
config <- spark_config()
config$spark.driver.cores   <- 2
config$spark.executor.cores <- 4
config$spark.executor.memory <- "4G"

spark_home <- "/opt/cloudera/parcels/SPARK2/lib/spark2"
spark_version <- "2.0.0"
sc <- spark_connect(master="yarn-client", version=spark_version, config=config, spark_home=spark_home)

tbl_change_db(sc,"flight")
dbSendQuery(sc, "DROP TABLE airports_str")
dbSendQuery(sc, "DROP TABLE airports")
dbSendQuery(sc, "DROP TABLE airlines_bi_pq")
