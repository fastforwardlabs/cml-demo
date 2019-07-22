library(sparklyr)
library(dplyr)

## Configure cluster
config <- spark_config()
config$spark.driver.cores   <- 2
config$spark.executor.cores <- 4
config$spark.executor.memory <- "4G"

spark_home <- "/opt/cloudera/parcels/SPARK2/lib/spark2"
spark_version <- "2.0.0"
sc <- spark_connect(master="yarn-client", version=spark_version, config=config, spark_home=spark_home)

airlines <- tbl(sc, "airlines_bi_pq")
airlines

#We will build a predictive model with MLlib. We use linear regression of MLlib.

#First, we will prepare training data. In order to handle categorical data, you should use tf_string_indexer for converting.

# build predictive model with linear regression
partitions <- airlines %>%
  filter(arrdelay >= 5) %>%
  sdf_mutate(
       carrier_cat = ft_string_indexer(carrier),
       origin_cat = ft_string_indexer(origin),
       dest_cat = ft_string_indexer(dest)
  ) %>%
  mutate(hour = floor(dep_time/100)) %>%
  sdf_partition(training = 0.5, test = 0.5, seed = 1099)
fit <- partitions$training %>%
   ml_linear_regression(
     response = "arrdelay",
     features = c(
        "month", "hour", "dayofweek", "carrier_cat", "depdelay", "origin_cat", "dest_cat", "distance"
       )
    )

fit

summary(fit)
