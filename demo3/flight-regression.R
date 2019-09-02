library(sparklyr)
library(dplyr)

## Configure cluster
spark_home_set("/etc/spark/")
config <- spark_config()
config$spark.hadoop.fs.s3a.aws.credentials.provider  <- "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider"

sc <- spark_connect(master = "yarn-client", config=config)

## Read in the flight data from S3

s3_link_all <-
  "s3a://ml-field/demo/flight-analysis/data/airlines_csv/2009.csv"


cols = list(
  FL_DATE = "date",
  OP_CARRIER = "character",
  OP_CARRIER_FL_NUM = "character",
  ORIGIN = "character",
  DEST = "character",
  CRS_DEP_TIME = "character",
  DEP_TIME = "character",
  DEP_DELAY = "double",
  TAXI_OUT = "double",
  WHEELS_OFF = "character",
  WHEELS_ON = "character",
  TAXI_IN = "double",
  CRS_ARR_TIME = "character",
  ARR_TIME = "character",
  ARR_DELAY = "double",
  CANCELLED = "double",
  CANCELLATION_CODE = "character",
  DIVERTED = "double",
  CRS_ELAPSED_TIME = "double",
  ACTUAL_ELAPSED_TIME = "double",
  AIR_TIME = "double",
  DISTANCE = "double",
  CARRIER_DELAY = "double",
  WEATHER_DELAY = "double",
  NAS_DELAY = "double",
  SECURITY_DELAY = "double",
  LATE_AIRCRAFT_DELAY = "double",
  'Unnamed: 27' = "logical"
)


spark_read_csv(
  sc,
  name = "flight_data",
  path = s3_link_all,
  infer_schema = FALSE,
  columns = cols,
  header = TRUE
)

airlines <- tbl(sc, "flight_data")

airlines

#We will build a predictive model with MLlib. We use linear regression of MLlib.

#First, we will prepare training data. In order to handle categorical data, you should use tf_string_indexer for converting.

# build predictive model with linear regression
airlines_filtered <- airlines %>%
  filter(ARR_DELAY >= 5) %>%
  mutate(hour = floor(dep_time/100))

airlines_filtered <-
  airlines_filtered %>% 
  mutate(month = month(FL_DATE), hour = hour(FL_DATE), dayofweek = dayofweek(FL_DATE)) 

partitions = airlines_filtered %>%
  ft_string_indexer(input_col="OP_CARRIER",output_col="carrier_index") %>%
  ft_string_indexer(input_col="ORIGIN",output_col="origin_index") %>%
  ft_string_indexer(input_col="DEST",output_col="dest_index") %>% 
  sdf_random_split(training = 0.7, test = 0.3, seed = 1099)


fit <- partitions$training %>%
   ml_linear_regression(
     response = "ARR_DELAY",
     features = c(
        "month", "hour", "dayofweek", "carrier_index", "DEP_DELAY", "origin_index", "dest_index", "DISTANCE"
       )
    )

fit

summary(fit)
