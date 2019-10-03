### Load libraries
library(ggplot2)
library(maps)
library(geosphere)
library(DBI)
library(sparklyr)
library(dplyr)

## Connect to Spark. Check spark_defaults.conf for the correct 
spark_home_set("/etc/spark/")

config <- spark_config()
config$spark.hadoop.fs.s3a.aws.credentials.provider  <- "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider"
config$spark.executor.memory <- "16g"
config$spark.executor.cores <- "4"
config$spark.driver.memory <- "6g"
config$spark.executor.instances <- "5"
config$spark.dynamicAllocation.enabled  <- "false"
#config$spark.ui.https.enabled <- "true"
#config$spark.ssl.enabled <- "true"
config$spark.hadoop.fs.s3a.metadatastore.impl <- "org.apache.hadoop.fs.s3a.s3guard.NullMetadataStore"
config$spark.sql.catalogImplementation <- "in-memory"
config$spark.yarn.access.hadoopFileSystems <- "s3a://ml-field/demo/flight-analysis/"

spark <- spark_connect(master = "yarn-client", config=config)

library(cdsw)
html(paste("<a href='http://spark-",Sys.getenv("CDSW_ENGINE_ID"),".",Sys.getenv("CDSW_DOMAIN"),"' target='_blank'>Spark UI<a>",sep=""))

## Read in the flight data from S3

s3_link_all <-
  "s3a://ml-field/demo/flight-analysis/data/airlines_csv/*" 

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

# Load all the flight data
spark_read_csv(
  spark,
  name = "flight_data",
  path = s3_link_all,
  infer_schema = FALSE,
  columns = cols,
  header = TRUE
)

airlines <- tbl(spark, "flight_data")

#Load all the airport data

spark_read_csv(
  spark,
  name = "airports",
  path = "s3a://ml-field/demo/flight-analysis/data/airports.csv",
  infer_schema = TRUE,
  header = TRUE
)

airports  <- tbl(spark, "airports")

airports <- airports %>% collect

## This is important, you can run spark.sql functions inside R

# Add year and month fields to the flight data
airlines <-
  airlines %>% 
  mutate(year = year(FL_DATE), month = month(FL_DATE)) 

# Plot number of flights per year

airline_counts_by_year <-
  airlines %>% 
  group_by(year) %>% 
  summarise(count = n()) %>% 
  collect()

g <- ggplot(airline_counts_by_year, aes(x = year, y = count))
g <- g + geom_line(colour = "magenta",
                   linetype = 1,
                   size = 0.8)
g <- g + xlab("Year")
g <- g + ylab("Flight number")
g <- g + ggtitle("US flights")
plot(g)


# #See flight number between 2010 and 2013
#Next, let’s dig it for the 2002 data. Let’s plot flight number betwewen 2001 and 2003.

airline_counts_by_month <-
  airlines %>% filter(year >= 2010 &
                        year <= 2013) %>% group_by(year, month) %>% summarise(count = n()) %>% collect

g <- ggplot(airline_counts_by_month,
            aes(x = as.Date(
              sprintf(
                "%d-%02d-01",
                airline_counts_by_month$year,
                airline_counts_by_month$month
              )
            ), y = count))
g <- g + geom_line(colour = "magenta",
                   linetype = 1,
                   size = 0.8)
g <- g + xlab("Year/Month")
g <- g + ylab("Flight number")
g <- g + ggtitle("US flights")
plot(g)

# Next, we will summarize the data by carrier, origin and dest.

flights <-
  airlines %>% 
  group_by(year, OP_CARRIER, ORIGIN, DEST) %>% 
  summarise(count = n()) 

flights

airports <- tbl(spark, "airports") %>% collect

#Now we extract AA’s flight in 2010.

flights_aa <-
  flights %>% filter(year == 2010) %>% filter(OP_CARRIER == "AA") %>% arrange(count) %>% collect
flights_aa

#Let’s plot the flight number of AA in 2007.

# draw map with line of AA
xlim <- c(-171.738281,-56.601563)
ylim <- c(12.039321, 71.856229)

# Color settings
pal <- colorRampPalette(c("#333333", "white", "#1292db"))
colors <- pal(100)

map(
  "world",
  col = "#6B6363",
  fill = TRUE,
  bg = "#000000",
  lwd = 0.05,
  xlim = xlim,
  ylim = ylim
)

maxcnt <- max(flights_aa$count)
for (j in 1:length(flights_aa$OP_CARRIER)) {
  air1 <- airports[airports$iata == flights_aa[j, ]$ORIGIN, ]
  air2 <- airports[airports$iata == flights_aa[j, ]$DEST, ]
  
  inter <-
    gcIntermediate(
      c(air1[1, ]$long, air1[1, ]$lat),
      c(air2[1, ]$long, air2[1, ]$lat),
      n = 100,
      addStartEnd = TRUE
    )
  colindex <-
    round((flights_aa[j, ]$count / maxcnt) * length(colors))
  
  lines(inter, col = colors[colindex], lwd = 0.8)
}

