## Load libraries
library(ggplot2)
library(maps)
library(geosphere)
library (DBI)

## Loading required package: sparklyr + dplyr

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

dbSendQuery(sc,"CREATE DATABASE IF NOT EXISTS flights")
tbl_change_db(sc,"flights")

dbSendQuery(sc,"CREATE EXTERNAL TABLE IF NOT EXISTS airports_str (   iata STRING,    airport STRING,    city STRING,    state STRING,    country STRING,    latitude DOUBLE,    longitude DOUBLE) ROW FORMAT  SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' STORED AS TEXTFILE LOCATION '/tmp/airports/' TBLPROPERTIES('skip.header.line.count'='1')")
dbSendQuery(sc,"create table if not exists airports as (select iata, city, state, country, cast(latitude as double), cast (longitude as double) from airports_str)")
dbSendQuery(sc,"CREATE EXTERNAL TABLE IF NOT EXISTS airlines_bi_pq ( year INT, month INT, day INT, dayofweek INT, dep_time INT, crs_dep_time INT, arr_time INT, crs_arr_time INT, carrier STRING, flight_num INT, tail_num INT, actual_elapsed_time INT, crs_elapsed_time INT, airtime INT, arrdelay INT, depdelay INT, origin STRING, dest STRING, distance INT, taxi_in INT, taxi_out INT, cancelled INT, cancellation_code STRING, diverted INT, carrier_delay INT, weather_delay INT, nas_delay INT, security_delay INT, late_aircraft_delay INT, date_yyyymm STRING) STORED AS PARQUET LOCATION '/tmp/airlines'")

airlines <- tbl(sc, "airlines_bi_pq")
airlines

airline_counts_by_year <- airlines %>% group_by(year) %>% summarise(count=n()) %>% collect
airline_counts_by_year %>% tbl_df %>% print(n=nrow(.))

g <- ggplot(airline_counts_by_year, aes(x=year, y=count))
g <- g + geom_line(
  colour = "magenta",
  linetype = 1,
  size = 0.8
)
g <- g + xlab("Year")
g <- g + ylab("Flight number")
g <- g + ggtitle("US flights")
plot(g)


# #See flight number between 2001 and 2003
#Next, let’s dig it for the 2002 data. Let’s plot flight number betwewen 2001 and 2003.

airline_counts_by_month <- airlines %>% filter(year>= 2001 & year<=2003) %>% group_by(year, month) %>% summarise(count=n()) %>% collect

g <- ggplot(
  airline_counts_by_month, 
  aes(x=as.Date(sprintf("%d-%02d-01", airline_counts_by_month$year, airline_counts_by_month$month)), y=count)
  )
g <- g + geom_line(
  colour = "magenta",
  linetype = 1,
  size = 0.8
)
g <- g + xlab("Year/Month")
g <- g + ylab("Flight number")
g <- g + ggtitle("US flights")
plot(g)

# Next, we will summarize the data by carrier, origin and dest.

flights <- airlines %>% group_by(year, carrier, origin, dest) %>% summarise(count=n()) %>% collect
flights

airports <- tbl(sc, "airports") %>% collect

#Now we extract AA’s flight in 2007.

flights_aa <- flights %>% filter(year==2007) %>% filter(carrier=="AA") %>% arrange(count)
flights_aa

#Let’s plot the flight number of AA in 2007.

# draw map with line of AA
xlim <- c(-171.738281, -56.601563)
ylim <- c(12.039321, 71.856229)

# Color settings
pal <- colorRampPalette(c("#333333", "white", "#1292db"))
colors <- pal(100)

map("world", col="#6B6363", fill=TRUE, bg="#000000", lwd=0.05, xlim=xlim, ylim=ylim)

maxcnt <- max(flights_aa$count)
for (j in 1:length(flights_aa$carrier)) {
  air1 <- airports[airports$iata == flights_aa[j,]$origin,]
  air2 <- airports[airports$iata == flights_aa[j,]$dest,]
  
  inter <- gcIntermediate(c(air1[1,]$longitude, air1[1,]$latitude), c(air2[1,]$longitude, air2[1,]$latitude), n=100, addStartEnd=TRUE)
  colindex <- round( (flights_aa[j,]$count / maxcnt) * length(colors) )
  
  lines(inter, col=colors[colindex], lwd=0.8)
}
