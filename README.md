# PROJECT: Sentiment Analysis Pilot 

The purpose of this project is to pilot development of customer sentiment analysis capabilities using an exemplary test-case based on publically available sentiment data.

## Details
1. Business Objective: Ready the team and process for deploying business-specific sentiment analysis capabilities using a pilot project to organize, exercise and document team workflow and skills required for future projects.   
2. LOB: **Marketing**
3. Data Source(s): airline-sentiment/data/Tweets.csv (airline sentiment csv files)
4. Dependent applications: TBD
...

## Project Phases
- Phase 1: Basic end-to-end workflow set-up.
  - PM Signoff 
- Phase 2: Set-up streaming ingest and nightly re-training.
- Phase 3: Automatic notification to LOB of sentiment change at threshold.


## Data Engineering
We will use a static dataset of Twitter feeds relating to airlines as a starting point for phase 1.

https://www.kaggle.com/crowdflower/twitter-airline-sentiment/downloads/Tweets.csv

Ingest: Initially these are in the data directory but will be moved to S3 integration.
In phase 2 we will change it to a streaming pipeline with Cloudera Data Flow

## Data Scientist
This is the a more general data analysis of the data that's present.

It starts with showing a table of all tweets, then narrows it down to tweet types by airline 
(i.e. positive, netural, negative) and plot as a stacked bar chart. This is followed by tweets by
location which is also plotted using leaflet.

Finally the complete tweet corpus is analysed to show word frequency across all words. This is also 
plotted on a bar chart and finally a wordcloud (just for Michael)

For this project and to meet the business objectives, the data that is usefull is sentiment
per airline, therefore we only need the tweet text for positive and negative tweets by airline.

## Modeller
The goal is to build a business application to understand and monitor sentiment of the airline and its competitors.  

Model is based on a RNN and uses pretrained vectors. It takes in a tweet and performs a binary classification.

The output is the classifier and the vocabulary as pickle files. 

