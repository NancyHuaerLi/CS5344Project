# ref: https://realpython.com/python-nltk-sentiment-analysis/#using-nltks-pre-trained-sentiment-analyzer
# nltk VADER classifier is good for tweets.
# VADER: Application on other langulage  is based entirely on
# using a machine-translation web service to automatically translate the text into English.


import pandas as pd
from py4j.protocol import Py4JJavaError
from pyspark import SparkConf, SparkContext
import pyspark
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import explode, col, udf, concat_ws, from_json, lit, array, expr, size, when, count, avg, sum
from pyspark.sql.functions import sum as _sum
from pyspark.sql.types import *
import json
import ast
import os
import gc
from pyspark.sql.types import BooleanType

conf = SparkConf()
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

spark = SparkSession.builder.appName("twitter_applications") \
    .config("spark.sql.files.ignoreCorruptFiles", "true").getOrCreate()

import glob

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# To get resources we need.
nltk.download(
    "names",
    "stopwords",
    "state_union",
    # "twitter_samples",
    # "movie_reviews",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
)

# siaa = SentimentIntensityAnalyzer()
# score = siaa.polarity_scores("Wow, I love you!")

# print(type(score))
# breakpoint()

@udf
def sentiment_scorer(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    return str(score)


@udf
def get_value_by_key_dict(dict_str, key=None):
    dict_str = dict_str.replace('=', ':')
    dict_value = ast.literal_eval(dict_str)
    return dict_value[key]


# You’ll get back a dictionary of different scores.
# The negative, neutral, and positive scores are related: They all add up to 1 and can’t be negative.
# The compound score is calculated differently. It’s not just an average, and it can range from -1 to 1.
# score = {'neg': 0.0, 'neu': 0.295, 'pos': 0.705, 'compound': 0.8012}


# step 1: load intermediate data
original_file_path = glob.glob('./xiangyi_sample_data/temp_result/intermediate_orig_df/*.csv')[0]
conn_file_path = glob.glob('./xiangyi_sample_data/temp_result/intermediate_conn_df/*.csv')[0]

original_df = spark.read.options(header='True').csv(original_file_path)
conn_df = spark.read.options(header='True').csv(conn_file_path)


# step 2: get sentiment score of original tweetsC
def get_sentiment_score(sdf, text_col):
    sdf = sdf.withColumn('sentiment_output', sentiment_scorer(col(text_col)))
    sdf = sdf.withColumn('sentiment_neg', get_value_by_key_dict(col('sentiment_output'), lit('neg')).cast('double'))
    sdf = sdf.withColumn('sentiment_neu', get_value_by_key_dict(col('sentiment_output'), lit('neu')).cast('double'))
    sdf = sdf.withColumn('sentiment_pos', get_value_by_key_dict(col('sentiment_output'), lit('pos')).cast('double'))
    sdf = sdf.withColumn('sentiment_compound', get_value_by_key_dict(col('sentiment_output'), lit('compound')).cast('double'))
    sdf = sdf.withColumn('sentiment_compound_ind', when(sdf.sentiment_compound >= 0, 1).otherwise(0))
    return sdf


# ###############################################
# ############ Original #################
# ###############################################
original_df = get_sentiment_score(original_df, 'text')

# computer average compound score according user's all original tweets
sen_original_df = original_df.select('user_id_str', 'sentiment_compound', 'sentiment_compound_ind')


agg_original_df = sen_original_df.groupBy("user_id_str").agg(
    count('*').alias('num_original_tweets'),
    sum('sentiment_compound_ind').alias('num_pos_tweets'),
    avg('sentiment_compound').alias('avg_sentiment_compound')
)
# agg_original_df.printSchema()
# agg_original_df.show()


# ###############################################
# ############ Connect #################
# ###############################################

conn_df = get_sentiment_score(conn_df, 'text')
conn_df.show()

conn_df = conn_df.groupBy("connected_user_single").agg(
    count('*').alias('num_original_tweets_interacting'),
    sum('sentiment_compound_ind').alias('num_pos_tweets_interacting'),
    avg('sentiment_compound').alias('avg_sentiment_compound_interacting')
)
conn_df.printSchema()
conn_df.show()

# conn_file_df.show()

# xx = original_df.select('created_at').collect()
# for x in xx:
#     print(x)

spark.stop()
