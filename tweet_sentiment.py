# ref: https://realpython.com/python-nltk-sentiment-analysis/#using-nltks-pre-trained-sentiment-analyzer
# nltk VADER classifier is good for tweets.
# VADER: Application on other langulage  is based entirely on
# using a machine-translation web service to automatically translate the text into English.


import pandas as pd
from py4j.protocol import Py4JJavaError
from pyspark import SparkConf, SparkContext
import pyspark
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import explode, col, udf, concat_ws, from_json, lit, array, expr, size, when, count, avg, sum


import ast
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

conf = SparkConf()
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

spark = SparkSession.builder.appName("twitter_applications") \
    .config("spark.sql.files.ignoreCorruptFiles", "true").getOrCreate()


# siaa = SentimentIntensityAnalyzer()
# score = siaa.polarity_scores("Wow, I love you!")
# print(score)
# print(type(score))


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
# original_file_path = glob.glob('./xiangyi_sample_data/temp_result/intermediate_orig_df/*.csv')[0]
# conn_file_path = glob.glob('./xiangyi_sample_data/temp_result/intermediate_conn_df/*.csv')[0]
# original_df = spark.read.options(header='True').csv(original_file_path)
# conn_df = spark.read.options(header='True').csv(conn_file_path)

original_df = spark.read.json('./xiangyi_aws_input/original_tweet_intermediate_full/*.json')
conn_df = spark.read.json('./xiangyi_aws_input/retweeted_tweet_intermediate_full/*.json')

print('Before filter by candidates')
# original_df.printSchema()
# conn_df.printSchema()
print(original_df.count())
print(conn_df.count())

cand_schema = StructType([
    StructField('user_id_str', StringType()),
    StructField('pagerank_score', FloatType()),
    StructField('user_name', StringType()),
])

# load candidate list
cand_sdf = spark.read.csv('./test_pokemon_damping/*.csv', schema=cand_schema).select('user_id_str').distinct()
cand_sdf.printSchema()

# cand_ls = list(cand_sdf.toPandas()['user_id_str'])
# print(type(cand_ls[0]))
# print(cand_ls)
# cand_original_sdf = original_df.filter(original_df.user_id_str.isin(cand_ls))
# cand_conn_sdf = conn_df.filter(conn_df.connected_user_single.isin(cand_ls))

cand_original_sdf = original_df.join(cand_sdf, 'user_id_str', 'inner')
cand_conn_sdf = conn_df.join(cand_sdf, [conn_df.connected_user_single == cand_sdf.user_id_str], 'inner')


# cand_original_sdf_tmp = cand_original_sdf.select('user_id_str').distinct()
# print(cand_original_sdf_tmp.count())
# cand_conn_sdf_tmp = cand_conn_sdf.select('connected_user_single').distinct()
# print(cand_conn_sdf_tmp.count())


print('After filter by candidates')
print(cand_original_sdf.count())
print(cand_conn_sdf.count())


# step 2: get sentiment score of original tweets
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
cand_original_sdf = get_sentiment_score(cand_original_sdf, 'text')

# computer average compound score according user's all original tweets
sen_original_df = cand_original_sdf.select('user_id_str', 'sentiment_compound', 'sentiment_compound_ind')


sen_original_df = sen_original_df.groupBy("user_id_str").agg(
    count('*').alias('num_original_tweets'),
    sum('sentiment_compound_ind').alias('num_pos_original_tweets'),
    avg('sentiment_compound').alias('avg_sentiment_compound')
)
sen_original_df = sen_original_df.withColumn('rate_pos_original_tweets',
                                             col('num_pos_original_tweets')/col('num_original_tweets'))

sen_original_df.printSchema()
# sen_original_df.show()


# ###############################################
# ############ Connect #################
# ###############################################

cand_conn_sdf = get_sentiment_score(cand_conn_sdf, 'text')
sen_conn_df = cand_conn_sdf.select('connected_user_single', 'sentiment_compound', 'sentiment_compound_ind')

sen_conn_df = sen_conn_df.groupBy("connected_user_single").agg(
    count('*').alias('num_original_tweets_interacting'),
    sum('sentiment_compound_ind').alias('num_pos_tweets_interacting'),
    avg('sentiment_compound').alias('avg_sentiment_compound_interacting')
)
sen_conn_df = sen_conn_df.withColumn('rate_pos_tweets_interacting',
                                     col('num_pos_tweets_interacting')/col('num_original_tweets_interacting'))
sen_conn_df.printSchema()
# sen_conn_df.show()

res_df = sen_original_df.join(sen_conn_df, [sen_original_df.user_id_str == sen_conn_df.connected_user_single], 'outer')
# res_df.show(50)
res_df.write.mode('overwrite').json('./sentiment_output/tweet_sentiment')

spark.stop()
