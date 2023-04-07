import pandas as pd
from py4j.protocol import Py4JJavaError
from pyspark import SparkConf, SparkContext
import pyspark
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import explode, col, udf, concat_ws, from_json, lit, array, expr, size, when, count, avg, sum



conf = SparkConf()
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

spark = SparkSession.builder.appName("twitter_applications") \
    .config("spark.sql.files.ignoreCorruptFiles", "true").getOrCreate()

cand_schema = StructType([
    StructField('user_id_str', StringType()),
    StructField('pagerank_score', FloatType()),
    StructField('user_name', StringType()),
])

# load candidate list
cand_sdf = spark.read.csv('./test_pokemon_damping/*.csv', schema=cand_schema).filter(col('user_name') != 'Gamersnanet_3')
cand_sdf.printSchema()
print(cand_sdf.count())
# cand_sdf.show()

# load sentiment

sen_sdf = spark.read.json('./sentiment_output/tweet_sentiment/*.json')
sen_sdf = sen_sdf.drop(col('user_id_str'))
sen_sdf.printSchema()
# sen_sdf.show()
# print(sen_sdf.count())


combined_sdf = cand_sdf.join(sen_sdf, [cand_sdf.user_id_str == sen_sdf.connected_user_single], 'inner')
combined_sdf = combined_sdf.na.fill(value=1, subset=['num_original_tweets', 'num_pos_original_tweets'])
combined_sdf = combined_sdf.na.fill(value=1.0, subset=['rate_pos_original_tweets'])
combined_sdf = combined_sdf.na.fill(value=0.0, subset=['avg_sentiment_compound'])
print(combined_sdf.count())
combined_sdf.show()
combined_sdf.write.mode('overwrite').json('./sentiment_output/tweet_sentiment_final_output')
