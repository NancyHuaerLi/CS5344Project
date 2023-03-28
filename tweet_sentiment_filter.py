from py4j.protocol import Py4JJavaError
from pyspark import SparkConf, SparkContext
import pyspark
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import explode, col, udf, concat_ws, from_json, lit, array, expr, size
from pyspark.sql.functions import sum as _sum
from pyspark.sql.types import *
import json
import os
import gc
from pyspark.sql.types import BooleanType

# Customized import
from tweet_preprocess import keyword_filter


conf = SparkConf()
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

spark = SparkSession.builder.appName("twitter_applications")\
    .config("spark.sql.files.ignoreCorruptFiles", "true").getOrCreate()

# filter based on the following keywords
# test_list = ['nintendo', 'pokemon', 'video game', 'game', 'pokémon legends: arceus', 'pokémon', 'legend of arceus',
#              'legend arceus', 'legends of arceus', 'pokemon legends: arceus', 'stream', 'twitch', 'arceus']

test_list = ['nintendo', 'pokemon', 'video game', 'game', 'pokémon legends: arceus', 'pokémon', 'legend of arceus',
             'legend arceus', 'legends of arceus', 'pokemon legends: arceus', 'twitch', 'arceus']

# regex for filter
regex_pattern = "(?i)" + "|".join(test_list)
regex_pattern_hashtag = "(?i)" + "|".join([x.strip(' ') for x in test_list])


raw_sdf = spark.read.json("./sample_data/*.json", allowBackslashEscapingAnyCharacter=True)

print('====')
print(raw_sdf.count())
raw_sdf.show()


spark.stop()
