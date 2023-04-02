import pandas as pd
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

conf = SparkConf()
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

spark = SparkSession.builder.appName("twitter_applications")\
    .config("spark.sql.files.ignoreCorruptFiles", "true").getOrCreate()

# filter based on the following keywords
# test_list = ['russia', 'ukraine', 'putin', 'zelensky'
#                                            'russian', 'ukrainian', 'keiv', 'kyiv']
# test_list = ['nintendo', 'pokemon', 'video game', 'game', 'pokémon legends: arceus', 'pokémon', 'legend of arceus',
#              'legend arceus', 'legends of arceus', 'pokemon legends: arceus', 'stream', 'twitch', 'arceus']

test_list = ['nintendo', 'pokemon', 'video game', 'game', 'pokémon legends: arceus', 'pokémon', 'legend of arceus',
             'legend arceus', 'legends of arceus', 'pokemon legends: arceus', 'twitch', 'arceus']


# regex for filter
regex_pattern = "(?i)" + "|".join(test_list)
regex_pattern_hashtag = "(?i)" + "|".join([x.strip(' ') for x in test_list])


# get candidate list, read csv into pdf

# cand_score_df = spark.read.csv('./sample_data/pagerank_top_10.csv')
# cand_df = cand_score_df.toDF('cand_id_str', 'rank_score').select('cand_id_str')
cand_df = pd.read_csv('./sample_data/pagerank_top_10.csv')
cand_ls = list(cand_df.iloc[:, 0])
cand_ls = [str(x) for x in cand_ls]


# mock-up top user
# cand_ls = ['1407673343973675010', '1389342743948894209']


# raw = spark.read.json("./sample_data/*.json", allowBackslashEscapingAnyCharacter=True)
# raw = spark.read.json("./202202*/*.json", allowBackslashEscapingAnyCharacter=True)
def pre_process(folder, candidate_ls):
    raw = spark.read.json(folder+"/*.json.gz", allowBackslashEscapingAnyCharacter=True)
    # twitter json counts
    # print(raw.count())
    '''
        Step 1:
        Save all tweets, include original tweets in the dataset and retweeted / quoted tweets

        only select necessary columns
        can refer to twitter API for better understanding:
        https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/tweet
    '''

    twitter_df = raw.select("created_at", "id_str", col("user.id_str").alias("user_id_str"),
                            col("user.screen_name").alias("user_twitter_handle"),
                            "in_reply_to_status_id_str", "in_reply_to_user_id_str", "in_reply_to_screen_name",
                            "retweeted_status",
                            "quoted_status",
                            "text", col("entities.hashtags.text").alias("hashtags"),
                            col("quoted_status.created_at").alias("quoted_time"),
                            col("quoted_status.id_str").alias("quoted_original_tweet_id_str"),
                            col("quoted_status.user.id_str").alias("quoted_original_user_id_str"),
                            col("quoted_status.user.screen_name").alias("quoted_original_user_twitter_handle"),
                            col("retweeted_status.created_at").alias("retweeted_time"),
                            col("retweeted_status.id_str").alias("retweeted_original_tweet_id_str"),
                            col("retweeted_status.user.screen_name").alias("retweeted_original_user_twitter_handle"),
                            col("retweeted_status.user.id_str").alias("retweeted_original_user_id_str")
                            )

    # filtered by candidates
    # twitter_df = twitter_df.filter((twitter_df.user_id_str in candidate_ls) |
    #                             (twitter_df.quoted_original_user_id_str in candidate_ls) |
    #                             (twitter_df.retweeted_original_user_id_str in candidate_ls) |
    #                             (twitter_df.in_reply_to_user_id_str in candidate_ls))

    # add all retweeted / quoted original tweet themselves to the twitter df
    retweet_rdd = twitter_df.filter(twitter_df.retweeted_status.isNotNull()).select('retweeted_status.*')
    quoted_rdd = twitter_df.filter(twitter_df.quoted_status.isNotNull()).select('quoted_status.*')


    # select columns
    retweet_rdd = retweet_rdd.select("created_at",
                                     "id_str",
                                     col("user.id_str").alias("user_id_str"),
                                     col("user.screen_name").alias("user_twitter_handle"),
                                     "in_reply_to_status_id_str",
                                     "in_reply_to_user_id_str",
                                     "in_reply_to_screen_name",
                                     # "quoted_status",
                                     "text",
                                     col("entities.hashtags.text").alias("hashtags"),
                                     col("quoted_status.created_at").alias("quoted_time"),
                                     col("quoted_status.id_str").alias("quoted_original_tweet_id_str"),
                                     col("quoted_status.user.id_str").alias("quoted_original_user_id_str"),
                                     col("quoted_status.user.screen_name").alias("quoted_original_user_twitter_handle")
                                     )

    quoted_rdd = quoted_rdd.select("created_at",
                                   "id_str",
                                   col("user.id_str").alias("user_id_str"),
                                   "in_reply_to_status_id_str",
                                   "in_reply_to_user_id_str",
                                   "in_reply_to_screen_name",
                                   "text",
                                   col("entities.hashtags.text").alias("hashtags")
                                   )
    twitter_df = twitter_df.drop("retweeted_status", "quoted_status")

    # add missing columns to make sure all columns match in the above 3 DFs
    for c in retweet_rdd.columns:
        if c not in twitter_df.columns:
            twitter_df = twitter_df.withColumn(c, lit(None))
        if c not in quoted_rdd.columns:
            quoted_rdd = quoted_rdd.withColumn(c, lit(None))

    for c in quoted_rdd.columns:
        if c not in twitter_df.columns:
            twitter_df = twitter_df.withColumn(c, lit(None))
        if c not in retweet_rdd.columns:
            retweet_rdd = retweet_rdd.withColumn(c, lit(None))

    for c in twitter_df.columns:
        if c not in quoted_rdd.columns:
            quoted_rdd = quoted_rdd.withColumn(c, lit(None))
        if c not in retweet_rdd.columns:
            retweet_rdd = retweet_rdd.withColumn(c, lit(None))

    # final twitter DFs
    combined_raw = twitter_df.unionByName(retweet_rdd).unionByName(quoted_rdd)
    del retweet_rdd
    del quoted_rdd
    del twitter_df
    gc.collect()

    # convert hashtag column (array type) to str for regex expression filter
    raw_df = combined_raw.withColumn("text_hashtag", concat_ws(",", col("hashtags")))

    filter_df = raw_df.filter(raw_df.text.rlike(regex_pattern) | raw_df.text_hashtag.rlike(regex_pattern)).distinct()
    del raw_df
    del combined_raw
    gc.collect()
    print("date " + folder[2:] + " # of tweets analyzed: " + str(filter_df.count()))
    return filter_df


cnt = 0
filter_df = None
for folder in [x[0] for x in os.walk("./sample_data")]:

    if '20220' in folder:  # inside a daily folder
        try:
            partial_df = pre_process(folder, cand_ls)  # add candidate list
            if cnt == 0:
                filter_df = partial_df
            else:
                filter_df = filter_df.unionByName(partial_df)
            del partial_df
            gc.collect()
        except Py4JJavaError:
            print(folder+" failed")
        cnt += 1

# id_name = filter_df.select('user_id_str', 'user_twitter_handle') \
#     .union(filter_df.select('in_reply_to_user_id_str', 'in_reply_to_screen_name')).na.drop().distinct()
# id_name.write.option("header", True).mode('overwrite').csv('id_name_dict')
# print('id name done')
# del id_name
# gc.collect()

filter_df = filter_df.drop("user_twitter_handle", "in_reply_to_screen_name",
                           "quoted_original_user_twitter_handle", "retweeted_original_user_twitter_handle")

'''
    Step 2:
    Find all interacted (reply, retweet, quote) users.
    if a user A retweet user B's tweet, there will be a directional edge between A and B, A -> B.
'''

filter_df = filter_df.withColumn("connected_user",
                                 array(filter_df.quoted_original_user_id_str, filter_df.in_reply_to_user_id_str,
                                       filter_df.retweeted_original_user_id_str))
filter_df = filter_df.withColumn("connected_user_clean", expr('filter(connected_user, x -> x is not null)')).drop(
    "connected_user")


# TODO: add a column to show if it is original tweet
# filter_df = filter_df.filter(size(filter_df.connected_user_clean) > 0)

filter_df.show()

# each line make sure have only one user and one connected user
sdf = filter_df.withColumn("connected_user_single", explode(filter_df.connected_user_clean)).rdd
print('====')
print(len(sdf.collect()))
to_save_sdf = sdf.toDF()  # .write.mode('overwrite').csv('intermediate_rdd')
to_save_sdf = to_save_sdf.select('user_id_str', 'connected_user_single', 'text')
to_save_sdf.printSchema()
to_save_sdf = to_save_sdf.filter(col('connected_user_single').isNull())
# to_save_sdf.printSchema()

# print('to save df')
# to_save_sdf.show()
# del filter_df
# gc.collect()

# # for each user how many times retweet/replay/quote other tweets
# user_interact_rdd = rdd.map(lambda x: (x[2], 1))
# user_interact_rdd = user_interact_rdd.reduceByKey(lambda a, b: a + b)
#
# # each pair of users (A, B), how many times A retweet this person B
# user_pair_rdd = rdd.map(lambda x: ((x[2], x[15]), 1)).reduceByKey(lambda a, b: a + b)
# user_pair_rdd = user_pair_rdd.map(lambda x: (x[0][0], x[0][1], x[1]))
#
# user_pair_rdd.toDF().write.mode('overwrite').csv('user_pair') #coalesce(1, shuffle = True).
# user_interact_rdd.toDF().write.mode('overwrite').csv('user_interact') #coalesce(1, shuffle = True).

spark.stop()
