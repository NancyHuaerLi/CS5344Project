from py4j.protocol import Py4JJavaError
from pyspark import SparkConf, SparkContext
import pyspark
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col, udf, concat_ws, from_json, lit, array, expr, size

import gc


# conf = SparkConf()
# sc = SparkContext(conf=conf)
# sc.setLogLevel("ERROR")
#
# spark = SparkSession.builder.appName("twitter_applications")\
#     .config("spark.sql.files.ignoreCorruptFiles", "true").getOrCreate()
#
# # filter based on the following keywords
# # test_list = ['russia', 'ukraine', 'putin', 'zelensky'
# #                                            'russian', 'ukrainian', 'keiv', 'kyiv']
# test_list = ['nintendo', 'pokemon', 'video game', 'game', 'pokémon legends: arceus', 'pokémon', 'legend of arceus',
#              'legend arceus', 'legends of arceus', 'pokemon legends: arceus', 'stream', 'twitch', 'arceus']
# # regex for filter
# regex_pattern = "(?i)" + "|".join(test_list)
# regex_pattern_hashtag = "(?i)" + "|".join([x.strip(' ') for x in test_list])
#
#
# # raw = spark.read.json("./sample_data/*.json", allowBackslashEscapingAnyCharacter=True)
# # raw = spark.read.json("./202202*/*.json", allowBackslashEscapingAnyCharacter=True)
def keyword_filter(spark, regex_pattern, folder):
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

    # add all retweeted / quoted original tweet themselves to the twitter df
    retweet_rdd = twitter_df.filter(twitter_df.retweeted_status.isNotNull()).select('retweeted_status.*')
    quoted_rdd = twitter_df.filter(twitter_df.quoted_status.isNotNull()).select('quoted_status.*')

    # select columns
    retweet_rdd = retweet_rdd.select("created_at", "id_str",
                                     col("user.id_str").alias("user_id_str"),
                                     col("user.screen_name").alias("user_twitter_handle"),
                                     "in_reply_to_status_id_str", "in_reply_to_user_id_str", "in_reply_to_screen_name",
                                     # "quoted_status",
                                     "text",
                                     col("entities.hashtags.text").alias("hashtags"),
                                     col("quoted_status.created_at").alias("quoted_time"),
                                     col("quoted_status.id_str").alias("quoted_original_tweet_id_str"),
                                     col("quoted_status.user.id_str").alias("quoted_original_user_id_str"),
                                     col("quoted_status.user.screen_name").alias("quoted_original_user_twitter_handle")
                                     )

    quoted_rdd = quoted_rdd.select("created_at", "id_str", col("user.id_str").alias("user_id_str"),
                                   "in_reply_to_status_id_str", "in_reply_to_user_id_str", "in_reply_to_screen_name",
                                   "text", col("entities.hashtags.text").alias("hashtags")
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