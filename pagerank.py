from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import explode, col, udf, concat_ws, from_json, lit, array, expr, size
from pyspark.sql.functions import sum as _sum
import json
from pyspark.sql.types import BooleanType

spark = SparkSession.builder.appName("twitter_applications").getOrCreate()
# filter based on the following keywords
test_list = ['russia', 'ukraine', 'putin', 'zelensky'
                                           'russian', 'ukrainian', 'keiv', 'kyiv']
# regex for filter
regex_pattern = "(?i)" + "|".join(test_list)

raw = spark.read.json("./sample_data/*.json", allowBackslashEscapingAnyCharacter=True)
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
                        "in_reply_to_status_id_str", "in_reply_to_user_id_str",
                        "retweeted_status",
                        "quoted_status",
                        "text", col("entities.hashtags.text").alias("hashtags"),
                        col("quoted_status.created_at").alias("quoted_time"),
                        col("quoted_status.id_str").alias("quoted_original_tweet_id_str"),
                        col("quoted_status.user.id_str").alias("quoted_original_user_id_str"),
                        col("retweeted_status.created_at").alias("retweeted_time"),
                        col("retweeted_status.id_str").alias("retweeted_original_tweet_id_str"),
                        col("retweeted_status.user.id_str").alias("retweeted_original_user_id_str")
                        )

# add all retweeted / quoted original tweet themselves to the twitter df
retweet_rdd = twitter_df.filter(twitter_df.retweeted_status.isNotNull()).select('retweeted_status.*')
quoted_rdd = twitter_df.filter(twitter_df.quoted_status.isNotNull()).select('quoted_status.*')

# select columns
retweet_rdd = retweet_rdd.select("created_at", "id_str",
                                 col("user.id_str").alias("user_id_str"),
                                 "in_reply_to_status_id_str", "in_reply_to_user_id_str",
                                 # "quoted_status",
                                 "text",
                                 col("entities.hashtags.text").alias("hashtags"),
                                 col("quoted_status.created_at").alias("quoted_time"),
                                 col("quoted_status.id_str").alias("quoted_original_tweet_id_str"),
                                 col("quoted_status.user.id_str").alias("quoted_original_user_id_str"),
                                 )

quoted_rdd = quoted_rdd.select("created_at", "id_str", col("user.id_str").alias("user_id_str"),
                               "in_reply_to_status_id_str", "in_reply_to_user_id_str",
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

# convert hashtag column (array type) to str for regex expression filter
raw_df = combined_raw.withColumn("text_hashtag", concat_ws(",", col("hashtags")))

filter_df = raw_df.filter(raw_df.text.rlike(regex_pattern) | raw_df.text_hashtag.rlike(regex_pattern)).distinct()
print("# of tweets analyzed: " + str(filter_df.count()))

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

rdd = filter_df.filter(size(filter_df.connected_user_clean) > 0)
# each line make sure have only one user and one connected user
rdd = filter_df.withColumn("connected_user_single", explode(filter_df.connected_user_clean)).rdd

# for each user how many times retweet/replay/quote other tweets
user_interact_rdd = rdd.map(lambda x: (x[2], 1))
user_interact_rdd = user_interact_rdd.reduceByKey(lambda a, b: a + b)

# each pair of users (A, B), how many times A retweet this person B
user_pair_rdd = rdd.map(lambda x: ((x[2], x[15]), 1)).reduceByKey(lambda a, b: a + b)
user_pair_rdd = user_pair_rdd.map(lambda x: (x[0][0], (x[0][1], x[1])))

'''
    Step 3:
    Implement basic PageRank as described in 
    http://www.diva-portal.org/smash/get/diva2:1104337/FULLTEXT01.pdf Section 3.4.1
'''

# if user A retweet B, the from_user = A, to_user = B
columns = ['from_user', 'to_user', 'retweet_ratio']
edge_rdd = user_pair_rdd.join(user_interact_rdd).map(lambda x: (x[0], x[1][0][0], x[1][0][1] / x[1][1])) \
    .filter(lambda x: x[0] != x[1]).toDF(columns)  # remove all self pointed edges

all_users = edge_rdd.select("from_user").union(edge_rdd.select("to_user")).distinct() \
    .withColumnRenamed("from_user", "users")
# count of all nodes in the graph, use to init the pagerank score
N = all_users.count()
print("number of users in the graph: " + str(N))
all_users = all_users.withColumn("rank", lit(1 / N))

# keep update the score until convergence
for cycle in range(10):
    temp = all_users.join(edge_rdd, all_users.users == edge_rdd.to_user, "left").na.fill(value=0,
                                                                                         subset=["retweet_ratio"])
    temp = temp.withColumn("contri", temp.retweet_ratio * temp.rank)
    all_users = temp.groupby("users").agg(_sum("contri").alias("rank"))
    norm = all_users.rdd.map(lambda x: (1, x[1])).reduceByKey(lambda a, b: a + b).collect()[0][1]
    all_users = all_users.withColumn("rank", all_users.rank / norm)
    all_users.sort(col("rank").desc()).show(5)

# write to CSV
all_users.sort(col("rank").desc()).limit(10).write.csv("./sample_top10")

spark.stop()
