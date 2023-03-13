from py4j.protocol import Py4JJavaError
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import explode, col, udf, concat_ws, from_json, lit, array, expr, size
from pyspark.sql.functions import sum as _sum
from pyspark.sql.types import *
import json
import os
from pyspark.sql.types import BooleanType
import sys
print(sys.version)
conf = SparkConf()
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

spark = SparkSession.builder.appName("twitter_applications")\
    .config("spark.sql.files.ignoreCorruptFiles", "true")\
    .config('spark.executor.memory', '45G')\
    .config('spark.driver.memory', '80G')\
    .config('spark.driver.maxResultSize', '10G')\
    .getOrCreate()

'''
    Step 3:
    Implement basic PageRank as described in 
    http://www.diva-portal.org/smash/get/diva2:1104337/FULLTEXT01.pdf Section 3.4.1
'''
user_pair_rdd = spark.read \
     .csv("./user_pair/*.csv").rdd
user_pair_rdd = user_pair_rdd.map(lambda x: (x[0], (x[1], x[2])))
user_interact_rdd = spark.read \
     .csv("./user_interact/*.csv").rdd
id_name = spark.read.option("header",True).csv("./id_name_dict/*.csv")
id_name.show()
# if user A retweet B, the from_user = A, to_user = B
columns = ['from_user', 'to_user', 'retweet_ratio']
edge_rdd = user_pair_rdd.join(user_interact_rdd).map(lambda x: (x[0], x[1][0][0], float(x[1][0][1]) / float(x[1][1]))) \
    .filter(lambda x: x[0] != x[1]).toDF(columns)  # remove all self pointed edges

all_users = edge_rdd.select("from_user").union(edge_rdd.select("to_user")).distinct() \
    .withColumnRenamed("from_user", "users")
# count of all nodes in the graph, use to init the pagerank score
N = all_users.count()
print("number of users in the graph: " + str(N))
all_users = all_users.withColumn("rank", lit(1 / N))
damping_factor = 0.85

# keep update the score until convergence
for cycle in range(10):
    temp = all_users.join(edge_rdd, all_users.users == edge_rdd.to_user, "left").na.fill(value=0,
                                                                                         subset=["retweet_ratio"])
    temp = temp.withColumn("contri", temp.retweet_ratio * temp.rank)
    all_users = temp.groupby("users").agg(_sum("contri").alias("rank")) #*lit(damping_factor) + lit((1 - damping_factor) / N)
    all_users = all_users.withColumn("rank", all_users.rank * damping_factor + (1-damping_factor) /N)
    norm = all_users.rdd.map(lambda x: (1, x[1])).reduceByKey(lambda a, b: a + b).collect()[0][1]
    all_users = all_users.withColumn("rank", all_users.rank / norm)
    # all_users.sort(col("rank").desc()).show(5)

results = all_users.sort(col("rank").desc()).limit(10).join(id_name, id_name.user_id_str == all_users.users, "left") \
    .select("users", "rank", "user_twitter_handle")
results.show(50)

# write to CSV
results.sort(col("rank").desc()).limit(50).write.mode('overwrite').csv("./test_pokemon_damping")

spark.stop()
