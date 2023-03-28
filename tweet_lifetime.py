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