# ref: https://realpython.com/python-nltk-sentiment-analysis/#using-nltks-pre-trained-sentiment-analyzer
# nltk VADER classifier is good for tweets.
# VADER: Application on other langulage  is based entirely on
# using a machine-translation web service to automatically translate the text into English.


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

sia = SentimentIntensityAnalyzer()
score = sia.polarity_scores("Wow, I love you!")

# You’ll get back a dictionary of different scores.
# The negative, neutral, and positive scores are related: They all add up to 1 and can’t be negative.
# The compound score is calculated differently. It’s not just an average, and it can range from -1 to 1.
# score = {'neg': 0.0, 'neu': 0.295, 'pos': 0.705, 'compound': 0.8012}

