**Sentiment Prediction of US Airline Tweets**

**Data Description**

The dataset used in this project is provided on Kaggle and originally collected by Crowdflower’s Data for Everyone library.

This Twitter data was scraped on February 2015. It contains tweets on six major United States(US) airlines.

The dataset contains 14640 instances which are tweets submitted by individual travelers and 15 features.And each instance is labeled as positive, negative or neutral.

**Features description:**

tweet_id: A numeric feature which give the twitter ID of the tweet’s writer.
airline_sentiment: A categorical feature contains labels for tweets, positive, negative or neutral.
airline_sentiment_confidence: A numeric feature representing the confidence level of classifying the tweet to one of the 3 classes.
negativereason: Categorical feature which represent the reason behind considering this tweet as negative.
negativereason_confidence: The level of confidence in determining the negative reason behind the negative tweet.
airline: Name of the airline Company
airline_sentiment_gold
negativereason_gold
retweet_count: Number of retweets of a tweet.
text: Original tweet posted by the user.
tweet_coord: The coordinates of the tweet.
tweet_created**: The date and the time of tweet.
tweet_location**: From where the tweet was posted.
user_timezone: The timezone of the user.



- In this work I want to determine which airlines tweeted about the most and the reason behind the negative tweets. Then we will see the most frequent words used in the tweets using the WordCloud technique. And finally We will predict the sentiment of a tweet without any other information but the tweet text itself, using carious machine learning algorithms.
