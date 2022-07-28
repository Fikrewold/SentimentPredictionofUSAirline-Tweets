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
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import shuffle
import seaborn as sns


import contractions                                     # Import contractions library.
from bs4 import BeautifulSoup     
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


#for some key natural language processing functions
import re
import re, string, unicodedata                          # Import Regex, string and unicodedata.
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords                       # Import stopwords.
from nltk.tokenize import word_tokenize, sent_tokenize  # Import Tokenizer.
from nltk.stem.wordnet import WordNetLemmatizer         # Import Lemmatizer.
import matplotlib.pyplot as plt   
from sklearn import metrics

# sklearn and keras for machine learning models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import keras
import keras.utils
from keras import utils as np_utils
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Activation, Dropout
from keras.models import Sequential



# Keras for neural networks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
#from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")
1. Data preparation
 - a. analyze missing values
 - b. remove redundant columns
tweets_df = pd.read_csv("Tweets.csv")
tweets_df.head()
tweet_id	airline_sentiment	airline_sentiment_confidence	negativereason	negativereason_confidence	airline	airline_sentiment_gold	name	negativereason_gold	retweet_count	text	tweet_coord	tweet_created	tweet_location	user_timezone
0	570306133677760513	neutral	1.0000	NaN	NaN	Virgin America	NaN	cairdin	NaN	0	@VirginAmerica What @dhepburn said.	NaN	2015-02-24 11:35:52 -0800	NaN	Eastern Time (US & Canada)
1	570301130888122368	positive	0.3486	NaN	0.0000	Virgin America	NaN	jnardino	NaN	0	@VirginAmerica plus you've added commercials t...	NaN	2015-02-24 11:15:59 -0800	NaN	Pacific Time (US & Canada)
2	570301083672813571	neutral	0.6837	NaN	NaN	Virgin America	NaN	yvonnalynn	NaN	0	@VirginAmerica I didn't today... Must mean I n...	NaN	2015-02-24 11:15:48 -0800	Lets Play	Central Time (US & Canada)
3	570301031407624196	negative	1.0000	Bad Flight	0.7033	Virgin America	NaN	jnardino	NaN	0	@VirginAmerica it's really aggressive to blast...	NaN	2015-02-24 11:15:36 -0800	NaN	Pacific Time (US & Canada)
4	570300817074462722	negative	1.0000	Can't Tell	1.0000	Virgin America	NaN	jnardino	NaN	0	@VirginAmerica and it's a really big bad thing...	NaN	2015-02-24 11:14:45 -0800	NaN	Pacific Time (US & Canada)
tweets_df.shape
(14640, 15)
tweets_df.columns
Index(['tweet_id', 'airline_sentiment', 'airline_sentiment_confidence',
       'negativereason', 'negativereason_confidence', 'airline',
       'airline_sentiment_gold', 'name', 'negativereason_gold',
       'retweet_count', 'text', 'tweet_coord', 'tweet_created',
       'tweet_location', 'user_timezone'],
      dtype='object')
Missing Value Analysis

#Check for missing values
100*tweets_df.isna().sum()/len(tweets_df)
tweet_id                         0.000000
airline_sentiment                0.000000
airline_sentiment_confidence     0.000000
negativereason                  37.308743
negativereason_confidence       28.128415
airline                          0.000000
airline_sentiment_gold          99.726776
name                             0.000000
negativereason_gold             99.781421
retweet_count                    0.000000
text                             0.000000
tweet_coord                     93.039617
tweet_created                    0.000000
tweet_location                  32.329235
user_timezone                   32.923497
dtype: float64
we observe that airline_sentiment_gold, negativereason_gold and tweet_coord have more tha 90% of missing values, let us drop them as they don't provide any constructive feedback

#Drop columns which have so many missing values.
tweets_df.drop(['airline_sentiment_gold', 'negativereason_gold', 'tweet_coord'], axis=1, inplace =True)
100*tweets_df.isna().sum()/len(tweets_df)
tweet_id                         0.000000
airline_sentiment                0.000000
airline_sentiment_confidence     0.000000
negativereason                  37.308743
negativereason_confidence       28.128415
airline                          0.000000
name                             0.000000
retweet_count                    0.000000
text                             0.000000
tweet_created                    0.000000
tweet_location                  32.329235
user_timezone                   32.923497
dtype: float64
tweets_df[['negativereason', 'negativereason_confidence', 'tweet_location', 'user_timezone']].head()
negativereason	negativereason_confidence	tweet_location	user_timezone
0	NaN	NaN	NaN	Eastern Time (US & Canada)
1	NaN	0.0000	NaN	Pacific Time (US & Canada)
2	NaN	NaN	Lets Play	Central Time (US & Canada)
3	Bad Flight	0.7033	NaN	Pacific Time (US & Canada)
4	Can't Tell	1.0000	NaN	Pacific Time (US & Canada)
2. Exploratory Data Analysis
# Data balance
def createPieChartFor(t_df):
    Lst = 100*t_df.value_counts()/len(t_df)
    
    # set data for pie chart
    labels = t_df.value_counts().index.values
    sizes =  Lst 
    
    # set labels
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
createPieChartFor(tweets_df.airline_sentiment)

from above we can see that we have majority of negative comments (63%) followed by neutral (21%) and positive (16%)

createPieChartFor(tweets_df.airline)

airline_sentiment_df = tweets_df.groupby(['airline','airline_sentiment']).airline_sentiment.count().unstack()
airline_sentiment_df.plot(kind='bar')
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left");

airline_sentiment_df = tweets_df.groupby(['airline','airline_sentiment']).airline_sentiment.count().unstack()
airline_sentiment_df.plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left");

From above graph we can see that

United, US Airways and American have substatially negative tweets, these also have got over all more tweets
Virgin America, Delta and Southwest have fairly balanced tweets
tweets_df.negativereason.value_counts()
Customer Service Issue         2910
Late Flight                    1665
Can't Tell                     1190
Cancelled Flight                847
Lost Luggage                    724
Bad Flight                      580
Flight Booking Problems         529
Flight Attendant Complaints     481
longlines                       178
Damaged Luggage                  74
Name: negativereason, dtype: int64
tweets_df.negativereason.value_counts().plot(kind='bar', figsize=(15,5));

As we can see majority tweets have said the reason are Customer servicec issue and Late flight
3.Text Pre-processing & Sentiment Analysis
Text Pre-processing:
Remove all the special characters
convert all letters to lower case
filter out english stop words
stemmer (optional)
Remove html tags.
Replace contractions in string. (e.g. replace I'm --> I am) and so on.
Remove numbers.
Tokenization
To remove Stopwords.
Lemmatized data
NLTK library will be used to tokenize words, remove stopwords and lemmatize the remaining words.

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")                    
    return soup.get_text()

tweets_df['text'] = tweets_df['text'].apply(lambda x: strip_html(x))
tweets_df.head()
tweet_id	airline_sentiment	airline_sentiment_confidence	negativereason	negativereason_confidence	airline	name	retweet_count	text	tweet_created	tweet_location	user_timezone
0	570306133677760513	neutral	1.0000	NaN	NaN	Virgin America	cairdin	0	@VirginAmerica What @dhepburn said.	2015-02-24 11:35:52 -0800	NaN	Eastern Time (US & Canada)
1	570301130888122368	positive	0.3486	NaN	0.0000	Virgin America	jnardino	0	@VirginAmerica plus you've added commercials t...	2015-02-24 11:15:59 -0800	NaN	Pacific Time (US & Canada)
2	570301083672813571	neutral	0.6837	NaN	NaN	Virgin America	yvonnalynn	0	@VirginAmerica I didn't today... Must mean I n...	2015-02-24 11:15:48 -0800	Lets Play	Central Time (US & Canada)
3	570301031407624196	negative	1.0000	Bad Flight	0.7033	Virgin America	jnardino	0	@VirginAmerica it's really aggressive to blast...	2015-02-24 11:15:36 -0800	NaN	Pacific Time (US & Canada)
4	570300817074462722	negative	1.0000	Can't Tell	1.0000	Virgin America	jnardino	0	@VirginAmerica and it's a really big bad thing...	2015-02-24 11:14:45 -0800	NaN	Pacific Time (US & Canada)
tweets_df.text
0                      @VirginAmerica What @dhepburn said.
1        @VirginAmerica plus you've added commercials t...
2        @VirginAmerica I didn't today... Must mean I n...
3        @VirginAmerica it's really aggressive to blast...
4        @VirginAmerica and it's a really big bad thing...
                               ...                        
14635    @AmericanAir thank you we got on a different f...
14636    @AmericanAir leaving over 20 minutes Late Flig...
14637    @AmericanAir Please bring American Airlines to...
14638    @AmericanAir you have my money, you change my ...
14639    @AmericanAir we have 8 ppl so we need 2 know h...
Name: text, Length: 14640, dtype: object
def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

tweets_df['text'] = tweets_df['text'].apply(lambda x: replace_contractions(x))
tweets_df.head()
tweet_id	airline_sentiment	airline_sentiment_confidence	negativereason	negativereason_confidence	airline	name	retweet_count	text	tweet_created	tweet_location	user_timezone
0	570306133677760513	neutral	1.0000	NaN	NaN	Virgin America	cairdin	0	@VirginAmerica What @dhepburn said.	2015-02-24 11:35:52 -0800	NaN	Eastern Time (US & Canada)
1	570301130888122368	positive	0.3486	NaN	0.0000	Virgin America	jnardino	0	@VirginAmerica plus you have added commercials...	2015-02-24 11:15:59 -0800	NaN	Pacific Time (US & Canada)
2	570301083672813571	neutral	0.6837	NaN	NaN	Virgin America	yvonnalynn	0	@VirginAmerica I did not today... Must mean I ...	2015-02-24 11:15:48 -0800	Lets Play	Central Time (US & Canada)
3	570301031407624196	negative	1.0000	Bad Flight	0.7033	Virgin America	jnardino	0	@VirginAmerica it is really aggressive to blas...	2015-02-24 11:15:36 -0800	NaN	Pacific Time (US & Canada)
4	570300817074462722	negative	1.0000	Can't Tell	1.0000	Virgin America	jnardino	0	@VirginAmerica and it is a really big bad thin...	2015-02-24 11:14:45 -0800	NaN	Pacific Time (US & Canada)
def remove_numbers(text):
  text = re.sub(r'\d+', '', text)
  return text

tweets_df['text'] = tweets_df['text'].apply(lambda x: remove_numbers(x))
tweets_df.text
0                      @VirginAmerica What @dhepburn said.
1        @VirginAmerica plus you have added commercials...
2        @VirginAmerica I did not today... Must mean I ...
3        @VirginAmerica it is really aggressive to blas...
4        @VirginAmerica and it is a really big bad thin...
                               ...                        
14635    @AmericanAir thank you we got on a different f...
14636    @AmericanAir leaving over  minutes Late Flight...
14637    @AmericanAir Please bring American Airlines to...
14638    @AmericanAir you have my money, you change my ...
14639    @AmericanAir we have  people so we need  know ...
Name: text, Length: 14640, dtype: object
tweets_df['text'] = tweets_df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1) # Tokenization of data
tweets_df.text
0           [@, VirginAmerica, What, @, dhepburn, said, .]
1        [@, VirginAmerica, plus, you, have, added, com...
2        [@, VirginAmerica, I, did, not, today, ..., Mu...
3        [@, VirginAmerica, it, is, really, aggressive,...
4        [@, VirginAmerica, and, it, is, a, really, big...
                               ...                        
14635    [@, AmericanAir, thank, you, we, got, on, a, d...
14636    [@, AmericanAir, leaving, over, minutes, Late,...
14637    [@, AmericanAir, Please, bring, American, Airl...
14638    [@, AmericanAir, you, have, my, money, ,, you,...
14639    [@, AmericanAir, we, have, people, so, we, nee...
Name: text, Length: 14640, dtype: object
stopwords = stopwords.words('english')

customlist = ['not', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
        "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
        "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
        "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

# Set custom stop-word's list as not, couldn't etc. words matter in Sentiment, so not removing them from original data.

stopwords = list(set(stopwords) - set(customlist))         
lemmatizer = WordNetLemmatizer()

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words

def lemmatize_list(words):
    new_words = []
    for word in words:
      new_words.append(lemmatizer.lemmatize(word, pos='v'))
    return new_words

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = lemmatize_list(words)
    return ' '.join(words)
tweets_df['clean_tweet'] = tweets_df.apply(lambda row: normalize(row['text']), axis=1)
tweets_df.clean_tweet
0                               virginamerica dhepburn say
1        virginamerica plus add commercials experience ...
2        virginamerica not today must mean need take an...
3        virginamerica really aggressive blast obnoxiou...
4                       virginamerica really big bad thing
                               ...                        
14635       americanair thank get different flight chicago
14636    americanair leave minutes late flight warn com...
14637    americanair please bring american airlines bla...
14638    americanair money change flight not answer pho...
14639    americanair people need know many seat next fl...
Name: clean_tweet, Length: 14640, dtype: object
Sentiment Analysis - Word Cloud
import wordcloud
def show_wordcloud(tweets_df,  title):
    text = ' '.join(tweets_df['clean_tweet'].astype(str).tolist())                 # Converting Summary column into list
    stopwords = set(wordcloud.STOPWORDS)                                  # instantiate the stopwords from wordcloud
    
    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords,background_color='white',          # Setting the different parameter of stopwords
                    colormap='viridis', width=800, height=600).generate(text)
    
    plt.figure(figsize=(14,11), frameon=True)                             
    plt.imshow(fig_wordcloud)  
    plt.axis('off')
    plt.title(title, fontsize=30)
    plt.show()
show_wordcloud(tweets_df,'Summary Word_Cloud')

we observe that flight, jetblue, southwestair, usairway, americanair, customer service, cancel flight present more frequently in word cloud.

show_wordcloud(tweets_df[tweets_df.airline_sentiment=='negative'], title = "Negative Sentiment")

As we notice that negative emotions gave words such as delayed, hold, canceled, lost, late, hours, waiting, service, ect. Furthermore, we have here US Airways and American airlines. I would say that the passenger who wrote these negative reviews must of faced challenges while flying under these airline, such as, delayed or canceled flight, the waiting time took longer than expected, the service was bad, their luggage were lost or destroyed.
show_wordcloud(tweets_df[tweets_df.airline_sentiment=='positive'], title = "Positive Sentiment")

The result shows that words expressing positive emotions are thanks, good, great, better, awesome. we can also see that Southwest and Virgin America are mentioned here. This indicate that the passengers who used SouthWest and Virgin America were delighted with their flight, the service was good, their luggage were unharmed, the plane came on time, ect.
4. Model Building
Spliting the Data into Training, Validation and Test Sets
seed =42
# Create test set
test_set = tweets_df[:1000]

# Create train and validation sets
X_train, X_val, y_train, y_val = train_test_split(tweets_df['clean_tweet'][1000:],
                                                      tweets_df['airline_sentiment'][1000:],
                                                      test_size=0.2,
                                                      random_state=seed)

# Get sentiment labels for test set
y_test = test_set['airline_sentiment']
X_train.shape, X_val.shape, y_test.shape
((10912,), (2728,), (1000,))
Now that we've split our data into train, validation and test sets, we'll TF-IDF vectorize them

TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_val = vectorizer.transform(X_val)
X_test = vectorizer.transform(test_set['clean_tweet'])
Next, we'll build a Decision Tree Classifier, Random Forest Classifier, Gradient-Boosting classifier (GBM), an XGBoost classifier, and a relatively straightforward neural network with keras and compare how each of these models performs. Oftentimes it's hard to tell which architecture will perform best without testing them out. Comet's project-level view helps make it easy to compare how different experiments are performing and let you easily move from model selection to model tuning.

Decision Tree classifier
# sklearn's Gradient Boosting Classifier (GBM)
dt =  DecisionTreeClassifier()

dt.fit(X_train, y_train)
# Check results
train_pred = dt.predict(X_train)
val_pred = dt.predict(X_val)
print(f'Accuracy on training set (DecisionTree): {round(accuracy_score(y_train, train_pred)*100, 4)}%')
print(f'Accuracy on validation set (DecisionTree): {round(accuracy_score(y_val,val_pred)*100, 4)}%')

val_accuracy = round(accuracy_score(y_val,val_pred)*100, 4)
train_accuracy = round(accuracy_score(y_train, train_pred)*100, 4)
Accuracy on training set (DecisionTree): 99.6151%
Accuracy on validation set (DecisionTree): 67.7786%
Random Forest Classifier
rf = RandomForestClassifier(n_estimators=200)

rf.fit(X_train, y_train)
# Check results
train_pred = rf.predict(X_train)
val_pred = rf.predict(X_val)
print(f'Accuracy on training set (RandomForest): {round(accuracy_score(y_train, train_pred)*100, 4)}%')
print(f'Accuracy on validation set (RandomForest): {round(accuracy_score(y_val,val_pred)*100, 4)}%')
Accuracy on training set (RandomForest): 99.6151%
Accuracy on validation set (RandomForest): 77.456%
High level of overfitting observed in Decission Tree and Random Forest models, despite better prediction observed by random forest model.
Gradient-Boosting classifier (GBM)
# sklearn's Gradient Boosting Classifier (GBM)
gbm = GradientBoostingClassifier(n_estimators=200, 
                                 max_depth=6, 
                                 random_state=seed)
gbm.fit(X_train, y_train)
# Check results
train_pred = gbm.predict(X_train)
val_pred = gbm.predict(X_val)
print(f'Accuracy on training set (GBM): {round(accuracy_score(y_train, train_pred)*100, 4)}%')
print(f'Accuracy on validation set (GBM): {round(accuracy_score(y_val,val_pred)*100, 4)}%')

val_accuracy = round(accuracy_score(y_val,val_pred)*100, 4)
train_accuracy = round(accuracy_score(y_train, train_pred)*100, 4)
Accuracy on training set (GBM): 88.0773%
Accuracy on validation set (GBM): 78.1891%
XGBoost classifier
xgb_params = {'objective' : 'multi:softmax',
              'eval_metric' : 'mlogloss',
              'eta' : 0.1,
              'max_depth' : 6,
              'num_class' : 3,
              'lambda' : 0.8,
              'estimators' : 200,
              'seed' : seed
              
}

# Transform categories into numbers
# negative = 0, neutral = 1 and positive = 2
target_train = y_train.astype('category').cat.codes
target_val = y_val.astype('category').cat.codes

# Transform data into a matrix so that we can use XGBoost
d_train = xgb.DMatrix(X_train, label = target_train)
d_val = xgb.DMatrix(X_val, label = target_val)

# Fit XGBoost
watchlist = [(d_train, 'train'), (d_val, 'validation')]
bst = xgb.train(xgb_params, 
                d_train, 
                400,  
                watchlist,
                early_stopping_rounds = 50, 
                verbose_eval = 0)

# Check results for XGBoost
train_pred = bst.predict(d_train)
val_pred = bst.predict(d_val)
print(f'Accuracy on training set (XGBoost): {round(accuracy_score(target_train, train_pred)*100, 4)}%')
print(f'Accuracy on validation set (XGBoost): {round(accuracy_score(target_val, val_pred)*100, 4)}%')
[22:11:10] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.6.0/src/learner.cc:627: 
Parameters: { "estimators" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.


Accuracy on training set (XGBoost): 89.7086%
Accuracy on validation set (XGBoost): 78.4824%
Neural Network
# Generator so we can easily feed batches of data to the neural network
def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = X.shape[0]/batch_size
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

# Onehot encoding of target variable
# Negative = [1,0,0], Neutral = [0,1,0], Positive = [0,0,1]

# Initialize sklearn's one-hot encoder class
onehot_encoder = OneHotEncoder(sparse=False)

# One hot encoding for training set
integer_encoded_train = np.array(y_train).reshape(len(y_train), 1)
onehot_encoded_train = onehot_encoder.fit_transform(integer_encoded_train)

# One hot encoding for validation set
integer_encoded_val = np.array(y_val).reshape(len(y_val), 1)
onehot_encoded_val = onehot_encoder.fit_transform(integer_encoded_val)

# Neural network architecture
initializer = keras.initializers.he_normal(seed=seed)
activation = keras.activations.elu
optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=4)

# Build model architecture
model = Sequential()
model.add(Dense(20, activation=activation, kernel_initializer=initializer, input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax', kernel_initializer=initializer))
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Hyperparameters
epochs = 15
batch_size = 16

# Fit the model using the batch_generator
hist = model.fit_generator(generator=batch_generator(X_train, onehot_encoded_train, batch_size=batch_size, shuffle=True),
                           epochs=epochs, validation_data=(X_val, onehot_encoded_val),
                           steps_per_epoch=X_train.shape[0]/batch_size, callbacks=[es])
Epoch 1/15
682/682 [==============================] - ETA: 0s - loss: 0.5949 - accuracy: 0.6269WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.5949 - accuracy: 0.6269 - val_loss: 0.5147 - val_accuracy: 0.6397
Epoch 2/15
647/682 [===========================>..] - ETA: 0s - loss: 0.4997 - accuracy: 0.6362WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.4996 - accuracy: 0.6354 - val_loss: 0.4689 - val_accuracy: 0.6433
Epoch 3/15
674/682 [============================>.] - ETA: 0s - loss: 0.4614 - accuracy: 0.6485WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.4611 - accuracy: 0.6486 - val_loss: 0.4383 - val_accuracy: 0.6675
Epoch 4/15
669/682 [============================>.] - ETA: 0s - loss: 0.4309 - accuracy: 0.6785WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.4297 - accuracy: 0.6804 - val_loss: 0.4120 - val_accuracy: 0.6968
Epoch 5/15
675/682 [============================>.] - ETA: 0s - loss: 0.4018 - accuracy: 0.7174WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.4022 - accuracy: 0.7166 - val_loss: 0.3914 - val_accuracy: 0.7331
Epoch 6/15
667/682 [============================>.] - ETA: 0s - loss: 0.3798 - accuracy: 0.7455WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.3792 - accuracy: 0.7466 - val_loss: 0.3756 - val_accuracy: 0.7485
Epoch 7/15
677/682 [============================>.] - ETA: 0s - loss: 0.3599 - accuracy: 0.7698WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.3601 - accuracy: 0.7697 - val_loss: 0.3632 - val_accuracy: 0.7636
Epoch 8/15
668/682 [============================>.] - ETA: 0s - loss: 0.3439 - accuracy: 0.7870WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.3440 - accuracy: 0.7866 - val_loss: 0.3528 - val_accuracy: 0.7768
Epoch 9/15
641/682 [===========================>..] - ETA: 0s - loss: 0.3299 - accuracy: 0.8075WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.3290 - accuracy: 0.8079 - val_loss: 0.3442 - val_accuracy: 0.7845
Epoch 10/15
646/682 [===========================>..] - ETA: 0s - loss: 0.3153 - accuracy: 0.8200WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.3154 - accuracy: 0.8203 - val_loss: 0.3369 - val_accuracy: 0.7907
Epoch 11/15
669/682 [============================>.] - ETA: 0s - loss: 0.3026 - accuracy: 0.8309WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.3035 - accuracy: 0.8302 - val_loss: 0.3309 - val_accuracy: 0.7922
Epoch 12/15
677/682 [============================>.] - ETA: 0s - loss: 0.2932 - accuracy: 0.8446WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.2929 - accuracy: 0.8449 - val_loss: 0.3259 - val_accuracy: 0.7929
Epoch 13/15
656/682 [===========================>..] - ETA: 0s - loss: 0.2798 - accuracy: 0.8499WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.2800 - accuracy: 0.8501 - val_loss: 0.3215 - val_accuracy: 0.7977
Epoch 14/15
668/682 [============================>.] - ETA: 0s - loss: 0.2705 - accuracy: 0.8617WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.2702 - accuracy: 0.8618 - val_loss: 0.3184 - val_accuracy: 0.7973
Epoch 15/15
663/682 [============================>.] - ETA: 0s - loss: 0.2612 - accuracy: 0.8660WARNING:tensorflow:Early stopping conditioned on metric `val_acc` which is not available. Available metrics are: loss,accuracy,val_loss,val_accuracy
682/682 [==============================] - 1s 1ms/step - loss: 0.2628 - accuracy: 0.8648 - val_loss: 0.3155 - val_accuracy: 0.8013
print(model.evaluate(X_train, onehot_encoded_train))  # Evaluate on train set
print(model.evaluate(X_val, onehot_encoded_val))  
341/341 [==============================] - 0s 682us/step - loss: 0.2370 - accuracy: 0.8844
[0.23695272207260132, 0.8844391703605652]
86/86 [==============================] - 0s 687us/step - loss: 0.3155 - accuracy: 0.8013
[0.315542072057724, 0.8013196587562561]
Comparing our models, we can see that our Neural Network models are outperforming the XGBoost and LGBM, Decision Treee, Random Forest models experiments by a considerable margin.
High level of overfitting observed in Decission Tree and Random Forest models.
The result shows that our neural network model model has accuracy of 80.13% on validation dataset, meaning that 80.13% of our data is correctly classified.
5. Conclusion
In this work we extracted many information about the given datatset. Which are:
United, US Airways and American have substatially negative tweets, these also have got over all more tweets Virgin America, Delta and Southwest have fairly balanced tweets
26% of tweets were about United airline, it was also the most complained about airline due to bad service and late flights.
More than 60% of the tweets expressed negative emotions.

The longer the tweet the more it expresses negative feelings.

As for the sentiment classification we found that the Neural Network model did best at predicting the sentiments.
