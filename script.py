import praw
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

#nltk.download('vader_lexicon')

# Set up Reddit API client
reddit = praw.Reddit(
    client_id='insert your client ID',
    client_secret='insert your client secret',
    user_agent='insert your reddit username'
)

# Function to search for posts containing specific keywords
def search_reddit_posts(subreddit, query, limit=100):
    posts = []
    for submission in reddit.subreddit(subreddit).search(query, limit=limit):
        posts.append({
            'title': submission.title,
            'selftext': submission.selftext,
            'score': submission.score,
            'num_comments': submission.num_comments,
            'created_utc': submission.created_utc
        })
    return pd.DataFrame(posts)

# Function to perform sentiment analysis
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)
    return sentiment

# Search for posts containing specific keywords
subreddit = 'slovenia'  # Replace with the subreddit you want to analyze
query = 'travel'  # Replace with the keywords you want to search for
posts_df = search_reddit_posts(subreddit, query)

# Combine title and selftext for sentiment analysis
posts_df['content'] = posts_df['title'] + ' ' + posts_df['selftext']

# Analyze sentiment for each post
posts_df['sentiment'] = posts_df['content'].apply(analyze_sentiment)

# Extract sentiment components into separate columns
sentiment_df = posts_df['sentiment'].apply(pd.Series)
posts_df = pd.concat([posts_df, sentiment_df], axis=1)
posts_df.drop(columns=['selftext'], inplace=True)

# Display the dataframe with sentiment scores
print(posts_df[['title', 'neg', 'neu', 'pos', 'compound']])

# Save the dataframe to a CSV file
csv_filename = 'reddit_posts_sentiment.csv'
posts_df.to_csv(csv_filename, index=False)

print(f"Data saved to {csv_filename}")
