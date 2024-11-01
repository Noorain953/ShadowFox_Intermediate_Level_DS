import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('X data.csv')
df.head()
print(df.head())
print(df.columns)
df['cleaned_text'] = df['clean_text'].apply(lambda x: re.sub(r'[^A-Za-z\s]', '', str(x)).lower().strip())

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Apply VADER to classify sentiments
df['scores'] = df['cleaned_text'].apply(lambda text: sid.polarity_scores(text))
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['sentiment'] = df['compound'].apply(lambda c: 'positive' if c > 0.05 else ('negative' if c < -0.05 else 'neutral'))

# Statistical analysis and visualization for sentiment distribution
sentiment_counts = df['sentiment'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title("Sentiment Distribution on X")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
