import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with Hugging Face model."""
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.classifier = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_model(self):
        """Load the Hugging Face sentiment analysis model."""
        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                return_all_scores=True
            )
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def clean_text(self, text):
        """Clean and preprocess text data."""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text."""
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of a single text."""
        if not self.classifier:
            return "Neutral", 0.5
        
        try:
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            if not cleaned_text:
                return "Neutral", 0.5
            
            # Get sentiment scores
            results = self.classifier(cleaned_text)
            
            # Extract scores
            scores = {result['label']: result['score'] for result in results[0]}
            
            # Determine sentiment and confidence
            if 'POSITIVE' in scores and 'NEGATIVE' in scores:
                if scores['POSITIVE'] > scores['NEGATIVE']:
                    if scores['POSITIVE'] > 0.6:
                        return "Positive", scores['POSITIVE']
                    else:
                        return "Neutral", scores['POSITIVE']
                else:
                    if scores['NEGATIVE'] > 0.6:
                        return "Negative", scores['NEGATIVE']
                    else:
                        return "Neutral", scores['NEGATIVE']
            else:
                return "Neutral", 0.5
                
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return "Neutral", 0.5
    
    def process_reviews(self, df, review_column='review'):
        """Process a DataFrame of reviews and add sentiment analysis."""
        if review_column not in df.columns:
            raise ValueError(f"Column '{review_column}' not found in DataFrame")
        
        # Initialize lists to store results
        sentiments = []
        scores = []
        cleaned_reviews = []
        
        print("Processing reviews...")
        for idx, review in enumerate(df[review_column]):
            if idx % 10 == 0:
                print(f"Processing review {idx + 1}/{len(df)}")
            
            # Clean text
            cleaned_text = self.clean_text(review)
            cleaned_reviews.append(cleaned_text)
            
            # Analyze sentiment
            sentiment, score = self.analyze_sentiment(review)
            sentiments.append(sentiment)
            scores.append(score)
        
        # Add results to DataFrame
        df_result = df.copy()
        df_result['cleaned_review'] = cleaned_reviews
        df_result['sentiment'] = sentiments
        df_result['sentiment_score'] = scores
        
        return df_result

class Visualizer:
    def __init__(self):
        """Initialize the visualizer."""
        self.colors = {
            'Positive': '#2E8B57',  # Sea Green
            'Negative': '#DC143C',  # Crimson
            'Neutral': '#4682B4'    # Steel Blue
        }
    
    def plot_sentiment_distribution(self, df, chart_type='pie'):
        """Create sentiment distribution chart."""
        sentiment_counts = df['sentiment'].value_counts()
        
        if chart_type == 'pie':
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map=self.colors
            )
        else:  # bar chart
            fig = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="Sentiment Distribution",
                color=sentiment_counts.index,
                color_discrete_map=self.colors
            )
            fig.update_layout(xaxis_title="Sentiment", yaxis_title="Count")
        
        fig.update_layout(height=400)
        return fig
    
    def plot_timeline(self, df, date_column='timestamp'):
        """Create timeline chart if timestamp column exists."""
        if date_column not in df.columns:
            return None
        
        try:
            # Convert timestamp to datetime
            df_temp = df.copy()
            df_temp[date_column] = pd.to_datetime(df_temp[date_column])
            
            # Group by date and sentiment
            timeline_data = df_temp.groupby([date_column, 'sentiment']).size().reset_index(name='count')
            
            fig = px.line(
                timeline_data,
                x=date_column,
                y='count',
                color='sentiment',
                title="Sentiment Timeline",
                color_discrete_map=self.colors
            )
            fig.update_layout(height=400)
            return fig
        except Exception as e:
            print(f"Error creating timeline: {e}")
            return None
    
    def create_wordcloud(self, df, sentiment, max_words=100):
        """Create word cloud for specific sentiment."""
        # Filter reviews by sentiment
        sentiment_reviews = df[df['sentiment'] == sentiment]['cleaned_review']
        
        # Combine all reviews
        text = ' '.join(sentiment_reviews.dropna())
        
        if not text:
            return None
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis'
        ).generate(text)
        
        return wordcloud
    
    def plot_sentiment_scores(self, df):
        """Plot distribution of sentiment scores."""
        fig = px.histogram(
            df,
            x='sentiment_score',
            color='sentiment',
            title="Sentiment Score Distribution",
            color_discrete_map=self.colors,
            nbins=20
        )
        fig.update_layout(height=400)
        return fig

def load_sample_data():
    """Load the sample dataset."""
    try:
        df = pd.read_csv('sample_reviews.csv')
        return df
    except FileNotFoundError:
        print("Sample data file not found. Please upload a CSV file.")
        return None

def filter_reviews(df, keyword, sentiment_filter=None):
    """Filter reviews by keyword and sentiment."""
    filtered_df = df.copy()
    
    # Filter by keyword
    if keyword:
        keyword_mask = filtered_df['review'].str.contains(keyword, case=False, na=False)
        filtered_df = filtered_df[keyword_mask]
    
    # Filter by sentiment
    if sentiment_filter and sentiment_filter != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]
    
    return filtered_df

def get_sample_reviews(df, sentiment, n=5):
    """Get sample reviews for a specific sentiment."""
    sentiment_reviews = df[df['sentiment'] == sentiment]['review'].head(n)
    return sentiment_reviews.tolist()

def calculate_metrics(df):
    """Calculate sentiment analysis metrics."""
    total_reviews = len(df)
    sentiment_counts = df['sentiment'].value_counts()
    
    metrics = {
        'total_reviews': total_reviews,
        'positive_count': sentiment_counts.get('Positive', 0),
        'negative_count': sentiment_counts.get('Negative', 0),
        'neutral_count': sentiment_counts.get('Neutral', 0),
        'positive_percentage': (sentiment_counts.get('Positive', 0) / total_reviews) * 100,
        'negative_percentage': (sentiment_counts.get('Negative', 0) / total_reviews) * 100,
        'neutral_percentage': (sentiment_counts.get('Neutral', 0) / total_reviews) * 100,
        'average_score': df['sentiment_score'].mean()
    }
    
    return metrics