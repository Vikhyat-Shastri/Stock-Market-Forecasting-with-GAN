"""
Advanced sentiment analysis module using FinBERT for financial news.
Scrapes and analyzes news from multiple sources to generate sentiment scores.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from functools import lru_cache
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class FinBERTSentimentAnalyzer:
    """
    Financial sentiment analysis using FinBERT model.
    Pretrained on financial news and SEC filings.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading FinBERT model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"FinBERT loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text to analyze
        
        Returns:
            Dictionary with sentiment scores (positive, negative, neutral)
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to scores
            scores = predictions.cpu().numpy()[0]
            
            return {
                'positive': float(scores[0]),
                'negative': float(scores[1]),
                'neutral': float(scores[2]),
                'compound': float(scores[0] - scores[1])  # Net sentiment
            }
        
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34,
                'compound': 0.0
            }
    
    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, float]]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
        
        Returns:
            List of sentiment dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Convert to scores
                scores = predictions.cpu().numpy()
                
                for score in scores:
                    results.append({
                        'positive': float(score[0]),
                        'negative': float(score[1]),
                        'neutral': float(score[2]),
                        'compound': float(score[0] - score[1])
                    })
            
            except Exception as e:
                logger.error(f"Batch sentiment analysis failed: {e}")
                # Return neutral scores for failed batch
                results.extend([{
                    'positive': 0.33,
                    'negative': 0.33,
                    'neutral': 0.34,
                    'compound': 0.0
                }] * len(batch))
        
        return results


class NewsScraperBase:
    """Base class for news scraping with common utilities."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = requests.Session()
    
    def _make_request(self, url: str, retries: int = 3) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic.
        
        Args:
            url: URL to request
            retries: Number of retries
        
        Returns:
            Response object or None
        """
        for attempt in range(retries):
            try:
                response = self.session.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                return response
            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None


class YahooFinanceNewsScraper(NewsScraperBase):
    """Scrape financial news from Yahoo Finance."""
    
    def scrape_news(self, ticker: str, max_articles: int = 10) -> List[Dict]:
        """
        Scrape news for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            max_articles: Maximum number of articles to scrape
        
        Returns:
            List of news articles with title, summary, date
        """
        logger.info(f"Scraping Yahoo Finance news for {ticker}")
        
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        response = self._make_request(url)
        
        if not response:
            logger.warning(f"Failed to scrape news for {ticker}")
            return []
        
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Find news articles (structure may change)
            news_items = soup.find_all('div', {'class': 'Ov(h)'})[:max_articles]
            
            for item in news_items:
                try:
                    title_elem = item.find('h3')
                    summary_elem = item.find('p')
                    time_elem = item.find('time')
                    
                    if title_elem:
                        article = {
                            'title': title_elem.text.strip() if title_elem else '',
                            'summary': summary_elem.text.strip() if summary_elem else '',
                            'date': time_elem.get('datetime') if time_elem else datetime.now().isoformat(),
                            'source': 'Yahoo Finance',
                            'ticker': ticker
                        }
                        articles.append(article)
                
                except Exception as e:
                    logger.debug(f"Failed to parse article: {e}")
                    continue
            
            logger.info(f"Scraped {len(articles)} articles for {ticker}")
            return articles
        
        except Exception as e:
            logger.error(f"News scraping failed: {e}")
            return []


class GoogleNewsAPIScraper:
    """
    Scrape news using Google News API (requires API key).
    More reliable than web scraping for production use.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Google News API scraper.
        
        Args:
            api_key: Google News API key (from environment variable if None)
        """
        self.api_key = api_key or os.getenv('GOOGLE_NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2/everything"
    
    def scrape_news(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        max_articles: int = 100
    ) -> List[Dict]:
        """
        Scrape news using Google News API.
        
        Args:
            query: Search query (e.g., "Goldman Sachs")
            from_date: Start date for news
            max_articles: Maximum articles to retrieve
        
        Returns:
            List of news articles
        """
        if not self.api_key:
            logger.warning("Google News API key not found")
            return []
        
        from_date = from_date or (datetime.now() - timedelta(days=30))
        
        params = {
            'q': query,
            'from': from_date.strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'pageSize': min(max_articles, 100),
            'apiKey': self.api_key,
            'language': 'en',
            'domains': 'bloomberg.com,wsj.com,ft.com,reuters.com,cnbc.com'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'summary': article.get('description', ''),
                    'content': article.get('content', ''),
                    'date': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', '')
                })
            
            logger.info(f"Retrieved {len(articles)} articles from Google News API")
            return articles
        
        except Exception as e:
            logger.error(f"Google News API request failed: {e}")
            return []


class SentimentFeatureGenerator:
    """
    Generate sentiment features for stock prediction.
    Combines news scraping and sentiment analysis.
    """
    
    def __init__(
        self,
        use_finbert: bool = True,
        google_api_key: Optional[str] = None
    ):
        """
        Initialize sentiment feature generator.
        
        Args:
            use_finbert: Whether to use FinBERT (requires GPU for speed)
            google_api_key: Google News API key (optional)
        """
        self.sentiment_analyzer = None
        if use_finbert:
            try:
                self.sentiment_analyzer = FinBERTSentimentAnalyzer()
            except Exception as e:
                logger.warning(f"Failed to load FinBERT, using fallback: {e}")
        
        self.yahoo_scraper = YahooFinanceNewsScraper()
        self.google_scraper = GoogleNewsAPIScraper(google_api_key)
    
    def generate_daily_sentiment(
        self,
        ticker: str,
        date: datetime,
        lookback_days: int = 1
    ) -> Dict[str, float]:
        """
        Generate sentiment scores for a specific date.
        
        Args:
            ticker: Stock ticker
            date: Target date
            lookback_days: Days to look back for news
        
        Returns:
            Dictionary with sentiment features
        """
        # Scrape news
        articles = self.yahoo_scraper.scrape_news(ticker, max_articles=20)
        
        if not articles:
            return self._get_neutral_sentiment()
        
        # Analyze sentiment
        texts = [f"{a['title']} {a['summary']}" for a in articles]
        
        if self.sentiment_analyzer:
            sentiments = self.sentiment_analyzer.analyze_batch(texts)
        else:
            # Fallback to neutral
            sentiments = [self._get_neutral_sentiment()] * len(texts)
        
        # Aggregate sentiments
        return self._aggregate_sentiments(sentiments, len(articles))
    
    def _get_neutral_sentiment(self) -> Dict[str, float]:
        """Return neutral sentiment scores."""
        return {
            'positive': 0.33,
            'negative': 0.33,
            'neutral': 0.34,
            'compound': 0.0,
            'article_count': 0
        }
    
    def _aggregate_sentiments(
        self,
        sentiments: List[Dict[str, float]],
        article_count: int
    ) -> Dict[str, float]:
        """
        Aggregate multiple sentiment scores.
        
        Args:
            sentiments: List of sentiment dictionaries
            article_count: Number of articles
        
        Returns:
            Aggregated sentiment scores
        """
        if not sentiments:
            return self._get_neutral_sentiment()
        
        # Calculate weighted average (recent articles weighted more)
        weights = np.exp(np.linspace(-1, 0, len(sentiments)))
        weights = weights / weights.sum()
        
        aggregated = {
            'positive': sum(s['positive'] * w for s, w in zip(sentiments, weights)),
            'negative': sum(s['negative'] * w for s, w in zip(sentiments, weights)),
            'neutral': sum(s['neutral'] * w for s, w in zip(sentiments, weights)),
            'compound': sum(s['compound'] * w for s, w in zip(sentiments, weights)),
            'article_count': article_count,
            'sentiment_std': np.std([s['compound'] for s in sentiments]),
            'sentiment_max': max(s['compound'] for s in sentiments),
            'sentiment_min': min(s['compound'] for s in sentiments)
        }
        
        return aggregated
    
    def add_sentiment_to_dataframe(
        self,
        df: pd.DataFrame,
        ticker: str,
        use_cache: bool = True,
        cache_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Add sentiment features to stock data DataFrame.
        
        Args:
            df: Input DataFrame with date index
            ticker: Stock ticker
            use_cache: Whether to use cached sentiments
            cache_file: Path to cache file
        
        Returns:
            DataFrame with sentiment features added
        """
        logger.info(f"Adding sentiment features for {ticker}")
        
        # Try to load from cache
        if use_cache and cache_file and os.path.exists(cache_file):
            try:
                sentiment_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                logger.info(f"Loaded sentiment from cache: {cache_file}")
                return df.join(sentiment_df, how='left').fillna(0.33)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Generate sentiments for each date
        sentiment_data = []
        
        for date in df.index:
            sentiment = self.generate_daily_sentiment(ticker, date)
            sentiment['date'] = date
            sentiment_data.append(sentiment)
            
            # Rate limiting
            time.sleep(0.5)
        
        # Create sentiment DataFrame
        sentiment_df = pd.DataFrame(sentiment_data)
        sentiment_df.set_index('date', inplace=True)
        
        # Save to cache
        if use_cache and cache_file:
            sentiment_df.to_csv(cache_file)
            logger.info(f"Saved sentiment to cache: {cache_file}")
        
        # Join with original DataFrame
        result_df = df.join(sentiment_df, how='left')
        
        # Fill missing values
        result_df.fillna({
            'positive': 0.33,
            'negative': 0.33,
            'neutral': 0.34,
            'compound': 0.0,
            'article_count': 0,
            'sentiment_std': 0.0,
            'sentiment_max': 0.0,
            'sentiment_min': 0.0
        }, inplace=True)
        
        logger.info(f"Added {len(sentiment_df.columns)} sentiment features")
        
        return result_df


# Import at module level for optional dependency
try:
    import os
except ImportError:
    pass


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test sentiment analyzer
    analyzer = FinBERTSentimentAnalyzer()
    
    text = "Goldman Sachs reports strong Q4 earnings, beating analyst expectations"
    sentiment = analyzer.analyze_sentiment(text)
    print(f"Sentiment: {sentiment}")
    
    # Test news scraping
    scraper = YahooFinanceNewsScraper()
    articles = scraper.scrape_news("GS", max_articles=5)
    print(f"Scraped {len(articles)} articles")
    
    # Test feature generation
    generator = SentimentFeatureGenerator()
    daily_sentiment = generator.generate_daily_sentiment("GS", datetime.now())
    print(f"Daily sentiment: {daily_sentiment}")
