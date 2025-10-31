"""
Alternative data sources for enhanced stock market forecasting.
Includes options flow, SEC filings, social media sentiment, and supply chain indicators.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class OptionsFlowAnalyzer:
    """
    Analyzes options flow data to detect unusual activity and sentiment.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize options flow analyzer.
        
        Args:
            api_key: API key for options data provider (optional)
        """
        self.api_key = api_key
        self.cache = {}
        
        logger.info("Initialized OptionsFlowAnalyzer")
    
    def fetch_options_flow(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch options flow data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
        
        Returns:
            DataFrame with options flow data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        cache_key = f"{ticker}_{start_date}_{end_date}"
        
        if cache_key in self.cache:
            logger.debug(f"Using cached options flow for {ticker}")
            return self.cache[cache_key]
        
        try:
            # Placeholder: Replace with actual API call
            logger.warning("Using simulated options flow data")
            
            # Simulate options data
            dates = pd.date_range(start_date, end_date, freq='D')
            n_samples = len(dates)
            
            data = {
                'date': dates,
                'call_volume': np.random.randint(1000, 50000, n_samples),
                'put_volume': np.random.randint(1000, 50000, n_samples),
                'call_oi': np.random.randint(10000, 200000, n_samples),
                'put_oi': np.random.randint(10000, 200000, n_samples),
                'call_premium': np.random.uniform(1e6, 50e6, n_samples),
                'put_premium': np.random.uniform(1e6, 50e6, n_samples),
                'iv_rank': np.random.uniform(10, 90, n_samples),
                'unusual_activity_count': np.random.poisson(2, n_samples)
            }
            
            df = pd.DataFrame(data)
            self.cache[cache_key] = df
            
            logger.info(f"Fetched options flow for {ticker}: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching options flow for {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_options_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate options-based sentiment metrics.
        
        Args:
            df: Options flow DataFrame
        
        Returns:
            DataFrame with calculated metrics
        """
        metrics = df.copy()
        
        # Put-Call Ratio (Volume)
        metrics['pcr_volume'] = df['put_volume'] / df['call_volume'].replace(0, 1)
        
        # Put-Call Ratio (Open Interest)
        metrics['pcr_oi'] = df['put_oi'] / df['call_oi'].replace(0, 1)
        
        # Premium Ratio
        metrics['premium_ratio'] = df['call_premium'] / df['put_premium'].replace(0, 1)
        
        # Net Bullish Flow (higher = more bullish)
        metrics['net_bullish_flow'] = (
            (df['call_volume'] - df['put_volume']) / 
            (df['call_volume'] + df['put_volume'])
        )
        
        # Smart Money Indicator (unusual activity)
        metrics['smart_money_score'] = (
            df['unusual_activity_count'] * metrics['net_bullish_flow']
        )
        
        # IV Rank signal (extreme values signal potential reversals)
        metrics['iv_extreme_signal'] = np.where(
            df['iv_rank'] > 80, -1,  # High IV -> potential reversal down
            np.where(df['iv_rank'] < 20, 1, 0)  # Low IV -> potential move up
        )
        
        # Rolling averages
        for col in ['pcr_volume', 'pcr_oi', 'net_bullish_flow']:
            metrics[f'{col}_ma5'] = metrics[col].rolling(5).mean()
            metrics[f'{col}_ma20'] = metrics[col].rolling(20).mean()
        
        logger.info(f"Calculated {len(metrics.columns) - len(df.columns)} new options metrics")
        
        return metrics
    
    def detect_unusual_activity(
        self,
        df: pd.DataFrame,
        volume_threshold: float = 2.0,
        premium_threshold: float = 1e6
    ) -> List[Dict]:
        """
        Detect unusual options activity.
        
        Args:
            df: Options flow DataFrame
            volume_threshold: Multiplier of average volume
            premium_threshold: Minimum premium for unusual activity
        
        Returns:
            List of unusual activity events
        """
        unusual_events = []
        
        # Calculate volume moving averages
        df['call_volume_ma'] = df['call_volume'].rolling(20).mean()
        df['put_volume_ma'] = df['put_volume'].rolling(20).mean()
        
        for idx, row in df.iterrows():
            # Check for unusual call activity
            if (row['call_volume'] > row['call_volume_ma'] * volume_threshold and
                row['call_premium'] > premium_threshold):
                unusual_events.append({
                    'date': row['date'],
                    'type': 'CALL_SWEEP',
                    'volume': row['call_volume'],
                    'premium': row['call_premium'],
                    'sentiment': 'BULLISH'
                })
            
            # Check for unusual put activity
            if (row['put_volume'] > row['put_volume_ma'] * volume_threshold and
                row['put_premium'] > premium_threshold):
                unusual_events.append({
                    'date': row['date'],
                    'type': 'PUT_SWEEP',
                    'volume': row['put_volume'],
                    'premium': row['put_premium'],
                    'sentiment': 'BEARISH'
                })
        
        logger.info(f"Detected {len(unusual_events)} unusual activity events")
        return unusual_events


class SECFilingsParser:
    """
    Parses SEC filings (10-K, 10-Q, 8-K) for fundamental insights.
    """
    
    def __init__(self, user_agent: str = "research-bot@example.com"):
        """
        Initialize SEC filings parser.
        
        Args:
            user_agent: User agent for SEC EDGAR requests
        """
        self.base_url = "https://www.sec.gov"
        self.headers = {'User-Agent': user_agent}
        self.cache = {}
        
        logger.info("Initialized SECFilingsParser")
    
    def get_company_filings(
        self,
        ticker: str,
        filing_types: List[str] = ['10-K', '10-Q'],
        count: int = 5
    ) -> List[Dict]:
        """
        Get recent filings for a company.
        
        Args:
            ticker: Stock ticker
            filing_types: Types of filings to fetch
            count: Number of filings to retrieve
        
        Returns:
            List of filing metadata
        """
        try:
            # Get CIK from ticker (simplified - real implementation needs mapping)
            logger.warning(f"Using placeholder CIK lookup for {ticker}")
            cik = "0000320193"  # Example: Apple's CIK
            
            # Construct search URL
            search_url = f"{self.base_url}/cgi-bin/browse-edgar"
            params = {
                'action': 'getcompany',
                'CIK': cik,
                'type': ','.join(filing_types),
                'count': count,
                'output': 'atom'
            }
            
            # Placeholder response
            filings = []
            for i in range(count):
                filings.append({
                    'filing_type': np.random.choice(filing_types),
                    'filing_date': datetime.now() - timedelta(days=90*i),
                    'url': f"{self.base_url}/example_filing_{i}.htm",
                    'ticker': ticker
                })
            
            logger.info(f"Retrieved {len(filings)} filings for {ticker}")
            return filings
            
        except Exception as e:
            logger.error(f"Error fetching filings for {ticker}: {e}")
            return []
    
    def parse_filing_sentiment(self, filing_url: str) -> Dict:
        """
        Parse filing text and extract sentiment signals.
        
        Args:
            filing_url: URL to filing document
        
        Returns:
            Dictionary with sentiment metrics
        """
        try:
            # Placeholder: Real implementation would fetch and parse HTML
            logger.warning("Using simulated filing sentiment")
            
            # Simulate sentiment analysis
            sentiment = {
                'risk_mentions': np.random.randint(10, 50),
                'opportunity_mentions': np.random.randint(5, 30),
                'litigation_mentions': np.random.randint(0, 10),
                'positive_tone_score': np.random.uniform(0.3, 0.7),
                'uncertainty_score': np.random.uniform(0.1, 0.5),
                'complex_language_ratio': np.random.uniform(0.2, 0.4),
                'forward_looking_statements': np.random.randint(5, 25)
            }
            
            # Calculate composite sentiment
            sentiment['composite_sentiment'] = (
                sentiment['positive_tone_score'] * 0.4 +
                (1 - sentiment['uncertainty_score']) * 0.3 +
                (sentiment['opportunity_mentions'] / 
                 (sentiment['risk_mentions'] + 1)) * 0.3
            )
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error parsing filing sentiment: {e}")
            return {}
    
    def extract_financial_metrics(self, filing_url: str) -> Dict:
        """
        Extract key financial metrics from filing.
        
        Args:
            filing_url: URL to filing document
        
        Returns:
            Dictionary with financial metrics
        """
        # Placeholder implementation
        metrics = {
            'revenue': np.random.uniform(50e9, 100e9),
            'net_income': np.random.uniform(5e9, 20e9),
            'total_assets': np.random.uniform(100e9, 500e9),
            'total_liabilities': np.random.uniform(50e9, 300e9),
            'cash_and_equivalents': np.random.uniform(10e9, 100e9),
            'revenue_growth_yoy': np.random.uniform(-0.1, 0.3),
            'margin': np.random.uniform(0.1, 0.4)
        }
        
        return metrics


class SocialMediaSentimentScraper:
    """
    Scrapes social media for stock sentiment (Reddit, Twitter).
    """
    
    def __init__(self, reddit_client_id: Optional[str] = None):
        """
        Initialize social media scraper.
        
        Args:
            reddit_client_id: Reddit API client ID (optional)
        """
        self.reddit_client_id = reddit_client_id
        self.cache = {}
        
        logger.info("Initialized SocialMediaSentimentScraper")
    
    def scrape_reddit_wallstreetbets(
        self,
        ticker: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Scrape r/WallStreetBets for ticker mentions.
        
        Args:
            ticker: Stock ticker
            limit: Maximum posts to scrape
        
        Returns:
            DataFrame with post data
        """
        try:
            # Placeholder: Real implementation uses PRAW
            logger.warning("Using simulated Reddit data")
            
            posts = []
            for i in range(limit):
                posts.append({
                    'timestamp': datetime.now() - timedelta(hours=i),
                    'title': f"Sample post about ${ticker}",
                    'score': np.random.randint(-50, 500),
                    'num_comments': np.random.randint(0, 200),
                    'awards': np.random.randint(0, 20),
                    'sentiment': np.random.choice(['BULLISH', 'BEARISH', 'NEUTRAL']),
                    'ticker': ticker
                })
            
            df = pd.DataFrame(posts)
            logger.info(f"Scraped {len(df)} Reddit posts for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping Reddit for {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_reddit_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate aggregated Reddit sentiment metrics.
        
        Args:
            df: DataFrame with Reddit post data
        
        Returns:
            Dictionary with metrics
        """
        if df.empty:
            return {}
        
        metrics = {
            'total_mentions': len(df),
            'avg_score': df['score'].mean(),
            'total_comments': df['num_comments'].sum(),
            'total_awards': df['awards'].sum(),
            'bullish_ratio': (df['sentiment'] == 'BULLISH').sum() / len(df),
            'bearish_ratio': (df['sentiment'] == 'BEARISH').sum() / len(df),
            'engagement_score': (df['score'] + df['num_comments'] * 2 + df['awards'] * 5).sum(),
            'viral_posts_count': (df['score'] > df['score'].quantile(0.9)).sum()
        }
        
        # Time-based metrics
        df['hour'] = df['timestamp'].dt.hour
        metrics['peak_mention_hour'] = df.groupby('hour').size().idxmax()
        
        return metrics
    
    def scrape_twitter_sentiment(
        self,
        ticker: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Scrape Twitter for ticker sentiment.
        
        Args:
            ticker: Stock ticker
            limit: Maximum tweets to scrape
        
        Returns:
            DataFrame with tweet data
        """
        # Placeholder implementation
        logger.warning("Using simulated Twitter data")
        
        tweets = []
        for i in range(limit):
            tweets.append({
                'timestamp': datetime.now() - timedelta(minutes=i*10),
                'text': f"Sample tweet about ${ticker}",
                'likes': np.random.randint(0, 1000),
                'retweets': np.random.randint(0, 500),
                'sentiment_score': np.random.uniform(-1, 1),
                'ticker': ticker
            })
        
        return pd.DataFrame(tweets)


class SupplyChainIndicators:
    """
    Tracks supply chain indicators and economic factors.
    """
    
    def __init__(self):
        """Initialize supply chain indicators tracker."""
        self.cache = {}
        logger.info("Initialized SupplyChainIndicators")
    
    def fetch_shipping_rates(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch global shipping rate data.
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with shipping rates
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=180)
        if end_date is None:
            end_date = datetime.now()
        
        # Placeholder data
        dates = pd.date_range(start_date, end_date, freq='W')
        
        data = {
            'date': dates,
            'container_index': np.random.uniform(1000, 5000, len(dates)),
            'baltic_dry_index': np.random.uniform(1000, 3000, len(dates)),
            'port_congestion_score': np.random.uniform(0, 100, len(dates))
        }
        
        return pd.DataFrame(data)
    
    def fetch_commodity_prices(
        self,
        commodities: List[str] = ['oil', 'copper', 'steel'],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch commodity price data.
        
        Args:
            commodities: List of commodities to track
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with commodity prices
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=180)
        if end_date is None:
            end_date = datetime.now()
        
        dates = pd.date_range(start_date, end_date, freq='D')
        
        data = {'date': dates}
        for commodity in commodities:
            data[f'{commodity}_price'] = np.random.uniform(50, 150, len(dates))
        
        return pd.DataFrame(data)


class AlternativeDataFusion:
    """
    Combines multiple alternative data sources into unified features.
    """
    
    def __init__(self):
        """Initialize alternative data fusion."""
        self.options_analyzer = OptionsFlowAnalyzer()
        self.sec_parser = SECFilingsParser()
        self.social_scraper = SocialMediaSentimentScraper()
        self.supply_chain = SupplyChainIndicators()
        
        logger.info("Initialized AlternativeDataFusion")
    
    def create_unified_features(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Create unified feature set from all alternative data sources.
        
        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with unified features
        """
        logger.info(f"Creating unified alternative data features for {ticker}")
        
        # Fetch all data sources
        options_df = self.options_analyzer.fetch_options_flow(ticker, start_date, end_date)
        options_metrics = self.options_analyzer.calculate_options_metrics(options_df)
        
        reddit_df = self.social_scraper.scrape_reddit_wallstreetbets(ticker)
        reddit_metrics = self.social_scraper.calculate_reddit_metrics(reddit_df)
        
        # Create feature DataFrame
        features = pd.DataFrame()
        
        # Options features
        if not options_metrics.empty:
            for col in ['pcr_volume', 'pcr_oi', 'net_bullish_flow', 'iv_rank', 'smart_money_score']:
                if col in options_metrics.columns:
                    features[f'options_{col}'] = options_metrics[col].values
        
        # Social sentiment features
        if reddit_metrics:
            for key, value in reddit_metrics.items():
                if isinstance(value, (int, float)):
                    features[f'reddit_{key}'] = value
        
        logger.info(f"Created {len(features.columns)} unified alternative data features")
        
        return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test options flow
    options_analyzer = OptionsFlowAnalyzer()
    options_df = options_analyzer.fetch_options_flow('AAPL')
    options_metrics = options_analyzer.calculate_options_metrics(options_df)
    print(f"\nOptions metrics shape: {options_metrics.shape}")
    print(f"Columns: {options_metrics.columns.tolist()[:5]}...")
    
    unusual = options_analyzer.detect_unusual_activity(options_df)
    print(f"Detected {len(unusual)} unusual activity events")
    
    # Test SEC parser
    sec_parser = SECFilingsParser()
    filings = sec_parser.get_company_filings('AAPL')
    print(f"\nRetrieved {len(filings)} SEC filings")
    
    # Test social media
    social_scraper = SocialMediaSentimentScraper()
    reddit_df = social_scraper.scrape_reddit_wallstreetbets('GME')
    reddit_metrics = social_scraper.calculate_reddit_metrics(reddit_df)
    print(f"\nReddit metrics: {list(reddit_metrics.keys())}")
    
    # Test fusion
    fusion = AlternativeDataFusion()
    unified_features = fusion.create_unified_features('AAPL')
    print(f"\nUnified features shape: {unified_features.shape}")
    print(f"Features: {unified_features.columns.tolist()}")
