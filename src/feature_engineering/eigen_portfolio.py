"""
Eigen Portfolio Analysis using Principal Component Analysis (PCA).
Reduces dimensionality of correlated assets and extracts systematic risk factors.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class EigenPortfolioAnalyzer:
    """
    Performs eigen portfolio analysis on correlated assets.
    Extracts principal components representing systematic risk factors.
    """
    
    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95
    ):
        """
        Initialize eigen portfolio analyzer.
        
        Args:
            n_components: Number of principal components (None for auto)
            variance_threshold: Cumulative variance to explain (if n_components is None)
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.explained_variance = None
        self.components_df = None
        
        logger.info(
            f"Initialized EigenPortfolioAnalyzer: "
            f"n_components={n_components}, variance_threshold={variance_threshold}"
        )
    
    def fit(
        self,
        returns_df: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> 'EigenPortfolioAnalyzer':
        """
        Fit PCA on asset returns.
        
        Args:
            returns_df: DataFrame of asset returns [samples, assets]
            feature_names: Names of assets (uses columns if None)
        
        Returns:
            Self for chaining
        """
        if feature_names is None:
            feature_names = returns_df.columns.tolist()
        
        self.feature_names = feature_names
        
        # Standardize returns
        returns_scaled = self.scaler.fit_transform(returns_df)
        
        # Determine number of components
        if self.n_components is None:
            # Fit with all components first
            pca_full = PCA()
            pca_full.fit(returns_scaled)
            
            # Find number needed for variance threshold
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            
            logger.info(
                f"Auto-selected {self.n_components} components "
                f"to explain {self.variance_threshold*100:.1f}% variance"
            )
        
        # Fit PCA with determined components
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(returns_scaled)
        
        # Store explained variance
        self.explained_variance = self.pca.explained_variance_ratio_
        
        # Create components DataFrame (loadings)
        self.components_df = pd.DataFrame(
            self.pca.components_.T,
            index=self.feature_names,
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )
        
        logger.info(
            f"PCA fitted: {self.n_components} components, "
            f"total variance explained: {self.explained_variance.sum()*100:.2f}%"
        )
        
        return self
    
    def transform(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        Transform returns to eigen portfolio space.
        
        Args:
            returns_df: DataFrame of asset returns
        
        Returns:
            Transformed data [samples, n_components]
        """
        if self.pca is None:
            raise ValueError("Must call fit() before transform()")
        
        returns_scaled = self.scaler.transform(returns_df)
        return self.pca.transform(returns_scaled)
    
    def fit_transform(
        self,
        returns_df: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            returns_df: DataFrame of asset returns
            feature_names: Names of assets
        
        Returns:
            Transformed data
        """
        self.fit(returns_df, feature_names)
        return self.transform(returns_df)
    
    def inverse_transform(self, eigen_portfolios: np.ndarray) -> np.ndarray:
        """
        Transform back from eigen portfolio space to asset space.
        
        Args:
            eigen_portfolios: Eigen portfolio values [samples, n_components]
        
        Returns:
            Reconstructed returns [samples, n_assets]
        """
        if self.pca is None:
            raise ValueError("Must call fit() before inverse_transform()")
        
        returns_scaled = self.pca.inverse_transform(eigen_portfolios)
        return self.scaler.inverse_transform(returns_scaled)
    
    def get_portfolio_weights(self, pc_index: int = 0) -> pd.Series:
        """
        Get asset weights for a specific principal component.
        
        Args:
            pc_index: Index of principal component (0-based)
        
        Returns:
            Series of asset weights
        """
        if self.components_df is None:
            raise ValueError("Must call fit() first")
        
        weights = self.components_df.iloc[:, pc_index]
        
        # Normalize to sum to 1 (absolute values)
        weights_normalized = weights / weights.abs().sum()
        
        return weights_normalized
    
    def interpret_components(self, top_n: int = 5) -> Dict[str, Dict]:
        """
        Interpret principal components by top contributing assets.
        
        Args:
            top_n: Number of top assets to show per component
        
        Returns:
            Dictionary with interpretation for each component
        """
        if self.components_df is None:
            raise ValueError("Must call fit() first")
        
        interpretations = {}
        
        for i, col in enumerate(self.components_df.columns):
            loadings = self.components_df[col].abs().sort_values(ascending=False)
            top_assets = loadings.head(top_n)
            
            interpretation = {
                'variance_explained': f"{self.explained_variance[i]*100:.2f}%",
                'cumulative_variance': f"{self.explained_variance[:i+1].sum()*100:.2f}%",
                'top_contributors': top_assets.to_dict(),
                'interpretation': self._interpret_factor(self.components_df[col], top_assets.index.tolist())
            }
            
            interpretations[col] = interpretation
        
        return interpretations
    
    def _interpret_factor(self, loadings: pd.Series, top_assets: List[str]) -> str:
        """
        Attempt to interpret what a factor represents.
        
        Args:
            loadings: Component loadings
            top_assets: Top contributing assets
        
        Returns:
            Human-readable interpretation
        """
        # Simple heuristics based on asset names
        asset_types = {
            'bank': ['JPM', 'BAC', 'C', 'WFC', 'MS', 'GS'],
            'index': ['SPY', 'DIA', 'QQQ', '^GSPC', '^DJI', '^IXIC'],
            'volatility': ['VIX', '^VIX'],
            'currency': ['EUR', 'GBP', 'JPY', 'EURUSD', 'GBPUSD'],
            'commodity': ['GLD', 'USO', 'GOLD', 'OIL']
        }
        
        # Count asset types in top contributors
        type_counts = {asset_type: 0 for asset_type in asset_types}
        
        for asset in top_assets:
            for asset_type, tickers in asset_types.items():
                if any(ticker in asset for ticker in tickers):
                    type_counts[asset_type] += 1
        
        # Determine dominant type
        dominant_type = max(type_counts, key=type_counts.get)
        
        if type_counts[dominant_type] >= 2:
            interpretations = {
                'bank': 'Financial Sector Factor',
                'index': 'Market Factor',
                'volatility': 'Volatility Factor',
                'currency': 'Currency/FX Factor',
                'commodity': 'Commodity Factor'
            }
            return interpretations.get(dominant_type, 'Mixed Factor')
        
        return 'Mixed/Idiosyncratic Factor'
    
    def plot_variance_explained(self, save_path: Optional[str] = None):
        """
        Plot explained variance by component.
        
        Args:
            save_path: Path to save plot (optional)
        """
        if self.explained_variance is None:
            raise ValueError("Must call fit() first")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        components = range(1, len(self.explained_variance) + 1)
        ax1.bar(components, self.explained_variance * 100)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained (%)')
        ax1.set_title('Variance Explained by Each Component')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative variance
        cumulative_variance = np.cumsum(self.explained_variance) * 100
        ax2.plot(components, cumulative_variance, marker='o', linewidth=2, markersize=8)
        ax2.axhline(y=95, color='r', linestyle='--', label='95% threshold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Variance Explained (%)')
        ax2.set_title('Cumulative Variance Explained')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved variance plot to {save_path}")
        
        plt.show()
    
    def plot_loadings_heatmap(
        self,
        n_components: Optional[int] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot heatmap of component loadings.
        
        Args:
            n_components: Number of components to show (None for all)
            save_path: Path to save plot
        """
        if self.components_df is None:
            raise ValueError("Must call fit() first")
        
        if n_components is None:
            n_components = self.n_components
        
        components_to_plot = self.components_df.iloc[:, :n_components]
        
        plt.figure(figsize=(12, max(8, len(self.feature_names) * 0.3)))
        sns.heatmap(
            components_to_plot,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Loading'},
            linewidths=0.5
        )
        plt.title(f'PCA Component Loadings (Top {n_components} Components)')
        plt.xlabel('Principal Component')
        plt.ylabel('Asset')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved loadings heatmap to {save_path}")
        
        plt.show()
    
    def create_eigen_portfolio_features(
        self,
        returns_df: pd.DataFrame,
        prefix: str = 'EP'
    ) -> pd.DataFrame:
        """
        Create eigen portfolio features from returns data.
        
        Args:
            returns_df: DataFrame of asset returns
            prefix: Prefix for feature names
        
        Returns:
            DataFrame with eigen portfolio features
        """
        eigen_portfolios = self.transform(returns_df)
        
        feature_names = [f'{prefix}{i+1}' for i in range(self.n_components)]
        
        ep_df = pd.DataFrame(
            eigen_portfolios,
            index=returns_df.index,
            columns=feature_names
        )
        
        # Add rolling statistics
        for col in feature_names:
            ep_df[f'{col}_MA5'] = ep_df[col].rolling(5).mean()
            ep_df[f'{col}_MA20'] = ep_df[col].rolling(20).mean()
            ep_df[f'{col}_Vol10'] = ep_df[col].rolling(10).std()
        
        logger.info(f"Created {len(ep_df.columns)} eigen portfolio features")
        
        return ep_df
    
    def save(self, filepath: str):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        import joblib
        
        model_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'n_components': self.n_components,
            'variance_threshold': self.variance_threshold,
            'feature_names': self.feature_names,
            'explained_variance': self.explained_variance,
            'components_df': self.components_df
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Saved eigen portfolio model to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EigenPortfolioAnalyzer':
        """
        Load model from disk.
        
        Args:
            filepath: Path to load model from
        
        Returns:
            Loaded EigenPortfolioAnalyzer instance
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        analyzer = cls(
            n_components=model_data['n_components'],
            variance_threshold=model_data['variance_threshold']
        )
        
        analyzer.pca = model_data['pca']
        analyzer.scaler = model_data['scaler']
        analyzer.feature_names = model_data['feature_names']
        analyzer.explained_variance = model_data['explained_variance']
        analyzer.components_df = model_data['components_df']
        
        logger.info(f"Loaded eigen portfolio model from {filepath}")
        
        return analyzer


def calculate_portfolio_returns(
    weights: pd.Series,
    returns_df: pd.DataFrame
) -> pd.Series:
    """
    Calculate portfolio returns given weights.
    
    Args:
        weights: Asset weights (normalized)
        returns_df: Asset returns DataFrame
    
    Returns:
        Portfolio returns series
    """
    # Align weights and returns
    common_assets = weights.index.intersection(returns_df.columns)
    weights_aligned = weights[common_assets]
    returns_aligned = returns_df[common_assets]
    
    # Normalize weights
    weights_normalized = weights_aligned / weights_aligned.sum()
    
    # Calculate portfolio returns
    portfolio_returns = (returns_aligned * weights_normalized).sum(axis=1)
    
    return portfolio_returns


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_assets = 10
    
    # Simulate correlated returns
    cov_matrix = np.random.rand(n_assets, n_assets)
    cov_matrix = cov_matrix @ cov_matrix.T  # Make positive semi-definite
    
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=cov_matrix * 0.01,
        size=n_samples
    )
    
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    returns_df = pd.DataFrame(returns, columns=asset_names)
    
    # Perform eigen portfolio analysis
    analyzer = EigenPortfolioAnalyzer(variance_threshold=0.95)
    eigen_portfolios = analyzer.fit_transform(returns_df)
    
    print(f"\nNumber of components: {analyzer.n_components}")
    print(f"Total variance explained: {analyzer.explained_variance.sum()*100:.2f}%")
    
    # Interpret components
    interpretations = analyzer.interpret_components(top_n=3)
    print("\nComponent Interpretations:")
    for pc, info in interpretations.items():
        print(f"\n{pc}:")
        print(f"  Variance: {info['variance_explained']}")
        print(f"  Top contributors: {list(info['top_contributors'].keys())}")
    
    # Create features
    ep_features = analyzer.create_eigen_portfolio_features(returns_df)
    print(f"\nCreated {len(ep_features.columns)} eigen portfolio features")
    print(f"Features: {ep_features.columns.tolist()[:5]}...")
    
    # Plot (commented out for script)
    # analyzer.plot_variance_explained()
    # analyzer.plot_loadings_heatmap(n_components=5)
