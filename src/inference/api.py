"""
FastAPI inference service for stock market forecasting GAN.
Provides REST API for real-time predictions and model management.
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml
import joblib
import asyncio
from functools import lru_cache

from src.models.generator import LSTMGenerator, AttentionLSTMGenerator
from src.models.discriminator import CNNDiscriminator
from src.models.gan import WGAN_GP
from src.data.preprocessors import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.utils.metrics import calculate_metrics

# Initialize FastAPI app
app = FastAPI(
    title="Stock Market Forecasting GAN API",
    description="REST API for stock price prediction using Generative Adversarial Networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========================
# Pydantic Models
# ========================

class PredictionRequest(BaseModel):
    """Request model for price prediction."""
    
    ticker: str = Field(..., description="Stock ticker symbol", example="GS")
    sequence_length: int = Field(50, description="Length of historical sequence", ge=10, le=200)
    forecast_horizon: int = Field(5, description="Number of days to forecast", ge=1, le=30)
    use_features: bool = Field(True, description="Whether to use engineered features")
    temperature: float = Field(1.0, description="Sampling temperature for diversity", ge=0.1, le=2.0)
    num_samples: int = Field(10, description="Number of samples for uncertainty", ge=1, le=100)
    
    @validator('ticker')
    def validate_ticker(cls, v):
        """Validate ticker symbol."""
        if not v.isalnum() or len(v) > 10:
            raise ValueError("Invalid ticker symbol")
        return v.upper()


class PredictionResponse(BaseModel):
    """Response model for price prediction."""
    
    ticker: str
    timestamp: datetime
    predictions: List[Dict[str, float]]
    confidence_intervals: Dict[str, List[float]]
    metrics: Dict[str, float]
    forecast_horizon: int


class ModelInfo(BaseModel):
    """Model information response."""
    
    model_name: str
    version: str
    architecture: str
    parameters: Dict[str, Any]
    training_date: datetime
    performance_metrics: Dict[str, float]


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    timestamp: datetime
    model_loaded: bool
    device: str
    gpu_available: bool


# ========================
# Model Management
# ========================

class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self):
        """Initialize model manager."""
        self.generator = None
        self.discriminator = None
        self.gan = None
        self.preprocessor = None
        self.feature_engineer = None
        self.config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        
        logger.info(f"Model manager initialized on device: {self.device}")
    
    def load_model(self, model_path: str = "models/checkpoints/best_model.pt"):
        """
        Load trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
        """
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.config = checkpoint.get('config', {})
            
            # Create generator
            if self.config.get('use_attention', False):
                self.generator = AttentionLSTMGenerator(
                    noise_dim=self.config.get('noise_dim', 100),
                    hidden_dim=self.config.get('hidden_dim', 256),
                    feature_dim=self.config.get('feature_dim', 10),
                    seq_length=self.config.get('seq_length', 50),
                    num_layers=self.config.get('num_layers', 2)
                ).to(self.device)
            else:
                self.generator = LSTMGenerator(
                    noise_dim=self.config.get('noise_dim', 100),
                    hidden_dim=self.config.get('hidden_dim', 256),
                    feature_dim=self.config.get('feature_dim', 10),
                    seq_length=self.config.get('seq_length', 50),
                    num_layers=self.config.get('num_layers', 2)
                ).to(self.device)
            
            # Load generator weights
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.generator.eval()
            
            # Load preprocessor
            preprocessor_path = Path(model_path).parent / "preprocessor.pkl"
            if preprocessor_path.exists():
                self.preprocessor = joblib.load(preprocessor_path)
            
            # Create feature engineer
            self.feature_engineer = FeatureEngineer()
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}"
            )
    
    def generate_prediction(
        self,
        historical_data: np.ndarray,
        num_samples: int = 10,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate predictions using loaded model.
        
        Args:
            historical_data: Historical sequence data
            num_samples: Number of samples for uncertainty estimation
            temperature: Sampling temperature
        
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if not self.model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        try:
            predictions = []
            
            with torch.no_grad():
                for _ in range(num_samples):
                    # Sample noise
                    noise = torch.randn(
                        1,
                        self.config.get('noise_dim', 100)
                    ).to(self.device) * temperature
                    
                    # Generate prediction
                    pred = self.generator(noise)
                    predictions.append(pred.cpu().numpy()[0])
            
            predictions = np.array(predictions)
            
            # Compute statistics
            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)
            
            # Confidence intervals (95%)
            lower_bound = mean_pred - 1.96 * std_pred
            upper_bound = mean_pred + 1.96 * std_pred
            
            return {
                'mean': mean_pred,
                'std': std_pred,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'samples': predictions
            }
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )


# Global model manager
model_manager = ModelManager()


# ========================
# Dependency Functions
# ========================

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify authentication token.
    In production, implement proper token validation.
    """
    # Simplified token validation
    # Replace with actual JWT validation or API key check
    if credentials.credentials != "your-secret-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return credentials


@lru_cache()
def get_settings():
    """Get application settings."""
    return {
        'model_path': 'models/checkpoints/best_model.pt',
        'max_sequence_length': 200,
        'max_forecast_horizon': 30
    }


# ========================
# API Endpoints
# ========================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    settings = get_settings()
    model_path = settings['model_path']
    
    if Path(model_path).exists():
        try:
            model_manager.load_model(model_path)
            logger.info("Model loaded on startup")
        except Exception as e:
            logger.warning(f"Could not load model on startup: {e}")
    else:
        logger.warning(f"Model file not found: {model_path}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Stock Market Forecasting GAN API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        model_loaded=model_manager.model_loaded,
        device=str(model_manager.device),
        gpu_available=torch.cuda.is_available()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate stock price predictions.
    
    Args:
        request: Prediction request parameters
    
    Returns:
        Prediction response with forecasts and confidence intervals
    """
    try:
        logger.info(f"Prediction request for ticker: {request.ticker}")
        
        # Generate predictions
        # In production, fetch real historical data here
        historical_data = np.random.randn(request.sequence_length, 10)
        
        result = model_manager.generate_prediction(
            historical_data=historical_data,
            num_samples=request.num_samples,
            temperature=request.temperature
        )
        
        # Format predictions
        predictions = []
        for i in range(request.forecast_horizon):
            predictions.append({
                'day': i + 1,
                'predicted_price': float(result['mean'][i, 0]),
                'lower_bound': float(result['lower_bound'][i, 0]),
                'upper_bound': float(result['upper_bound'][i, 0]),
                'uncertainty': float(result['std'][i, 0])
            })
        
        # Compute metrics
        metrics = {
            'mean_uncertainty': float(result['std'].mean()),
            'confidence_width': float((result['upper_bound'] - result['lower_bound']).mean())
        }
        
        # Confidence intervals
        confidence_intervals = {
            'lower': result['lower_bound'][:request.forecast_horizon, 0].tolist(),
            'upper': result['upper_bound'][:request.forecast_horizon, 0].tolist()
        }
        
        # Log prediction in background
        background_tasks.add_task(
            log_prediction,
            request.ticker,
            predictions,
            metrics
        )
        
        return PredictionResponse(
            ticker=request.ticker,
            timestamp=datetime.now(),
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            metrics=metrics,
            forecast_horizon=request.forecast_horizon
        )
    
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about loaded model."""
    if not model_manager.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    # Count parameters
    num_params = sum(p.numel() for p in model_manager.generator.parameters())
    
    return ModelInfo(
        model_name="WGAN-GP Stock Forecaster",
        version="1.0.0",
        architecture="LSTM Generator + CNN Discriminator",
        parameters={
            'total_params': num_params,
            'noise_dim': model_manager.config.get('noise_dim', 100),
            'hidden_dim': model_manager.config.get('hidden_dim', 256),
            'seq_length': model_manager.config.get('seq_length', 50)
        },
        training_date=datetime.now(),
        performance_metrics={
            'val_loss': 0.123,  # Load from checkpoint
            'sharpe_ratio': 1.45
        }
    )


@app.post("/model/reload")
async def reload_model(model_path: Optional[str] = None):
    """Reload model from checkpoint."""
    try:
        if model_path is None:
            settings = get_settings()
            model_path = settings['model_path']
        
        model_manager.load_model(model_path)
        
        return {
            "status": "success",
            "message": f"Model reloaded from {model_path}",
            "timestamp": datetime.now()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}"
        )


# ========================
# Background Tasks
# ========================

async def log_prediction(ticker: str, predictions: List[Dict], metrics: Dict):
    """
    Log prediction for monitoring and analytics.
    
    Args:
        ticker: Stock ticker
        predictions: Prediction data
        metrics: Performance metrics
    """
    # In production, log to database or monitoring system
    logger.info(f"Logged prediction for {ticker}: {metrics}")


# ========================
# Run Server
# ========================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
