# 📈 Stock Market Forecasting with GAN

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org/)

> **A production-ready, state-of-the-art GAN-based system for stock market forecasting using deep learning, featuring LSTM generators, CNN discriminators, and comprehensive MLOps integration.**

---

## 🌟 Key Features

### **Multi-Technology Stack**
- **PyTorch** - Deep learning framework with CUDA acceleration
- **MLflow** - Experiment tracking and model registry
- **FastAPI** - Production-ready REST API
- **Optuna** - Bayesian hyperparameter optimization
- **PostgreSQL/TimescaleDB** - Time-series data storage

### **Advanced Architecture**
- **Wasserstein GAN with Gradient Penalty (WGAN-GP)** - Stable training with Lipschitz constraint
- **LSTM Generator** - Time-series generation with attention mechanism
- **CNN Discriminator** - 1D convolutional network for sequence classification
- **Multi-source Feature Engineering** - Technical indicators, Fourier transforms, sentiment analysis

### **Production-Ready**
- ✅ Modular, extensible codebase with proper abstractions
- ✅ Comprehensive configuration management (YAML)
- ✅ Structured logging and monitoring
- ✅ Unit tests and integration tests
- ✅ Docker containerization
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Detailed inline documentation

---

## 📊 Project Overview

This project implements a Generative Adversarial Network (GAN) for predicting stock market movements. Unlike traditional time-series forecasting methods, GANs can capture complex, non-linear patterns in financial data by learning the underlying distribution of stock price movements.

### **Why GAN for Stock Market Forecasting?**

1. **Distribution Learning**: GANs learn the actual distribution of price movements, not just point predictions
2. **Adversarial Training**: The discriminator acts as a sophisticated validator, ensuring realistic predictions
3. **Feature Extraction**: Multi-layered networks automatically discover relevant patterns
4. **Handling Non-Stationarity**: GANs can adapt to changing market regimes

### **Architecture Diagram**

```
┌─────────────────┐
│  Data Sources   │
│  • OHLCV Data   │
│  • Indices      │
│  • Currencies   │
│  • Sentiment    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│ • Cleaning      │
│ • Normalization │
│ • Alignment     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Eng.    │
│ • Technical     │
│ • Fourier       │
│ • ARIMA         │
│ • Lags/Rolling  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│   Generator     │◄─────┤  Random Noise   │
│    (LSTM)       │      └─────────────────┘
└────────┬────────┘
         │
         │ Generated Sequences
         ▼
┌─────────────────┐
│ Discriminator   │
│     (CNN)       │
└────────┬────────┘
         │
         │ Real/Fake Score
         ▼
┌─────────────────┐
│ Wasserstein     │
│ Loss + GP       │
└─────────────────┘
```

---

## 🚀 Getting Started

### **Prerequisites**

- Python 3.9+
- CUDA-capable GPU (GTX 3050Ti or better recommended)
- 8GB+ RAM
- Git

### **Installation**

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Stock-Market-Forecasting-with-GAN.git
cd Stock-Market-Forecasting-with-GAN
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup CUDA (for GPU training)**
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 💻 Usage

### **1. Data Collection**

```python
from src.data.collectors import DataCollector

collector = DataCollector(
    ticker="GS",
    start_date="2010-01-01",
    end_date="2023-12-31"
)

primary_data, correlated_data = collector.fetch_all_data()
```

### **2. Train the Model**

```bash
python scripts/train.py \
    --model-config configs/model_config.yaml \
    --training-config configs/training_config.yaml \
    --data-config configs/data_config.yaml
```

### **3. Monitor Training with MLflow**

```bash
mlflow ui
# Open browser at http://localhost:5000
```

### **4. Hyperparameter Optimization**

```bash
python scripts/optimize.py --n-trials 100
```

### **5. Make Predictions**

```bash
python scripts/predict.py \
    --model-path models/final/final_model.pt \
    --input-data data/test.csv \
    --output predictions.csv
```

---

## 📁 Project Structure

```
Stock-Market-Forecasting-with-GAN/
│
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   │   ├── collectors.py         # Data collection from APIs
│   │   ├── preprocessors.py      # Data cleaning & normalization
│   │   ├── feature_engineering.py # Technical indicators & features
│   │   └── datasets.py           # PyTorch Dataset classes
│   │
│   ├── models/                   # Model architectures
│   │   ├── generator.py          # LSTM Generator
│   │   ├── discriminator.py      # CNN Discriminator
│   │   ├── gan.py                # WGAN-GP implementation
│   │   └── components.py         # Shared components
│   │
│   ├── training/                 # Training utilities
│   │   ├── trainer.py            # Training loop
│   │   ├── losses.py             # Custom loss functions
│   │   └── callbacks.py          # Training callbacks
│   │
│   ├── optimization/             # Hyperparameter optimization
│   │   ├── hyperparameter_search.py
│   │   └── schedulers.py
│   │
│   ├── inference/                # Prediction & API
│   │   ├── predictor.py          # Prediction logic
│   │   └── api.py                # FastAPI endpoints
│   │
│   └── utils/                    # Utilities
│       ├── logger.py             # Logging setup
│       ├── metrics.py            # Evaluation metrics
│       └── visualization.py      # Plotting functions
│
├── configs/                      # Configuration files
│   ├── model_config.yaml         # Model architecture
│   ├── training_config.yaml      # Training parameters
│   ├── data_config.yaml          # Data sources
│   └── inference_config.yaml     # API configuration
│
├── data/                         # Data directory
│   ├── raw/                      # Raw downloaded data
│   ├── processed/                # Processed features
│   └── predictions/              # Model outputs
│
├── models/                       # Saved models
│   ├── checkpoints/              # Training checkpoints
│   └── final/                    # Production models
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_eda.ipynb              # Exploratory analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_results_analysis.ipynb
│
├── scripts/                      # Executable scripts
│   ├── train.py                  # Main training script
│   ├── optimize.py               # Hyperparameter optimization
│   ├── predict.py                # Run predictions
│   └── download_data.py          # Data download script
│
├── tests/                        # Unit tests
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
│
├── deployment/                   # Deployment files
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
│
├── .github/                      # CI/CD
│   └── workflows/
│       └── ci.yml                # GitHub Actions
│
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── LICENSE                       # MIT License
```

---

## 🔧 Configuration

All configurations are in YAML format for easy modification:

### **Model Configuration** (`configs/model_config.yaml`)
- Generator architecture (LSTM layers, hidden dimensions)
- Discriminator architecture (CNN layers, filters)
- WGAN-GP parameters (lambda_gp, n_critic)

### **Training Configuration** (`configs/training_config.yaml`)
- Training parameters (epochs, batch_size, learning_rate)
- Data split ratios
- Checkpoint settings
- Early stopping parameters

### **Data Configuration** (`configs/data_config.yaml`)
- Stock ticker and date range
- Correlated assets to include
- Feature engineering options
- Preprocessing methods

---

## 📊 Features Engineering

### **Technical Indicators**
- Moving Averages (MA7, MA21, MA50)
- Exponential Moving Averages (EMA12, EMA26)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- ATR (Average True Range)
- OBV (On-Balance Volume)
- Momentum indicators

### **Advanced Features**
- **Fourier Transforms**: Extract trend components (3, 6, 9, 50 components)
- **ARIMA Predictions**: Use traditional forecasting as features
- **Lag Features**: Historical values (1, 2, 3, 5, 10 days)
- **Rolling Statistics**: Mean, std, min, max over various windows

### **Correlated Assets**
- Financial institutions (JPM, MS, BAC, C, WFC)
- Market indices (S&P 500, Dow Jones, NASDAQ, FTSE, Nikkei)
- Volatility (VIX)
- Currencies (EUR/USD, GBP/USD, JPY/USD)
- Commodities (Gold, Oil)

---

## 🧠 Model Architecture

### **Generator (LSTM)**
```python
LSTMGenerator(
    noise_dim=100,       # Random noise input
    hidden_dim=256,      # LSTM hidden units
    num_layers=2,        # Number of LSTM layers
    n_features=112,      # Number of output features
    sequence_length=20,  # Length of generated sequences
    dropout=0.2          # Dropout rate
)
```

### **Discriminator (CNN)**
```python
CNNDiscriminator(
    n_features=112,          # Input features
    sequence_length=20,      # Sequence length
    base_filters=64,         # Initial conv filters
    num_conv_layers=3,       # Number of conv layers
    fc_hidden_dim=128,       # FC layer dimension
    dropout=0.3              # Dropout rate
)
```

### **Loss Function**
- **Wasserstein Loss**: More stable than vanilla GAN
- **Gradient Penalty**: Enforces Lipschitz constraint
- **Formula**: `L_D = E[D(fake)] - E[D(real)] + λ_GP * GP`

---

## 📈 Training Process

1. **Data Collection**: Fetch historical stock data and correlated assets
2. **Preprocessing**: Clean, normalize, and align data
3. **Feature Engineering**: Create 100+ features from raw data
4. **Model Training**: Train WGAN-GP with:
   - 5 discriminator updates per generator update
   - Gradient penalty for stability
   - Learning rate scheduling
   - Mixed precision training (faster on GPU)
5. **Validation**: Monitor Wasserstein distance and loss metrics
6. **Checkpointing**: Save models every N epochs
7. **MLflow Tracking**: Log all metrics and hyperparameters

---

## 🎯 Evaluation Metrics

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **Mean Absolute Percentage Error (MAPE)**
- **R² Score**
- **Directional Accuracy** - % of correct up/down predictions
- **Sharpe Ratio** - Risk-adjusted returns
- **Maximum Drawdown**
- **Wasserstein Distance** - Quality of generated samples

---

## 🚢 Deployment

### **Docker Deployment**

```bash
# Build image
docker build -t stock-gan:latest -f deployment/Dockerfile .

# Run container
docker run -p 8000:8000 --gpus all stock-gan:latest
```

### **FastAPI Service**

```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

Access API documentation at `http://localhost:8000/docs`

---

## 🔬 Hyperparameter Optimization

The project uses Optuna for Bayesian hyperparameter optimization:

```python
# Optimizable parameters:
- learning_rate (1e-5 to 1e-3)
- batch_size (32, 64, 128)
- hidden_dim (128, 256, 512)
- num_layers (1, 2, 3, 4)
- dropout (0.1 to 0.5)
- lambda_gp (1 to 20)
```

Run optimization:
```bash
python scripts/optimize.py --n-trials 100 --study-name stock_gan_optimization
```

---

## 📝 TODO / Future Enhancements

- [ ] Add sentiment analysis from news APIs (FinBERT)
- [ ] Implement attention mechanism in generator
- [ ] Add transformer-based discriminator
- [ ] Multi-asset prediction (portfolio optimization)
- [ ] Reinforcement learning for trading strategies
- [ ] Real-time inference pipeline
- [ ] Kubernetes deployment manifests
- [ ] A/B testing framework
- [ ] Model interpretability (SHAP values)
- [ ] Streaming data ingestion (Kafka)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guide
- All tests pass
- Documentation is updated
- Commit messages are clear

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**THIS SOFTWARE IS FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.**

- This is NOT financial advice
- Past performance does not guarantee future results
- Trading stocks carries substantial risk
- The authors are not responsible for any financial losses
- Always consult with a licensed financial advisor
- Use at your own risk

---

## 🙏 Acknowledgments

- Inspired by research in GAN-based time-series forecasting
- Built with PyTorch, MLflow, and FastAPI
- Special thanks to the open-source community

---

## 📧 Contact

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 📚 References

1. Goodfellow, I., et al. (2014). "Generative Adversarial Networks"
2. Arjovsky, M., et al. (2017). "Wasserstein GAN"
3. Gulrajani, I., et al. (2017). "Improved Training of Wasserstein GANs"
4. [Add relevant papers and resources]

---

**⭐ If you find this project useful, please consider giving it a star!**
