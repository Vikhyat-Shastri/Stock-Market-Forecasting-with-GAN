# 🚀 Quick Start Guide - Stock Market GAN

## Overview
This guide will help you get started with the Stock Market Forecasting GAN project in under 30 minutes.

---

## Prerequisites Checklist

✅ **Required:**
- [ ] Python 3.9 or higher installed
- [ ] Git installed
- [ ] 8GB+ RAM
- [ ] 10GB+ free disk space

✅ **Recommended for GPU Training:**
- [ ] NVIDIA GPU (GTX 1060 or better)
- [ ] CUDA 11.8+ installed
- [ ] cuDNN installed

✅ **Optional:**
- [ ] Docker installed (for containerized deployment)
- [ ] MLflow account (for experiment tracking)

---

## Step-by-Step Setup

### 1. Clone and Setup Environment (5 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/Stock-Market-Forecasting-with-GAN.git
cd Stock-Market-Forecasting-with-GAN

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify Installation (2 minutes)

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Run tests
pytest tests/ -v
```

### 3. Download Sample Data (3 minutes)

```bash
# Download data for a single stock (fast for testing)
python scripts/download_data.py --ticker GS --start-date 2020-01-01 --end-date 2023-12-31
```

Or use Python:
```python
from src.data.collectors import DataCollector

collector = DataCollector(
    ticker="GS",
    start_date="2020-01-01", 
    end_date="2023-12-31"
)
primary, correlated = collector.fetch_all_data()
print(f"Downloaded {len(primary)} days of data")
```

### 4. Configure Settings (2 minutes)

Edit `configs/training_config.yaml` for quick training:

```yaml
training:
  epochs: 10  # Reduce from 200 for quick test
  batch_size: 32
  device: "cuda"  # or "cpu"

checkpoint:
  save_interval: 5
```

### 5. Train Your First Model (10 minutes)

```bash
# Quick training run
python scripts/train.py \
    --model-config configs/model_config.yaml \
    --training-config configs/training_config.yaml \
    --data-config configs/data_config.yaml
```

**What happens during training:**
- Data is loaded and preprocessed
- Features are engineered (technical indicators, Fourier transforms, etc.)
- WGAN-GP trains Generator and Discriminator
- Metrics are logged to console and MLflow
- Checkpoints are saved to `models/checkpoints/`

### 6. Monitor Training (Optional)

**Option A: MLflow UI**
```bash
# Start MLflow UI in new terminal
mlflow ui

# Open browser: http://localhost:5000
```

**Option B: Check logs**
```bash
# View latest log file
cat logs/train_*.log | tail -100
```

### 7. Make Predictions (3 minutes)

```python
from src.models.gan import WGAN_GP
from src.models.generator import LSTMGenerator
from src.models.discriminator import CNNDiscriminator
import torch

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = LSTMGenerator(
    noise_dim=100,
    hidden_dim=256,
    num_layers=2,
    n_features=112,
    sequence_length=20
).to(device)

wgan = WGAN_GP(generator=generator, discriminator=None, device=device)
wgan.load_checkpoint('models/checkpoints/checkpoint_epoch_10.pt')

# Generate predictions
samples = wgan.generate_samples(n_samples=100)
print(f"Generated shape: {samples.shape}")
```

---

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory
**Solution:**
```yaml
# In training_config.yaml, reduce:
training:
  batch_size: 16  # Reduce from 64
```

### Issue 2: ModuleNotFoundError
**Solution:**
```bash
# Install package in development mode
pip install -e .
```

### Issue 3: Slow Training on CPU
**Solution:**
- Reduce model size in `configs/model_config.yaml`
- Reduce number of epochs
- Use smaller dataset (shorter date range)

### Issue 4: Data Download Fails
**Solution:**
```python
# Try different ticker or date range
# Check internet connection
# Verify yfinance is installed
pip install --upgrade yfinance
```

---

## Next Steps

### A. Hyperparameter Optimization
```bash
python scripts/optimize.py --n-trials 50
```

### B. Full Training Run
```yaml
# Edit configs/training_config.yaml
training:
  epochs: 200
  batch_size: 64
```

### C. Deploy with Docker
```bash
docker-compose -f deployment/docker-compose.yml up
```

### D. Create Custom Features
Edit `src/data/feature_engineering.py` to add your own indicators.

### E. Experiment with Architectures
- Try `AttentionLSTMGenerator` instead of `LSTMGenerator`
- Try `ResidualCNNDiscriminator` instead of `CNNDiscriminator`

---

## Project Structure Quick Reference

```
📁 Important Directories:
├── src/data/          → Data collection & preprocessing
├── src/models/        → Neural network architectures
├── configs/           → YAML configuration files
├── scripts/           → Executable training/prediction scripts
├── models/            → Saved model checkpoints
├── data/              → Raw and processed data
└── logs/              → Training logs

📄 Key Files:
├── scripts/train.py                → Main training script
├── configs/training_config.yaml    → Training parameters
├── configs/model_config.yaml       → Model architecture
├── configs/data_config.yaml        → Data sources & features
└── README.md                       → Full documentation
```

---

## Useful Commands

```bash
# Install package in dev mode
pip install -e .

# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Lint code
flake8 src/ scripts/ tests/

# Start MLflow
mlflow ui

# Build Docker image
docker build -t stock-gan:latest -f deployment/Dockerfile .

# Run Docker container
docker run --gpus all -p 8000:8000 stock-gan:latest

# View GPU usage
nvidia-smi -l 1
```

---

## Performance Tips

### For Faster Training:
1. **Use GPU**: Set `device: "cuda"` in config
2. **Increase batch size**: If you have enough memory
3. **Use mixed precision**: Set `mixed_precision: true`
4. **Reduce features**: Disable ARIMA (it's slow)
5. **Use fewer correlated assets**: Edit data_config.yaml

### For Better Results:
1. **More epochs**: Train for 200+ epochs
2. **Hyperparameter tuning**: Use Optuna optimization
3. **More data**: Increase date range
4. **Feature engineering**: Add domain-specific features
5. **Ensemble models**: Train multiple models and average

---

## Getting Help

- 📖 **Full Documentation**: See [README.md](README.md)
- 🐛 **Report Issues**: GitHub Issues
- 💬 **Discussions**: GitHub Discussions
- 📧 **Contact**: your.email@example.com

---

## Success Checklist

By the end of this guide, you should have:

- ✅ Working Python environment with all dependencies
- ✅ Downloaded sample stock market data
- ✅ Trained a GAN model (even if just 10 epochs)
- ✅ Generated predictions from the model
- ✅ Understand the project structure
- ✅ Know how to modify configurations
- ✅ Access to training metrics and logs

**Congratulations! You're ready to start experimenting! 🎉**

---

## What's Next?

1. **Experiment**: Try different stocks, features, architectures
2. **Optimize**: Run hyperparameter optimization
3. **Deploy**: Create a production API
4. **Share**: Contribute improvements back to the project

---

**Happy Forecasting! 📈**
