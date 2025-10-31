"""
Main training script for Stock Market Forecasting GAN.
Integrates all components for end-to-end training.
"""

import torch
import argparse
import yaml
import logging
from pathlib import Path
import sys
import mlflow
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.collectors import DataCollector
from src.data.preprocessors import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.data.datasets import create_dataloaders
from src.models.generator import LSTMGenerator, AttentionLSTMGenerator
from src.models.discriminator import CNNDiscriminator, ResidualCNNDiscriminator
from src.models.gan import WGAN_GP
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_mlflow(config: dict):
    """Setup MLflow experiment tracking."""
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])


def create_generator(config: dict, n_features: int, sequence_length: int) -> torch.nn.Module:
    """Create generator model based on config."""
    gen_config = config['generator']
    
    if gen_config['type'] == 'lstm':
        return LSTMGenerator(
            noise_dim=gen_config['noise_dim'],
            hidden_dim=gen_config['hidden_dim'],
            num_layers=gen_config['num_layers'],
            n_features=n_features,
            sequence_length=sequence_length,
            dropout=gen_config['dropout'],
            bidirectional=gen_config['bidirectional']
        )
    elif gen_config['type'] == 'attention_lstm':
        return AttentionLSTMGenerator(
            noise_dim=gen_config['noise_dim'],
            hidden_dim=gen_config['hidden_dim'],
            num_layers=gen_config['num_layers'],
            n_features=n_features,
            sequence_length=sequence_length,
            dropout=gen_config['dropout']
        )
    else:
        raise ValueError(f"Unknown generator type: {gen_config['type']}")


def create_discriminator(config: dict, n_features: int, sequence_length: int) -> torch.nn.Module:
    """Create discriminator model based on config."""
    disc_config = config['discriminator']
    
    if disc_config['type'] == 'cnn':
        return CNNDiscriminator(
            n_features=n_features,
            sequence_length=sequence_length,
            base_filters=disc_config['base_filters'],
            num_conv_layers=disc_config['num_conv_layers'],
            fc_hidden_dim=disc_config['fc_hidden_dim'],
            dropout=disc_config['dropout'],
            use_spectral_norm=disc_config['use_spectral_norm']
        )
    elif disc_config['type'] == 'residual_cnn':
        return ResidualCNNDiscriminator(
            n_features=n_features,
            sequence_length=sequence_length,
            base_filters=disc_config['base_filters'],
            num_blocks=disc_config['num_conv_layers'],
            fc_hidden_dim=disc_config['fc_hidden_dim'],
            dropout=disc_config['dropout']
        )
    else:
        raise ValueError(f"Unknown discriminator type: {disc_config['type']}")


def train(args):
    """Main training function."""
    # Load configurations
    model_config = load_config(Path(args.model_config))
    training_config = load_config(Path(args.training_config))
    data_config = load_config(Path(args.data_config))
    
    # Setup logging
    setup_logger(
        log_dir=Path(training_config['logging']['log_dir']),
        level=training_config['logging']['level']
    )
    
    logger.info("=" * 70)
    logger.info("Stock Market Forecasting with GAN - Training Started")
    logger.info("=" * 70)
    
    # Setup MLflow
    if training_config['logging']['mlflow']:
        setup_mlflow(training_config)
        mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Set random seed
    torch.manual_seed(training_config['training']['seed'])
    
    # Setup device
    device = torch.device(
        training_config['training']['device']
        if torch.cuda.is_available() else 'cpu'
    )
    logger.info(f"Using device: {device}")
    
    # ========================================
    # Step 1: Data Collection
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Data Collection")
    logger.info("=" * 70)
    
    collector = DataCollector(
        ticker=data_config['primary_stock']['ticker'],
        start_date=data_config['date_range']['start_date'],
        end_date=data_config['date_range']['end_date'],
        cache_dir=Path(data_config['data_paths']['raw_data'])
    )
    
    primary_data, correlated_data = collector.fetch_all_data()
    logger.info(f"Collected data: {len(primary_data)} days, {len(correlated_data)} assets")
    
    # ========================================
    # Step 2: Data Preprocessing
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Data Preprocessing")
    logger.info("=" * 70)
    
    preprocessor = DataPreprocessor(
        scaling_method=data_config['preprocessing']['scaling_method'],
        handle_outliers=data_config['preprocessing']['handle_outliers'],
        outlier_std=data_config['preprocessing']['outlier_std']
    )
    
    primary_clean, correlated_clean = preprocessor.preprocess_pipeline(
        primary_data,
        correlated_data,
        fit=True
    )
    
    # ========================================
    # Step 3: Feature Engineering
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Feature Engineering")
    logger.info("=" * 70)
    
    engineer = FeatureEngineer()
    
    features_df = engineer.create_all_features(
        primary_clean,
        add_technical=data_config['features']['technical_indicators']['enabled'],
        add_fourier=data_config['features']['fourier']['enabled'],
        add_arima=data_config['features']['arima']['enabled'],
        add_lags=data_config['features']['lag_features']['enabled'],
        add_rolling=data_config['features']['rolling_stats']['enabled']
    )
    
    logger.info(f"Total features created: {len(features_df.columns)}")
    
    # ========================================
    # Step 4: Split Data
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Data Splitting")
    logger.info("=" * 70)
    
    train_size = int(len(features_df) * data_config['data_split']['train_ratio'])
    val_size = int(len(features_df) * data_config['data_split']['val_ratio'])
    
    train_df = features_df.iloc[:train_size]
    val_df = features_df.iloc[train_size:train_size + val_size]
    test_df = features_df.iloc[train_size + val_size:]
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # ========================================
    # Step 5: Create DataLoaders
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Creating DataLoaders")
    logger.info("=" * 70)
    
    train_loader, val_loader = create_dataloaders(
        train_df,
        val_df,
        batch_size=training_config['training']['batch_size'],
        sequence_length=model_config['generator']['sequence_length'],
        num_workers=training_config['training']['num_workers'],
        dataset_type='gan'
    )
    
    # ========================================
    # Step 6: Create Models
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: Creating Models")
    logger.info("=" * 70)
    
    n_features = train_df.shape[1]
    sequence_length = model_config['generator']['sequence_length']
    
    generator = create_generator(model_config, n_features, sequence_length)
    discriminator = create_discriminator(model_config, n_features, sequence_length)
    
    # ========================================
    # Step 7: Create WGAN-GP
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: Initializing WGAN-GP")
    logger.info("=" * 70)
    
    wgan = WGAN_GP(
        generator=generator,
        discriminator=discriminator,
        noise_dim=model_config['generator']['noise_dim'],
        lambda_gp=model_config['wgan']['lambda_gp'],
        n_critic=model_config['wgan']['n_critic'],
        device=device
    )
    
    # Configure optimizers
    wgan.configure_optimizers(
        g_lr=model_config['optimizer']['g_lr'],
        d_lr=model_config['optimizer']['d_lr'],
        betas=tuple(model_config['optimizer']['betas'])
    )
    
    # ========================================
    # Step 8: Training
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: Training GAN")
    logger.info("=" * 70)
    
    # Training callback for MLflow logging
    def training_callback(epoch, metrics, model):
        if training_config['logging']['mlflow']:
            mlflow.log_metrics(metrics, step=epoch)
    
    wgan.fit(
        dataloader=train_loader,
        epochs=training_config['training']['epochs'],
        save_dir=Path(training_config['checkpoint']['save_dir']),
        save_interval=training_config['checkpoint']['save_interval'],
        callback=training_callback if training_config['logging']['mlflow'] else None
    )
    
    # ========================================
    # Step 9: Save Final Model
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 9: Saving Final Model")
    logger.info("=" * 70)
    
    final_model_dir = Path("models/final")
    final_model_dir.mkdir(parents=True, exist_ok=True)
    
    wgan.save_checkpoint(
        final_model_dir / "final_model.pt",
        epoch=training_config['training']['epochs']
    )
    
    # Log model to MLflow
    if training_config['logging']['mlflow'] and training_config['mlflow']['log_models']:
        mlflow.pytorch.log_model(generator, "generator")
        mlflow.pytorch.log_model(discriminator, "discriminator")
    
    # End MLflow run
    if training_config['logging']['mlflow']:
        mlflow.end_run()
    
    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description="Train Stock Market Forecasting GAN"
    )
    
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--training-config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data configuration file"
    )
    
    args = parser.parse_args()
    
    try:
        train(args)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
