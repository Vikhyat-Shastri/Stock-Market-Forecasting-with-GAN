"""
Bayesian hyperparameter optimization using Optuna.
Optimizes GAN training hyperparameters with efficient search strategy.
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
import mlflow
import yaml
from pathlib import Path
import numpy as np

from src.models.generator import LSTMGenerator, AttentionLSTMGenerator
from src.models.discriminator import CNNDiscriminator, ResidualCNNDiscriminator
from src.models.gan import WGAN_GP
from src.data.collectors import DataCollector
from src.data.preprocessors import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.data.datasets import create_dataloaders
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


class GANObjective:
    """
    Objective function for Optuna optimization.
    Trains GAN with given hyperparameters and returns validation metric.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_path: str,
        device: torch.device,
        max_epochs: int = 50,
        early_stopping_patience: int = 10
    ):
        """
        Initialize objective function.
        
        Args:
            config: Base configuration
            data_path: Path to preprocessed data
            device: Training device
            max_epochs: Maximum training epochs
            early_stopping_patience: Early stopping patience
        """
        self.config = config
        self.data_path = data_path
        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        
        logger.info("Initialized GAN objective for Optuna")
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function called by Optuna.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Validation metric (to minimize)
        """
        # Suggest hyperparameters
        hyperparams = self._suggest_hyperparameters(trial)
        
        logger.info(f"Trial {trial.number} hyperparameters: {hyperparams}")
        
        try:
            # Train GAN with suggested hyperparameters
            val_metric = self._train_gan(hyperparams, trial)
            
            # Log to MLflow
            with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
                mlflow.log_params(hyperparams)
                mlflow.log_metric("val_loss", val_metric)
            
            return val_metric
        
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned()
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for current trial.
        
        Args:
            trial: Optuna trial
        
        Returns:
            Dictionary of hyperparameters
        """
        return {
            # Generator hyperparameters
            'gen_hidden_dim': trial.suggest_categorical('gen_hidden_dim', [128, 256, 512]),
            'gen_num_layers': trial.suggest_int('gen_num_layers', 1, 4),
            'gen_dropout': trial.suggest_float('gen_dropout', 0.1, 0.5),
            'gen_bidirectional': trial.suggest_categorical('gen_bidirectional', [True, False]),
            
            # Discriminator hyperparameters
            'disc_base_filters': trial.suggest_categorical('disc_base_filters', [32, 64, 128]),
            'disc_num_layers': trial.suggest_int('disc_num_layers', 2, 5),
            'disc_dropout': trial.suggest_float('disc_dropout', 0.1, 0.5),
            
            # Training hyperparameters
            'gen_lr': trial.suggest_float('gen_lr', 1e-5, 1e-3, log=True),
            'disc_lr': trial.suggest_float('disc_lr', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'lambda_gp': trial.suggest_float('lambda_gp', 5.0, 15.0),
            'n_critic': trial.suggest_int('n_critic', 3, 7),
            
            # Regularization
            'l1_weight': trial.suggest_float('l1_weight', 0.0, 0.01),
            'l2_weight': trial.suggest_float('l2_weight', 0.0, 0.01),
            
            # Architecture variants
            'use_attention': trial.suggest_categorical('use_attention', [True, False]),
            'use_residual': trial.suggest_categorical('use_residual', [True, False]),
        }
    
    def _train_gan(
        self,
        hyperparams: Dict[str, Any],
        trial: optuna.Trial
    ) -> float:
        """
        Train GAN with given hyperparameters.
        
        Args:
            hyperparams: Hyperparameters dictionary
            trial: Optuna trial for pruning
        
        Returns:
            Validation metric
        """
        # Load data
        train_loader, val_loader = self._load_data(hyperparams['batch_size'])
        
        # Create models
        generator = self._create_generator(hyperparams)
        discriminator = self._create_discriminator(hyperparams)
        
        # Create GAN
        gan = WGAN_GP(
            generator=generator,
            discriminator=discriminator,
            lambda_gp=hyperparams['lambda_gp'],
            n_critic=hyperparams['n_critic'],
            device=self.device
        )
        
        # Optimizers
        gen_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=hyperparams['gen_lr'],
            betas=(0.5, 0.999),
            weight_decay=hyperparams['l2_weight']
        )
        
        disc_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=hyperparams['disc_lr'],
            betas=(0.5, 0.999),
            weight_decay=hyperparams['l2_weight']
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            # Training
            train_metrics = self._train_epoch(
                gan, train_loader, gen_optimizer, disc_optimizer, hyperparams
            )
            
            # Validation
            val_metrics = self._validate_epoch(gan, val_loader)
            
            val_loss = val_metrics['val_loss']
            
            # Report to Optuna for pruning
            trial.report(val_loss, epoch)
            
            # Pruning
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            logger.info(
                f"Epoch {epoch}: Train Loss={train_metrics['train_loss']:.4f}, "
                f"Val Loss={val_loss:.4f}"
            )
        
        return best_val_loss
    
    def _create_generator(self, hyperparams: Dict[str, Any]) -> nn.Module:
        """Create generator with given hyperparameters."""
        if hyperparams['use_attention']:
            return AttentionLSTMGenerator(
                noise_dim=self.config['noise_dim'],
                hidden_dim=hyperparams['gen_hidden_dim'],
                feature_dim=self.config['feature_dim'],
                seq_length=self.config['seq_length'],
                num_layers=hyperparams['gen_num_layers'],
                dropout=hyperparams['gen_dropout'],
                bidirectional=hyperparams['gen_bidirectional']
            ).to(self.device)
        else:
            return LSTMGenerator(
                noise_dim=self.config['noise_dim'],
                hidden_dim=hyperparams['gen_hidden_dim'],
                feature_dim=self.config['feature_dim'],
                seq_length=self.config['seq_length'],
                num_layers=hyperparams['gen_num_layers'],
                dropout=hyperparams['gen_dropout'],
                bidirectional=hyperparams['gen_bidirectional']
            ).to(self.device)
    
    def _create_discriminator(self, hyperparams: Dict[str, Any]) -> nn.Module:
        """Create discriminator with given hyperparameters."""
        if hyperparams['use_residual']:
            return ResidualCNNDiscriminator(
                feature_dim=self.config['feature_dim'],
                seq_length=self.config['seq_length'],
                base_filters=hyperparams['disc_base_filters'],
                num_conv_layers=hyperparams['disc_num_layers'],
                dropout=hyperparams['disc_dropout']
            ).to(self.device)
        else:
            return CNNDiscriminator(
                feature_dim=self.config['feature_dim'],
                seq_length=self.config['seq_length'],
                base_filters=hyperparams['disc_base_filters'],
                num_conv_layers=hyperparams['disc_num_layers'],
                dropout=hyperparams['disc_dropout']
            ).to(self.device)
    
    def _load_data(self, batch_size: int):
        """Load training and validation data."""
        # Simplified data loading
        # In practice, load preprocessed data from self.data_path
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        # Dummy data for illustration
        X_train = torch.randn(1000, self.config['seq_length'], self.config['feature_dim'])
        y_train = torch.randn(1000, self.config['feature_dim'])
        
        X_val = torch.randn(200, self.config['seq_length'], self.config['feature_dim'])
        y_val = torch.randn(200, self.config['feature_dim'])
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def _train_epoch(self, gan, train_loader, gen_optimizer, disc_optimizer, hyperparams):
        """Train for one epoch."""
        gan.generator.train()
        gan.discriminator.train()
        
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            
            # Train discriminator
            disc_loss = gan.train_discriminator(data, disc_optimizer)
            
            # Train generator
            if batch_idx % gan.n_critic == 0:
                gen_loss = gan.train_generator(data.size(0), gen_optimizer)
            
            total_loss += disc_loss
        
        return {'train_loss': total_loss / len(train_loader)}
    
    def _validate_epoch(self, gan, val_loader):
        """Validate for one epoch."""
        gan.generator.eval()
        gan.discriminator.eval()
        
        total_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                
                # Generate fake data
                noise = torch.randn(data.size(0), self.config['noise_dim']).to(self.device)
                fake_data = gan.generator(noise)
                
                # Compute discriminator scores
                real_score = gan.discriminator(data).mean()
                fake_score = gan.discriminator(fake_data).mean()
                
                # Wasserstein distance
                loss = fake_score - real_score
                total_loss += loss.item()
        
        return {'val_loss': total_loss / len(val_loader)}


def optimize_hyperparameters(
    config_path: str,
    data_path: str,
    n_trials: int = 100,
    study_name: str = "gan_optimization",
    storage: Optional[str] = None,
    n_jobs: int = 1
) -> optuna.Study:
    """
    Run hyperparameter optimization.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to preprocessed data
        n_trials: Number of optimization trials
        study_name: Name of Optuna study
        storage: Database URL for distributed optimization
        n_jobs: Number of parallel jobs
    
    Returns:
        Optuna study object
    """
    # Setup logging
    setup_logger()
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True
    )
    
    # Create objective
    objective = GANObjective(
        config=config,
        data_path=data_path,
        device=device,
        max_epochs=50,
        early_stopping_patience=10
    )
    
    # Start MLflow run
    mlflow.set_experiment("gan_hyperparameter_optimization")
    
    with mlflow.start_run(run_name=f"optuna_study_{study_name}"):
        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        # Log best parameters
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_value", study.best_value)
        
        # Save study
        study_path = Path("models") / "optuna_studies" / f"{study_name}.pkl"
        study_path.parent.mkdir(parents=True, exist_ok=True)
        
        import joblib
        joblib.dump(study, study_path)
        mlflow.log_artifact(study_path)
    
    # Print results
    logger.info(f"\nBest trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.6f}")
    logger.info(f"Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Plot optimization history
    try:
        import optuna.visualization as vis
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html("models/optuna_studies/optimization_history.html")
        
        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html("models/optuna_studies/param_importances.html")
        
        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html("models/optuna_studies/parallel_coordinate.html")
        
        logger.info("Saved optimization visualizations")
    except Exception as e:
        logger.warning(f"Failed to create visualizations: {e}")
    
    return study


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for GAN")
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/processed/stock_data.pkl',
                        help='Path to preprocessed data')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of optimization trials')
    parser.add_argument('--study-name', type=str, default='gan_optimization',
                        help='Name of Optuna study')
    parser.add_argument('--storage', type=str, default=None,
                        help='Database URL for distributed optimization')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    # Run optimization
    study = optimize_hyperparameters(
        config_path=args.config,
        data_path=args.data,
        n_trials=args.trials,
        study_name=args.study_name,
        storage=args.storage,
        n_jobs=args.n_jobs
    )
