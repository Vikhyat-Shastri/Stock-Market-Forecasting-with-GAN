"""
Model monitoring and concept drift detection for production ML systems.
Includes data drift detection, performance degradation alerts, and automatic retraining triggers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import logging
from datetime import datetime, timedelta
from scipy import stats
from collections import deque
import json
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


class KolmogorovSmirnovTest:
    """
    Performs Kolmogorov-Smirnov test for distribution drift detection.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize KS test.
        
        Args:
            significance_level: Significance level for hypothesis test
        """
        self.significance_level = significance_level
        logger.info(f"Initialized KS test with alpha={significance_level}")
    
    def detect_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_name: str = "feature"
    ) -> Dict:
        """
        Detect distribution drift using KS test.
        
        Args:
            reference_data: Reference distribution (training data)
            current_data: Current distribution (production data)
            feature_name: Name of feature being tested
        
        Returns:
            Dictionary with test results
        """
        # Perform KS test
        statistic, p_value = stats.ks_2samp(reference_data, current_data)
        
        # Detect drift
        drift_detected = p_value < self.significance_level
        
        result = {
            'feature': feature_name,
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected,
            'significance_level': self.significance_level,
            'reference_size': len(reference_data),
            'current_size': len(current_data)
        }
        
        if drift_detected:
            logger.warning(
                f"Drift detected in {feature_name}: "
                f"KS={statistic:.4f}, p={p_value:.4f}"
            )
        
        return result


class PopulationStabilityIndex:
    """
    Calculates Population Stability Index (PSI) for feature drift detection.
    """
    
    def __init__(self, bins: int = 10, threshold: float = 0.2):
        """
        Initialize PSI calculator.
        
        Args:
            bins: Number of bins for discretization
            threshold: PSI threshold (>0.2 indicates significant drift)
        """
        self.bins = bins
        self.threshold = threshold
        logger.info(f"Initialized PSI with bins={bins}, threshold={threshold}")
    
    def calculate_psi(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_name: str = "feature"
    ) -> Dict:
        """
        Calculate PSI between reference and current data.
        
        Args:
            reference_data: Reference distribution
            current_data: Current distribution
            feature_name: Feature name
        
        Returns:
            Dictionary with PSI results
        """
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference_data, bins=self.bins)
        
        # Calculate distributions
        ref_counts, _ = np.histogram(reference_data, bins=bin_edges)
        curr_counts, _ = np.histogram(current_data, bins=bin_edges)
        
        # Convert to proportions
        ref_props = ref_counts / len(reference_data)
        curr_props = curr_counts / len(current_data)
        
        # Avoid division by zero
        ref_props = np.where(ref_props == 0, 0.0001, ref_props)
        curr_props = np.where(curr_props == 0, 0.0001, curr_props)
        
        # Calculate PSI
        psi_values = (curr_props - ref_props) * np.log(curr_props / ref_props)
        psi = np.sum(psi_values)
        
        # Interpret PSI
        if psi < 0.1:
            interpretation = "No significant change"
        elif psi < 0.2:
            interpretation = "Slight change"
        else:
            interpretation = "Significant change - retraining recommended"
        
        drift_detected = psi > self.threshold
        
        result = {
            'feature': feature_name,
            'psi': psi,
            'drift_detected': drift_detected,
            'threshold': self.threshold,
            'interpretation': interpretation,
            'bin_edges': bin_edges.tolist(),
            'ref_props': ref_props.tolist(),
            'curr_props': curr_props.tolist()
        }
        
        if drift_detected:
            logger.warning(
                f"PSI drift detected in {feature_name}: "
                f"PSI={psi:.4f} (threshold={self.threshold})"
            )
        
        return result


class PerformanceDegradationDetector:
    """
    Monitors model performance over time and detects degradation.
    """
    
    def __init__(
        self,
        metric_name: str = "accuracy",
        window_size: int = 100,
        degradation_threshold: float = 0.1
    ):
        """
        Initialize performance detector.
        
        Args:
            metric_name: Name of performance metric
            window_size: Window size for rolling statistics
            degradation_threshold: Threshold for degradation (e.g., 0.1 = 10% drop)
        """
        self.metric_name = metric_name
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        
        self.baseline_performance = None
        self.performance_history = deque(maxlen=window_size * 10)
        
        logger.info(
            f"Initialized PerformanceDegradationDetector: "
            f"metric={metric_name}, window={window_size}"
        )
    
    def set_baseline(self, baseline_value: float):
        """
        Set baseline performance from validation set.
        
        Args:
            baseline_value: Baseline metric value
        """
        self.baseline_performance = baseline_value
        logger.info(f"Set baseline {self.metric_name}: {baseline_value:.4f}")
    
    def add_observation(self, timestamp: datetime, metric_value: float):
        """
        Add new performance observation.
        
        Args:
            timestamp: Observation timestamp
            metric_value: Metric value
        """
        self.performance_history.append({
            'timestamp': timestamp,
            'value': metric_value
        })
    
    def detect_degradation(self) -> Dict:
        """
        Detect performance degradation.
        
        Returns:
            Dictionary with degradation analysis
        """
        if self.baseline_performance is None:
            raise ValueError("Must set baseline performance first")
        
        if len(self.performance_history) < self.window_size:
            return {
                'degradation_detected': False,
                'reason': f'Insufficient data ({len(self.performance_history)}/{self.window_size})'
            }
        
        # Get recent performance
        recent_values = [obs['value'] for obs in list(self.performance_history)[-self.window_size:]]
        current_performance = np.mean(recent_values)
        performance_std = np.std(recent_values)
        
        # Calculate degradation
        degradation = (self.baseline_performance - current_performance) / self.baseline_performance
        
        # Detect significant degradation
        degradation_detected = degradation > self.degradation_threshold
        
        # Statistical test
        t_stat, p_value = stats.ttest_1samp(recent_values, self.baseline_performance)
        statistically_significant = p_value < 0.05 and current_performance < self.baseline_performance
        
        result = {
            'degradation_detected': degradation_detected or statistically_significant,
            'baseline_performance': self.baseline_performance,
            'current_performance': current_performance,
            'performance_std': performance_std,
            'degradation_pct': degradation * 100,
            'threshold_pct': self.degradation_threshold * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'statistically_significant': statistically_significant,
            'observations_count': len(recent_values)
        }
        
        if result['degradation_detected']:
            logger.warning(
                f"Performance degradation detected: "
                f"{self.metric_name} dropped {degradation*100:.2f}% "
                f"(from {self.baseline_performance:.4f} to {current_performance:.4f})"
            )
        
        return result
    
    def get_performance_trend(self) -> pd.DataFrame:
        """
        Get performance trend over time.
        
        Returns:
            DataFrame with performance history
        """
        if len(self.performance_history) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(list(self.performance_history))
        df['rolling_mean'] = df['value'].rolling(self.window_size, min_periods=1).mean()
        df['rolling_std'] = df['value'].rolling(self.window_size, min_periods=1).std()
        
        return df


class ConceptDriftDetector:
    """
    Comprehensive concept drift detector combining multiple methods.
    """
    
    def __init__(
        self,
        feature_names: List[str],
        performance_metric: str = "accuracy",
        drift_sensitivity: str = "medium"
    ):
        """
        Initialize concept drift detector.
        
        Args:
            feature_names: List of feature names to monitor
            performance_metric: Performance metric to track
            drift_sensitivity: Sensitivity level ('low', 'medium', 'high')
        """
        self.feature_names = feature_names
        self.performance_metric = performance_metric
        
        # Set thresholds based on sensitivity
        sensitivity_params = {
            'low': {'psi': 0.25, 'ks_alpha': 0.01, 'perf_threshold': 0.15},
            'medium': {'psi': 0.20, 'ks_alpha': 0.05, 'perf_threshold': 0.10},
            'high': {'psi': 0.15, 'ks_alpha': 0.10, 'perf_threshold': 0.05}
        }
        
        params = sensitivity_params.get(drift_sensitivity, sensitivity_params['medium'])
        
        # Initialize detectors
        self.ks_test = KolmogorovSmirnovTest(significance_level=params['ks_alpha'])
        self.psi_calculator = PopulationStabilityIndex(threshold=params['psi'])
        self.perf_detector = PerformanceDegradationDetector(
            metric_name=performance_metric,
            degradation_threshold=params['perf_threshold']
        )
        
        # Store reference data
        self.reference_data = None
        self.drift_history = []
        
        logger.info(
            f"Initialized ConceptDriftDetector: "
            f"features={len(feature_names)}, sensitivity={drift_sensitivity}"
        )
    
    def set_reference_data(self, reference_df: pd.DataFrame):
        """
        Set reference data distribution (training data).
        
        Args:
            reference_df: Reference DataFrame with features
        """
        self.reference_data = reference_df[self.feature_names].copy()
        logger.info(f"Set reference data: {len(self.reference_data)} samples")
    
    def set_baseline_performance(self, performance: float):
        """
        Set baseline model performance.
        
        Args:
            performance: Baseline performance value
        """
        self.perf_detector.set_baseline(performance)
    
    def detect_drift(
        self,
        current_df: pd.DataFrame,
        current_performance: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Detect drift in current data.
        
        Args:
            current_df: Current DataFrame with features
            current_performance: Current model performance (optional)
            timestamp: Current timestamp
        
        Returns:
            Dictionary with comprehensive drift analysis
        """
        if self.reference_data is None:
            raise ValueError("Must set reference data first")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Test each feature
        feature_drifts = []
        drifted_features = []
        
        for feature in self.feature_names:
            if feature not in current_df.columns:
                logger.warning(f"Feature {feature} not found in current data")
                continue
            
            ref_values = self.reference_data[feature].dropna().values
            curr_values = current_df[feature].dropna().values
            
            if len(curr_values) == 0:
                continue
            
            # KS test
            ks_result = self.ks_test.detect_drift(ref_values, curr_values, feature)
            
            # PSI
            psi_result = self.psi_calculator.calculate_psi(ref_values, curr_values, feature)
            
            # Combine results
            drift_detected = ks_result['drift_detected'] or psi_result['drift_detected']
            
            feature_drift = {
                'feature': feature,
                'drift_detected': drift_detected,
                'ks_statistic': ks_result['statistic'],
                'ks_p_value': ks_result['p_value'],
                'psi': psi_result['psi'],
                'psi_interpretation': psi_result['interpretation']
            }
            
            feature_drifts.append(feature_drift)
            
            if drift_detected:
                drifted_features.append(feature)
        
        # Performance degradation
        perf_degradation = None
        if current_performance is not None:
            self.perf_detector.add_observation(timestamp, current_performance)
            
            if len(self.perf_detector.performance_history) >= self.perf_detector.window_size:
                perf_degradation = self.perf_detector.detect_degradation()
        
        # Overall drift assessment
        feature_drift_ratio = len(drifted_features) / len(self.feature_names)
        
        overall_drift_detected = (
            feature_drift_ratio > 0.2 or  # More than 20% features drifted
            (perf_degradation and perf_degradation['degradation_detected'])
        )
        
        # Determine severity
        if feature_drift_ratio > 0.5 or (perf_degradation and perf_degradation.get('degradation_pct', 0) > 20):
            severity = "HIGH"
            action = "RETRAIN_IMMEDIATELY"
        elif feature_drift_ratio > 0.2 or (perf_degradation and perf_degradation.get('degradation_pct', 0) > 10):
            severity = "MEDIUM"
            action = "SCHEDULE_RETRAINING"
        else:
            severity = "LOW"
            action = "MONITOR"
        
        result = {
            'timestamp': timestamp,
            'overall_drift_detected': overall_drift_detected,
            'severity': severity,
            'recommended_action': action,
            'drifted_features': drifted_features,
            'feature_drift_ratio': feature_drift_ratio,
            'total_features_tested': len(self.feature_names),
            'feature_drifts': feature_drifts,
            'performance_degradation': perf_degradation
        }
        
        # Store in history
        self.drift_history.append(result)
        
        if overall_drift_detected:
            logger.warning(
                f"DRIFT DETECTED - Severity: {severity}, Action: {action}\n"
                f"Drifted features: {drifted_features}\n"
                f"Feature drift ratio: {feature_drift_ratio*100:.1f}%"
            )
        
        return result
    
    def get_drift_report(self) -> pd.DataFrame:
        """
        Generate drift monitoring report.
        
        Returns:
            DataFrame with drift history
        """
        if len(self.drift_history) == 0:
            return pd.DataFrame()
        
        records = []
        for entry in self.drift_history:
            record = {
                'timestamp': entry['timestamp'],
                'drift_detected': entry['overall_drift_detected'],
                'severity': entry['severity'],
                'action': entry['recommended_action'],
                'drifted_features_count': len(entry['drifted_features']),
                'feature_drift_ratio': entry['feature_drift_ratio']
            }
            
            if entry['performance_degradation']:
                record['performance_degradation'] = entry['performance_degradation'].get('degradation_detected', False)
                record['current_performance'] = entry['performance_degradation'].get('current_performance')
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def save_report(self, filepath: str):
        """
        Save drift report to file.
        
        Args:
            filepath: Path to save report
        """
        report = self.get_drift_report()
        report.to_csv(filepath, index=False)
        logger.info(f"Saved drift report to {filepath}")


class AutoRetrainingTrigger:
    """
    Automatic model retraining trigger based on drift detection.
    """
    
    def __init__(
        self,
        retrain_callback: Callable,
        min_retrain_interval: timedelta = timedelta(days=7),
        cooldown_period: timedelta = timedelta(hours=24)
    ):
        """
        Initialize auto-retraining trigger.
        
        Args:
            retrain_callback: Function to call for retraining
            min_retrain_interval: Minimum time between retraining
            cooldown_period: Cooldown after retraining
        """
        self.retrain_callback = retrain_callback
        self.min_retrain_interval = min_retrain_interval
        self.cooldown_period = cooldown_period
        
        self.last_retrain_time = None
        self.retrain_history = []
        
        logger.info("Initialized AutoRetrainingTrigger")
    
    def check_and_trigger(self, drift_result: Dict) -> bool:
        """
        Check drift result and trigger retraining if needed.
        
        Args:
            drift_result: Result from ConceptDriftDetector
        
        Returns:
            True if retraining was triggered
        """
        now = datetime.now()
        
        # Check if in cooldown
        if self.last_retrain_time:
            time_since_retrain = now - self.last_retrain_time
            
            if time_since_retrain < self.cooldown_period:
                logger.info(f"In cooldown period ({time_since_retrain} since last retrain)")
                return False
            
            if time_since_retrain < self.min_retrain_interval:
                # Only trigger if high severity
                if drift_result.get('severity') != 'HIGH':
                    logger.info(f"Not enough time since last retrain for {drift_result.get('severity')} severity")
                    return False
        
        # Check if retraining is recommended
        action = drift_result.get('recommended_action')
        
        if action in ['RETRAIN_IMMEDIATELY', 'SCHEDULE_RETRAINING']:
            logger.info(f"Triggering retraining: {action}")
            
            try:
                # Call retraining function
                self.retrain_callback(drift_result)
                
                # Update state
                self.last_retrain_time = now
                self.retrain_history.append({
                    'timestamp': now,
                    'reason': action,
                    'drift_result': drift_result
                })
                
                logger.info("Retraining completed successfully")
                return True
                
            except Exception as e:
                logger.error(f"Retraining failed: {e}")
                return False
        
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    
    # Reference data (training)
    n_samples = 1000
    reference_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(5, 2, n_samples),
        'feature3': np.random.exponential(2, n_samples)
    })
    
    # Current data (with drift)
    current_df = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, n_samples),  # Mean shift
        'feature2': np.random.normal(5, 2, n_samples),      # No drift
        'feature3': np.random.exponential(3, n_samples)      # Scale change
    })
    
    # Initialize detector
    detector = ConceptDriftDetector(
        feature_names=['feature1', 'feature2', 'feature3'],
        drift_sensitivity='medium'
    )
    
    detector.set_reference_data(reference_df)
    detector.set_baseline_performance(0.85)
    
    # Detect drift
    drift_result = detector.detect_drift(
        current_df,
        current_performance=0.78
    )
    
    print("\nDrift Detection Results:")
    print(f"Overall Drift Detected: {drift_result['overall_drift_detected']}")
    print(f"Severity: {drift_result['severity']}")
    print(f"Recommended Action: {drift_result['recommended_action']}")
    print(f"Drifted Features: {drift_result['drifted_features']}")
    print(f"Feature Drift Ratio: {drift_result['feature_drift_ratio']*100:.1f}%")
