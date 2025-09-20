import os
#!/usr/bin/env python3
"""
Market Economics Processor - Phase 7 Alternative Market Intelligence Platform
Place in: /home/starlord/Projects/Bev/src/alternative_market/economics_processor.py

Supply/demand modeling with price volatility analysis and market manipulation detection.
"""

import asyncio
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import asyncpg
import aioredis
from aiokafka import AIOKafkaProducer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import networkx as nx
from decimal import Decimal, ROUND_HALF_UP
import uuid
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """Individual market data point"""
    timestamp: datetime
    marketplace: str
    category: str
    product_id: str
    price: Decimal
    currency: str
    volume: int
    listings_count: int
    vendor_count: int
    demand_indicators: Dict[str, float]
    supply_indicators: Dict[str, float]

@dataclass
class SupplyDemandModel:
    """Supply and demand model results"""
    model_id: str
    marketplace: str
    category: str
    time_period: Tuple[datetime, datetime]
    supply_elasticity: float
    demand_elasticity: float
    equilibrium_price: Decimal
    equilibrium_quantity: int
    supply_curve_params: Dict[str, float]
    demand_curve_params: Dict[str, float]
    r_squared: float
    prediction_accuracy: float
    last_updated: datetime

@dataclass
class PriceVolatilityAnalysis:
    """Price volatility analysis results"""
    analysis_id: str
    marketplace: str
    category: str
    time_period: Tuple[datetime, datetime]
    price_volatility: float  # Standard deviation of returns
    volume_volatility: float
    volatility_trend: str  # 'increasing', 'decreasing', 'stable'
    volatility_clusters: List[Tuple[datetime, datetime]]
    regime_changes: List[datetime]
    garch_parameters: Dict[str, float]
    var_95: float  # Value at Risk at 95% confidence
    expected_shortfall: float

@dataclass
class MarketManipulationAlert:
    """Market manipulation detection alert"""
    alert_id: str
    marketplace: str
    category: str
    manipulation_type: str  # 'pump_dump', 'wash_trading', 'cornering', 'spoofing'
    confidence_score: float
    detection_timestamp: datetime
    evidence: Dict[str, Any]
    affected_products: Set[str]
    affected_vendors: Set[str]
    estimated_impact: Dict[str, float]
    status: str  # 'active', 'resolved', 'false_positive'

@dataclass
class EconomicForecast:
    """Economic trend forecast"""
    forecast_id: str
    marketplace: str
    category: str
    forecast_horizon_days: int
    price_forecast: List[Tuple[datetime, float, float]]  # (timestamp, price, confidence_interval)
    volume_forecast: List[Tuple[datetime, int, float]]
    trend_direction: str  # 'bullish', 'bearish', 'sideways'
    forecast_accuracy: float
    model_type: str
    generated_at: datetime

@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity detection"""
    opportunity_id: str
    product_category: str
    source_marketplace: str
    target_marketplace: str
    source_price: Decimal
    target_price: Decimal
    price_difference: Decimal
    price_difference_percentage: float
    potential_profit: Decimal
    risk_score: float
    time_window_minutes: int
    detected_at: datetime
    status: str  # 'active', 'expired', 'executed'


class SupplyDemandAnalyzer:
    """Analyzes supply and demand dynamics"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()

    async def analyze_supply_demand(self, market_data: List[MarketDataPoint],
                                  marketplace: str, category: str) -> SupplyDemandModel:
        """Analyze supply and demand dynamics"""

        if len(market_data) < 20:
            raise ValueError("Insufficient data for supply/demand analysis")

        logger.info(f"Analyzing supply/demand for {marketplace}/{category} with {len(market_data)} data points")

        # Prepare data
        df = self._prepare_supply_demand_data(market_data)

        # Calculate supply and demand curves
        supply_curve = self._calculate_supply_curve(df)
        demand_curve = self._calculate_demand_curve(df)

        # Find equilibrium
        equilibrium = self._find_equilibrium(supply_curve, demand_curve)

        # Calculate elasticities
        supply_elasticity = self._calculate_supply_elasticity(df)
        demand_elasticity = self._calculate_demand_elasticity(df)

        # Model validation
        r_squared = self._calculate_model_fit(df, supply_curve, demand_curve)
        accuracy = self._validate_predictions(df, supply_curve, demand_curve)

        model = SupplyDemandModel(
            model_id=str(uuid.uuid4()),
            marketplace=marketplace,
            category=category,
            time_period=(min(d.timestamp for d in market_data),
                        max(d.timestamp for d in market_data)),
            supply_elasticity=supply_elasticity,
            demand_elasticity=demand_elasticity,
            equilibrium_price=Decimal(str(equilibrium['price'])),
            equilibrium_quantity=int(equilibrium['quantity']),
            supply_curve_params=supply_curve,
            demand_curve_params=demand_curve,
            r_squared=r_squared,
            prediction_accuracy=accuracy,
            last_updated=datetime.now()
        )

        return model

    def _prepare_supply_demand_data(self, market_data: List[MarketDataPoint]) -> pd.DataFrame:
        """Prepare data for supply/demand analysis"""

        data = []
        for point in market_data:
            data.append({
                'timestamp': point.timestamp,
                'price': float(point.price),
                'volume': point.volume,
                'listings_count': point.listings_count,
                'vendor_count': point.vendor_count,
                'demand_score': np.mean(list(point.demand_indicators.values())) if point.demand_indicators else 0,
                'supply_score': np.mean(list(point.supply_indicators.values())) if point.supply_indicators else 0
            })

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        # Calculate additional indicators
        df['price_change'] = df['price'].pct_change()
        df['volume_ma'] = df['volume'].rolling(window=7).mean()
        df['price_ma'] = df['price'].rolling(window=7).mean()

        return df.dropna()

    def _calculate_supply_curve(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate supply curve parameters"""

        # Supply generally increases with price
        X = df[['price', 'vendor_count', 'supply_score']].values
        y = df['listings_count'].values

        # Fit linear model for supply curve
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)

        return {
            'price_coefficient': model.coef_[0],
            'vendor_coefficient': model.coef_[1],
            'supply_coefficient': model.coef_[2],
            'intercept': model.intercept_,
            'score': model.score(X, y)
        }

    def _calculate_demand_curve(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate demand curve parameters"""

        # Demand generally decreases with price
        X = df[['price', 'demand_score']].values
        y = df['volume'].values

        # Fit linear model for demand curve
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)

        return {
            'price_coefficient': model.coef_[0],
            'demand_coefficient': model.coef_[1],
            'intercept': model.intercept_,
            'score': model.score(X, y)
        }

    def _find_equilibrium(self, supply_curve: Dict[str, float],
                         demand_curve: Dict[str, float]) -> Dict[str, float]:
        """Find supply/demand equilibrium point"""

        # Simplified equilibrium calculation
        # In practice, this would solve the system of equations

        # Assume average values for external factors
        avg_vendor_count = 10
        avg_supply_score = 0.5
        avg_demand_score = 0.5

        # Find intersection of supply and demand curves
        # Supply: listings = supply_price_coef * price + supply_intercept
        # Demand: volume = demand_price_coef * price + demand_intercept

        # Set supply = demand and solve for price
        a = supply_curve['price_coefficient'] - demand_curve['price_coefficient']
        b = demand_curve['intercept'] - supply_curve['intercept']

        if abs(a) < 1e-10:  # Parallel lines
            equilibrium_price = 100.0  # Default
        else:
            equilibrium_price = b / a

        # Calculate equilibrium quantity
        equilibrium_quantity = (supply_curve['price_coefficient'] * equilibrium_price +
                              supply_curve['intercept'])

        return {
            'price': max(1.0, equilibrium_price),  # Ensure positive price
            'quantity': max(1, equilibrium_quantity)  # Ensure positive quantity
        }

    def _calculate_supply_elasticity(self, df: pd.DataFrame) -> float:
        """Calculate price elasticity of supply"""

        if len(df) < 10:
            return 0.0

        # Calculate percentage changes
        price_changes = df['price'].pct_change().dropna()
        quantity_changes = df['listings_count'].pct_change().dropna()

        # Align the series
        min_len = min(len(price_changes), len(quantity_changes))
        price_changes = price_changes.iloc[-min_len:]
        quantity_changes = quantity_changes.iloc[-min_len:]

        # Remove zero price changes to avoid division by zero
        non_zero_mask = abs(price_changes) > 1e-6
        price_changes = price_changes[non_zero_mask]
        quantity_changes = quantity_changes[non_zero_mask]

        if len(price_changes) == 0:
            return 0.0

        # Elasticity = % change in quantity / % change in price
        elasticities = quantity_changes / price_changes
        elasticities = elasticities.replace([np.inf, -np.inf], np.nan).dropna()

        return float(np.median(elasticities)) if len(elasticities) > 0 else 0.0

    def _calculate_demand_elasticity(self, df: pd.DataFrame) -> float:
        """Calculate price elasticity of demand"""

        if len(df) < 10:
            return 0.0

        # Calculate percentage changes
        price_changes = df['price'].pct_change().dropna()
        quantity_changes = df['volume'].pct_change().dropna()

        # Align the series
        min_len = min(len(price_changes), len(quantity_changes))
        price_changes = price_changes.iloc[-min_len:]
        quantity_changes = quantity_changes.iloc[-min_len:]

        # Remove zero price changes
        non_zero_mask = abs(price_changes) > 1e-6
        price_changes = price_changes[non_zero_mask]
        quantity_changes = quantity_changes[non_zero_mask]

        if len(price_changes) == 0:
            return 0.0

        elasticities = quantity_changes / price_changes
        elasticities = elasticities.replace([np.inf, -np.inf], np.nan).dropna()

        return float(np.median(elasticities)) if len(elasticities) > 0 else 0.0

    def _calculate_model_fit(self, df: pd.DataFrame,
                           supply_curve: Dict[str, float],
                           demand_curve: Dict[str, float]) -> float:
        """Calculate model fit (R-squared)"""

        # Simplified R-squared calculation
        return min(supply_curve.get('score', 0.0), demand_curve.get('score', 0.0))

    def _validate_predictions(self, df: pd.DataFrame,
                            supply_curve: Dict[str, float],
                            demand_curve: Dict[str, float]) -> float:
        """Validate model predictions"""

        # Use last 20% of data for validation
        validation_size = max(1, len(df) // 5)
        validation_data = df.tail(validation_size)

        if len(validation_data) == 0:
            return 0.0

        # Predict supply
        predicted_supply = (supply_curve['price_coefficient'] * validation_data['price'] +
                          supply_curve['intercept'])

        # Calculate accuracy
        actual_supply = validation_data['listings_count']
        supply_mape = np.mean(np.abs((actual_supply - predicted_supply) / actual_supply)) * 100

        return max(0.0, 100.0 - supply_mape) / 100.0  # Convert to 0-1 scale


class VolatilityAnalyzer:
    """Analyzes price and volume volatility"""

    def __init__(self):
        self.lookback_periods = [7, 14, 30, 60]  # Days

    async def analyze_volatility(self, market_data: List[MarketDataPoint],
                               marketplace: str, category: str) -> PriceVolatilityAnalysis:
        """Comprehensive volatility analysis"""

        if len(market_data) < 30:
            raise ValueError("Insufficient data for volatility analysis")

        logger.info(f"Analyzing volatility for {marketplace}/{category}")

        # Prepare data
        df = self._prepare_volatility_data(market_data)

        # Calculate volatility metrics
        price_volatility = self._calculate_price_volatility(df)
        volume_volatility = self._calculate_volume_volatility(df)

        # Detect volatility trend
        volatility_trend = self._detect_volatility_trend(df)

        # Identify volatility clusters
        volatility_clusters = self._identify_volatility_clusters(df)

        # Detect regime changes
        regime_changes = self._detect_regime_changes(df)

        # Fit GARCH model
        garch_params = self._fit_garch_model(df)

        # Calculate risk metrics
        var_95 = self._calculate_var(df, confidence_level=0.95)
        expected_shortfall = self._calculate_expected_shortfall(df, confidence_level=0.95)

        analysis = PriceVolatilityAnalysis(
            analysis_id=str(uuid.uuid4()),
            marketplace=marketplace,
            category=category,
            time_period=(min(d.timestamp for d in market_data),
                        max(d.timestamp for d in market_data)),
            price_volatility=price_volatility,
            volume_volatility=volume_volatility,
            volatility_trend=volatility_trend,
            volatility_clusters=volatility_clusters,
            regime_changes=regime_changes,
            garch_parameters=garch_params,
            var_95=var_95,
            expected_shortfall=expected_shortfall
        )

        return analysis

    def _prepare_volatility_data(self, market_data: List[MarketDataPoint]) -> pd.DataFrame:
        """Prepare data for volatility analysis"""

        data = []
        for point in market_data:
            data.append({
                'timestamp': point.timestamp,
                'price': float(point.price),
                'volume': point.volume,
                'listings_count': point.listings_count
            })

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        # Calculate returns
        df['price_return'] = df['price'].pct_change()
        df['volume_return'] = df['volume'].pct_change()

        # Calculate rolling volatilities
        for period in self.lookback_periods:
            df[f'volatility_{period}d'] = df['price_return'].rolling(window=period).std()

        return df.dropna()

    def _calculate_price_volatility(self, df: pd.DataFrame) -> float:
        """Calculate overall price volatility"""

        returns = df['price_return'].dropna()
        if len(returns) == 0:
            return 0.0

        # Annualized volatility
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(365)

        return float(annualized_vol)

    def _calculate_volume_volatility(self, df: pd.DataFrame) -> float:
        """Calculate volume volatility"""

        volume_returns = df['volume_return'].dropna()
        if len(volume_returns) == 0:
            return 0.0

        return float(volume_returns.std())

    def _detect_volatility_trend(self, df: pd.DataFrame) -> str:
        """Detect volatility trend"""

        if 'volatility_30d' not in df.columns:
            return 'stable'

        recent_vol = df['volatility_30d'].tail(10).mean()
        historical_vol = df['volatility_30d'].head(-10).mean()

        if pd.isna(recent_vol) or pd.isna(historical_vol):
            return 'stable'

        change_ratio = recent_vol / historical_vol

        if change_ratio > 1.2:
            return 'increasing'
        elif change_ratio < 0.8:
            return 'decreasing'
        else:
            return 'stable'

    def _identify_volatility_clusters(self, df: pd.DataFrame) -> List[Tuple[datetime, datetime]]:
        """Identify periods of high volatility clustering"""

        if 'volatility_7d' not in df.columns:
            return []

        vol_series = df['volatility_7d'].dropna()
        if len(vol_series) == 0:
            return []

        # Define high volatility threshold (90th percentile)
        threshold = vol_series.quantile(0.9)

        # Find high volatility periods
        high_vol_mask = vol_series > threshold
        clusters = []

        cluster_start = None
        for timestamp, is_high_vol in high_vol_mask.items():
            if is_high_vol and cluster_start is None:
                cluster_start = timestamp
            elif not is_high_vol and cluster_start is not None:
                clusters.append((cluster_start, timestamp))
                cluster_start = None

        # Handle case where cluster extends to end of data
        if cluster_start is not None:
            clusters.append((cluster_start, vol_series.index[-1]))

        return clusters

    def _detect_regime_changes(self, df: pd.DataFrame) -> List[datetime]:
        """Detect volatility regime changes"""

        if len(df) < 50:
            return []

        returns = df['price_return'].dropna()
        if len(returns) < 30:
            return []

        # Use rolling window to detect changes in volatility regime
        window = 20
        vol_series = returns.rolling(window=window).std()

        # Detect significant changes
        vol_changes = vol_series.pct_change()
        threshold = vol_changes.std() * 2  # 2 standard deviations

        regime_changes = []
        for timestamp, change in vol_changes.items():
            if abs(change) > threshold:
                regime_changes.append(timestamp)

        return regime_changes

    def _fit_garch_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """Fit GARCH model for volatility forecasting"""

        returns = df['price_return'].dropna()
        if len(returns) < 50:
            return {'alpha': 0.0, 'beta': 0.0, 'omega': 0.0}

        try:
            # Simplified GARCH(1,1) parameters
            # In production, would use proper GARCH fitting library

            # Calculate sample parameters
            returns_squared = returns ** 2

            # Simple correlation-based estimation
            alpha = 0.1  # Typically 0.05-0.15
            beta = 0.85  # Typically 0.8-0.95
            omega = returns_squared.mean() * (1 - alpha - beta)

            return {
                'alpha': float(alpha),
                'beta': float(beta),
                'omega': float(omega),
                'persistence': float(alpha + beta)
            }

        except Exception as e:
            logger.warning(f"GARCH model fitting failed: {e}")
            return {'alpha': 0.0, 'beta': 0.0, 'omega': 0.0, 'persistence': 0.0}

    def _calculate_var(self, df: pd.DataFrame, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""

        returns = df['price_return'].dropna()
        if len(returns) == 0:
            return 0.0

        # Historical VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)

        return float(abs(var))

    def _calculate_expected_shortfall(self, df: pd.DataFrame,
                                    confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""

        returns = df['price_return'].dropna()
        if len(returns) == 0:
            return 0.0

        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, var_percentile)

        # Expected shortfall is the mean of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]
        if len(tail_returns) == 0:
            return 0.0

        expected_shortfall = tail_returns.mean()
        return float(abs(expected_shortfall))


class ManipulationDetector:
    """Detects market manipulation patterns"""

    def __init__(self):
        self.detection_methods = {
            'pump_dump': self._detect_pump_dump,
            'wash_trading': self._detect_wash_trading,
            'cornering': self._detect_market_cornering,
            'spoofing': self._detect_spoofing
        }

    async def detect_manipulation(self, market_data: List[MarketDataPoint],
                                transaction_data: List[Dict[str, Any]]) -> List[MarketManipulationAlert]:
        """Detect various types of market manipulation"""

        alerts = []

        for manipulation_type, detector in self.detection_methods.items():
            try:
                alert = await detector(market_data, transaction_data)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                logger.warning(f"Error detecting {manipulation_type}: {e}")

        return alerts

    async def _detect_pump_dump(self, market_data: List[MarketDataPoint],
                              transaction_data: List[Dict[str, Any]]) -> Optional[MarketManipulationAlert]:
        """Detect pump and dump schemes"""

        if len(market_data) < 20:
            return None

        # Prepare price data
        prices = [float(d.price) for d in market_data]
        volumes = [d.volume for d in market_data]
        timestamps = [d.timestamp for d in market_data]

        # Look for rapid price increases followed by rapid decreases
        price_changes = np.diff(prices) / prices[:-1]

        # Find peaks in price
        peaks, _ = find_peaks(prices, height=np.mean(prices) * 1.2)

        for peak_idx in peaks:
            if peak_idx < 5 or peak_idx >= len(prices) - 5:
                continue

            # Check for rapid increase before peak
            pre_peak_changes = price_changes[max(0, peak_idx-5):peak_idx]
            post_peak_changes = price_changes[peak_idx:min(len(price_changes), peak_idx+5)]

            pre_peak_increase = np.sum(pre_peak_changes)
            post_peak_decrease = np.sum(post_peak_changes)

            # Check for volume spike
            peak_volume = volumes[peak_idx]
            avg_volume = np.mean(volumes)

            if (pre_peak_increase > 0.5 and  # 50% increase
                post_peak_decrease < -0.3 and  # 30% decrease
                peak_volume > avg_volume * 2):  # 2x volume spike

                return MarketManipulationAlert(
                    alert_id=str(uuid.uuid4()),
                    marketplace=market_data[0].marketplace,
                    category=market_data[0].category,
                    manipulation_type='pump_dump',
                    confidence_score=0.8,
                    detection_timestamp=timestamps[peak_idx],
                    evidence={
                        'pre_peak_increase': pre_peak_increase,
                        'post_peak_decrease': post_peak_decrease,
                        'volume_spike': peak_volume / avg_volume,
                        'peak_timestamp': timestamps[peak_idx].isoformat()
                    },
                    affected_products=set(),
                    affected_vendors=set(),
                    estimated_impact={'price_manipulation': pre_peak_increase + abs(post_peak_decrease)},
                    status='active'
                )

        return None

    async def _detect_wash_trading(self, market_data: List[MarketDataPoint],
                                 transaction_data: List[Dict[str, Any]]) -> Optional[MarketManipulationAlert]:
        """Detect wash trading patterns"""

        if len(transaction_data) < 50:
            return None

        # Analyze transaction patterns for wash trading indicators
        vendor_volumes = defaultdict(float)
        buyer_volumes = defaultdict(float)
        cross_trading = defaultdict(float)

        for tx in transaction_data:
            vendor_id = tx.get('vendor_id', '')
            buyer_id = tx.get('buyer_id', '')
            amount = float(tx.get('amount', 0))

            vendor_volumes[vendor_id] += amount
            buyer_volumes[buyer_id] += amount

            # Check for cross-trading (vendor also buying)
            if vendor_id in buyer_volumes and buyer_id in vendor_volumes:
                cross_trading[f"{vendor_id}_{buyer_id}"] += amount

        # Detect suspicious patterns
        total_volume = sum(vendor_volumes.values())
        cross_volume = sum(cross_trading.values())

        if cross_volume > total_volume * 0.1:  # More than 10% cross-trading
            return MarketManipulationAlert(
                alert_id=str(uuid.uuid4()),
                marketplace=market_data[0].marketplace if market_data else 'unknown',
                category=market_data[0].category if market_data else 'unknown',
                manipulation_type='wash_trading',
                confidence_score=cross_volume / total_volume,
                detection_timestamp=datetime.now(),
                evidence={
                    'cross_trading_volume': cross_volume,
                    'total_volume': total_volume,
                    'cross_trading_ratio': cross_volume / total_volume,
                    'suspicious_pairs': len(cross_trading)
                },
                affected_products=set(),
                affected_vendors=set(cross_trading.keys()),
                estimated_impact={'volume_inflation': cross_volume / total_volume},
                status='active'
            )

        return None

    async def _detect_market_cornering(self, market_data: List[MarketDataPoint],
                                     transaction_data: List[Dict[str, Any]]) -> Optional[MarketManipulationAlert]:
        """Detect market cornering attempts"""

        if len(transaction_data) < 20:
            return None

        # Analyze vendor concentration
        vendor_volumes = defaultdict(float)
        for tx in transaction_data:
            vendor_id = tx.get('vendor_id', '')
            amount = float(tx.get('amount', 0))
            vendor_volumes[vendor_id] += amount

        if not vendor_volumes:
            return None

        total_volume = sum(vendor_volumes.values())
        sorted_vendors = sorted(vendor_volumes.items(), key=lambda x: x[1], reverse=True)

        # Check for market concentration
        top_vendor_share = sorted_vendors[0][1] / total_volume
        top_3_share = sum(v[1] for v in sorted_vendors[:3]) / total_volume

        if top_vendor_share > 0.6 or top_3_share > 0.8:
            return MarketManipulationAlert(
                alert_id=str(uuid.uuid4()),
                marketplace=market_data[0].marketplace if market_data else 'unknown',
                category=market_data[0].category if market_data else 'unknown',
                manipulation_type='cornering',
                confidence_score=min(top_vendor_share, top_3_share),
                detection_timestamp=datetime.now(),
                evidence={
                    'top_vendor_share': top_vendor_share,
                    'top_3_share': top_3_share,
                    'dominant_vendor': sorted_vendors[0][0],
                    'market_concentration': len(vendor_volumes)
                },
                affected_products=set(),
                affected_vendors={sorted_vendors[0][0]},
                estimated_impact={'market_concentration': top_vendor_share},
                status='active'
            )

        return None

    async def _detect_spoofing(self, market_data: List[MarketDataPoint],
                             transaction_data: List[Dict[str, Any]]) -> Optional[MarketManipulationAlert]:
        """Detect spoofing patterns"""

        # Spoofing detection would require order book data
        # This is a placeholder implementation
        return None


class EconomicForecaster:
    """Economic trend forecasting using machine learning"""

    def __init__(self):
        self.models = {
            'price': RandomForestRegressor(n_estimators=100, random_state=42),
            'volume': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.scalers = {
            'price': StandardScaler(),
            'volume': StandardScaler()
        }

    async def generate_forecast(self, market_data: List[MarketDataPoint],
                              forecast_horizon_days: int = 30) -> EconomicForecast:
        """Generate economic forecast"""

        if len(market_data) < 50:
            raise ValueError("Insufficient data for forecasting")

        logger.info(f"Generating {forecast_horizon_days}-day forecast")

        # Prepare data
        df = self._prepare_forecast_data(market_data)

        # Generate price forecast
        price_forecast = await self._forecast_prices(df, forecast_horizon_days)

        # Generate volume forecast
        volume_forecast = await self._forecast_volume(df, forecast_horizon_days)

        # Determine trend direction
        trend_direction = self._determine_trend_direction(price_forecast)

        # Calculate forecast accuracy
        forecast_accuracy = self._calculate_forecast_accuracy(df)

        forecast = EconomicForecast(
            forecast_id=str(uuid.uuid4()),
            marketplace=market_data[0].marketplace,
            category=market_data[0].category,
            forecast_horizon_days=forecast_horizon_days,
            price_forecast=price_forecast,
            volume_forecast=volume_forecast,
            trend_direction=trend_direction,
            forecast_accuracy=forecast_accuracy,
            model_type='random_forest',
            generated_at=datetime.now()
        )

        return forecast

    def _prepare_forecast_data(self, market_data: List[MarketDataPoint]) -> pd.DataFrame:
        """Prepare data for forecasting"""

        data = []
        for point in market_data:
            data.append({
                'timestamp': point.timestamp,
                'price': float(point.price),
                'volume': point.volume,
                'listings_count': point.listings_count,
                'vendor_count': point.vendor_count
            })

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        # Feature engineering
        df['price_ma_7'] = df['price'].rolling(window=7).mean()
        df['price_ma_14'] = df['price'].rolling(window=14).mean()
        df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
        df['price_volatility'] = df['price'].rolling(window=7).std()
        df['price_momentum'] = df['price'].pct_change(periods=7)

        # Lag features
        for lag in [1, 3, 7]:
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

        return df.dropna()

    async def _forecast_prices(self, df: pd.DataFrame,
                             horizon_days: int) -> List[Tuple[datetime, float, float]]:
        """Forecast prices using machine learning"""

        # Prepare features and target
        feature_cols = ['price_ma_7', 'price_ma_14', 'volume_ma_7', 'price_volatility',
                       'price_momentum', 'listings_count', 'vendor_count'] + \
                      [f'price_lag_{lag}' for lag in [1, 3, 7]]

        X = df[feature_cols].dropna()
        y = df['price'].loc[X.index]

        if len(X) < 20:
            # Return simple trend extrapolation
            last_price = df['price'].iloc[-1]
            trend = df['price'].pct_change().tail(7).mean()

            forecast = []
            current_price = last_price
            for i in range(horizon_days):
                current_price *= (1 + trend)
                future_date = df.index[-1] + timedelta(days=i+1)
                confidence_interval = current_price * 0.1  # 10% confidence interval
                forecast.append((future_date, float(current_price), float(confidence_interval)))

            return forecast

        # Train model
        X_scaled = self.scalers['price'].fit_transform(X)
        self.models['price'].fit(X_scaled, y)

        # Generate forecasts
        forecast = []
        last_features = X.iloc[-1].copy()

        for i in range(horizon_days):
            # Predict next price
            features_scaled = self.scalers['price'].transform([last_features])
            predicted_price = self.models['price'].predict(features_scaled)[0]

            # Calculate confidence interval (simplified)
            prediction_std = np.std(y - self.models['price'].predict(X_scaled))
            confidence_interval = prediction_std * 1.96  # 95% confidence

            future_date = df.index[-1] + timedelta(days=i+1)
            forecast.append((future_date, float(predicted_price), float(confidence_interval)))

            # Update features for next prediction
            last_features = self._update_features_for_next_step(last_features, predicted_price)

        return forecast

    async def _forecast_volume(self, df: pd.DataFrame,
                             horizon_days: int) -> List[Tuple[datetime, int, float]]:
        """Forecast volume using machine learning"""

        # Similar to price forecasting but for volume
        feature_cols = ['volume_ma_7', 'price_ma_7', 'listings_count', 'vendor_count'] + \
                      [f'volume_lag_{lag}' for lag in [1, 3, 7]]

        X = df[feature_cols].dropna()
        y = df['volume'].loc[X.index]

        if len(X) < 20:
            # Return simple trend extrapolation
            last_volume = df['volume'].iloc[-1]
            trend = df['volume'].pct_change().tail(7).mean()

            forecast = []
            current_volume = last_volume
            for i in range(horizon_days):
                current_volume *= (1 + trend)
                future_date = df.index[-1] + timedelta(days=i+1)
                confidence_interval = current_volume * 0.15  # 15% confidence interval
                forecast.append((future_date, int(max(0, current_volume)), float(confidence_interval)))

            return forecast

        # Train model
        X_scaled = self.scalers['volume'].fit_transform(X)
        self.models['volume'].fit(X_scaled, y)

        # Generate forecasts
        forecast = []
        last_features = X.iloc[-1].copy()

        for i in range(horizon_days):
            features_scaled = self.scalers['volume'].transform([last_features])
            predicted_volume = self.models['volume'].predict(features_scaled)[0]

            prediction_std = np.std(y - self.models['volume'].predict(X_scaled))
            confidence_interval = prediction_std * 1.96

            future_date = df.index[-1] + timedelta(days=i+1)
            forecast.append((future_date, int(max(0, predicted_volume)), float(confidence_interval)))

            # Update features for next prediction
            last_features = self._update_volume_features_for_next_step(last_features, predicted_volume)

        return forecast

    def _update_features_for_next_step(self, features: pd.Series, predicted_price: float) -> pd.Series:
        """Update features for next prediction step"""

        updated_features = features.copy()

        # Update lag features
        if 'price_lag_3' in features:
            updated_features['price_lag_3'] = features.get('price_lag_1', predicted_price)
        if 'price_lag_1' in features:
            updated_features['price_lag_1'] = predicted_price

        return updated_features

    def _update_volume_features_for_next_step(self, features: pd.Series, predicted_volume: float) -> pd.Series:
        """Update volume features for next prediction step"""

        updated_features = features.copy()

        # Update lag features
        if 'volume_lag_3' in features:
            updated_features['volume_lag_3'] = features.get('volume_lag_1', predicted_volume)
        if 'volume_lag_1' in features:
            updated_features['volume_lag_1'] = predicted_volume

        return updated_features

    def _determine_trend_direction(self, price_forecast: List[Tuple[datetime, float, float]]) -> str:
        """Determine overall trend direction"""

        if len(price_forecast) < 2:
            return 'sideways'

        first_price = price_forecast[0][1]
        last_price = price_forecast[-1][1]

        price_change = (last_price - first_price) / first_price

        if price_change > 0.05:  # 5% increase
            return 'bullish'
        elif price_change < -0.05:  # 5% decrease
            return 'bearish'
        else:
            return 'sideways'

    def _calculate_forecast_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate historical forecast accuracy"""

        if len(df) < 30:
            return 0.5  # Default accuracy

        # Use last 20% of data for accuracy calculation
        test_size = len(df) // 5
        train_data = df.iloc[:-test_size]
        test_data = df.iloc[-test_size:]

        try:
            # Simple accuracy calculation using moving average
            predictions = train_data['price'].rolling(window=7).mean().iloc[-test_size:]
            actual = test_data['price']

            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            accuracy = max(0.0, (100 - mape) / 100)

            return float(accuracy)

        except Exception:
            return 0.5


class ArbitrageDetector:
    """Detects arbitrage opportunities across marketplaces"""

    def __init__(self):
        self.price_threshold = 0.05  # 5% minimum price difference
        self.time_window_minutes = 60  # 1-hour time window

    async def detect_arbitrage_opportunities(self,
                                           multi_market_data: Dict[str, List[MarketDataPoint]]) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities across marketplaces"""

        opportunities = []

        marketplaces = list(multi_market_data.keys())

        # Compare prices across all marketplace pairs
        for i, marketplace1 in enumerate(marketplaces):
            for marketplace2 in marketplaces[i+1:]:

                market1_data = multi_market_data[marketplace1]
                market2_data = multi_market_data[marketplace2]

                # Find opportunities between these marketplaces
                pair_opportunities = await self._find_pair_opportunities(
                    marketplace1, market1_data,
                    marketplace2, market2_data
                )

                opportunities.extend(pair_opportunities)

        return opportunities

    async def _find_pair_opportunities(self, marketplace1: str, data1: List[MarketDataPoint],
                                     marketplace2: str, data2: List[MarketDataPoint]) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities between two marketplaces"""

        opportunities = []

        # Group data by category
        categories1 = defaultdict(list)
        categories2 = defaultdict(list)

        for point in data1:
            categories1[point.category].append(point)

        for point in data2:
            categories2[point.category].append(point)

        # Find common categories
        common_categories = set(categories1.keys()) & set(categories2.keys())

        for category in common_categories:
            cat_data1 = categories1[category]
            cat_data2 = categories2[category]

            # Get recent prices
            recent_data1 = [p for p in cat_data1 if
                          (datetime.now() - p.timestamp).total_seconds() < self.time_window_minutes * 60]
            recent_data2 = [p for p in cat_data2 if
                          (datetime.now() - p.timestamp).total_seconds() < self.time_window_minutes * 60]

            if not recent_data1 or not recent_data2:
                continue

            # Calculate average prices
            avg_price1 = np.mean([float(p.price) for p in recent_data1])
            avg_price2 = np.mean([float(p.price) for p in recent_data2])

            # Check for arbitrage opportunity
            price_diff = abs(avg_price1 - avg_price2)
            price_diff_pct = price_diff / min(avg_price1, avg_price2)

            if price_diff_pct > self.price_threshold:
                source_marketplace = marketplace1 if avg_price1 < avg_price2 else marketplace2
                target_marketplace = marketplace2 if avg_price1 < avg_price2 else marketplace1
                source_price = min(avg_price1, avg_price2)
                target_price = max(avg_price1, avg_price2)

                # Calculate risk score
                risk_score = self._calculate_arbitrage_risk(
                    recent_data1, recent_data2, price_diff_pct
                )

                opportunity = ArbitrageOpportunity(
                    opportunity_id=str(uuid.uuid4()),
                    product_category=category,
                    source_marketplace=source_marketplace,
                    target_marketplace=target_marketplace,
                    source_price=Decimal(str(source_price)),
                    target_price=Decimal(str(target_price)),
                    price_difference=Decimal(str(price_diff)),
                    price_difference_percentage=price_diff_pct * 100,
                    potential_profit=Decimal(str(price_diff * 0.8)),  # Assume 20% costs
                    risk_score=risk_score,
                    time_window_minutes=self.time_window_minutes,
                    detected_at=datetime.now(),
                    status='active'
                )

                opportunities.append(opportunity)

        return opportunities

    def _calculate_arbitrage_risk(self, data1: List[MarketDataPoint],
                                data2: List[MarketDataPoint],
                                price_diff_pct: float) -> float:
        """Calculate risk score for arbitrage opportunity"""

        risk_score = 0.0

        # Price volatility risk
        prices1 = [float(p.price) for p in data1]
        prices2 = [float(p.price) for p in data2]

        vol1 = np.std(prices1) / np.mean(prices1) if prices1 else 0
        vol2 = np.std(prices2) / np.mean(prices2) if prices2 else 0

        avg_volatility = (vol1 + vol2) / 2
        risk_score += min(0.5, avg_volatility * 2)  # Cap at 0.5

        # Liquidity risk
        avg_volume1 = np.mean([p.volume for p in data1]) if data1 else 0
        avg_volume2 = np.mean([p.volume for p in data2]) if data2 else 0

        min_volume = min(avg_volume1, avg_volume2)
        if min_volume < 10:  # Low liquidity
            risk_score += 0.3

        # Excessive price difference risk (may indicate data error)
        if price_diff_pct > 0.5:  # 50% difference seems suspicious
            risk_score += 0.2

        return min(1.0, risk_score)


class MarketEconomicsProcessor:
    """Main market economics processing engine"""

    def __init__(self, db_config: Dict[str, Any], redis_config: Dict[str, Any],
                 kafka_config: Dict[str, Any]):

        self.db_config = db_config
        self.redis_config = redis_config
        self.kafka_config = kafka_config

        # Core components
        self.supply_demand_analyzer = SupplyDemandAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.manipulation_detector = ManipulationDetector()
        self.economic_forecaster = EconomicForecaster()
        self.arbitrage_detector = ArbitrageDetector()

        # Data storage
        self.db_pool = None
        self.redis_client = None
        self.kafka_producer = None

    async def initialize(self):
        """Initialize all components"""

        logger.info("Initializing Market Economics Processor...")

        # Database connection
        self.db_pool = await asyncpg.create_pool(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            min_size=5,
            max_size=20
        )

        # Redis connection
        self.redis_client = aioredis.from_url(
            f"redis://{self.redis_config['host']}:{self.redis_config['port']}/3"
        )

        # Kafka producer
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_config['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )
        await self.kafka_producer.start()

        # Initialize database tables
        await self._initialize_database_tables()

        logger.info("Market Economics Processor initialized successfully")

    async def _initialize_database_tables(self):
        """Create database tables for economics analysis"""

        tables = [
            '''
            CREATE TABLE IF NOT EXISTS supply_demand_models (
                model_id VARCHAR(255) PRIMARY KEY,
                marketplace VARCHAR(100),
                category VARCHAR(100),
                time_start TIMESTAMP,
                time_end TIMESTAMP,
                supply_elasticity FLOAT,
                demand_elasticity FLOAT,
                equilibrium_price DECIMAL(15,2),
                equilibrium_quantity INTEGER,
                supply_curve_params JSONB,
                demand_curve_params JSONB,
                r_squared FLOAT,
                prediction_accuracy FLOAT,
                last_updated TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW()
            );
            ''',
            '''
            CREATE TABLE IF NOT EXISTS volatility_analysis (
                analysis_id VARCHAR(255) PRIMARY KEY,
                marketplace VARCHAR(100),
                category VARCHAR(100),
                time_start TIMESTAMP,
                time_end TIMESTAMP,
                price_volatility FLOAT,
                volume_volatility FLOAT,
                volatility_trend VARCHAR(20),
                volatility_clusters JSONB,
                regime_changes JSONB,
                garch_parameters JSONB,
                var_95 FLOAT,
                expected_shortfall FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            );
            ''',
            '''
            CREATE TABLE IF NOT EXISTS manipulation_alerts (
                alert_id VARCHAR(255) PRIMARY KEY,
                marketplace VARCHAR(100),
                category VARCHAR(100),
                manipulation_type VARCHAR(50),
                confidence_score FLOAT,
                detection_timestamp TIMESTAMP,
                evidence JSONB,
                affected_products TEXT[],
                affected_vendors TEXT[],
                estimated_impact JSONB,
                status VARCHAR(20),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            ''',
            '''
            CREATE TABLE IF NOT EXISTS economic_forecasts (
                forecast_id VARCHAR(255) PRIMARY KEY,
                marketplace VARCHAR(100),
                category VARCHAR(100),
                forecast_horizon_days INTEGER,
                price_forecast JSONB,
                volume_forecast JSONB,
                trend_direction VARCHAR(20),
                forecast_accuracy FLOAT,
                model_type VARCHAR(50),
                generated_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW()
            );
            ''',
            '''
            CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
                opportunity_id VARCHAR(255) PRIMARY KEY,
                product_category VARCHAR(100),
                source_marketplace VARCHAR(100),
                target_marketplace VARCHAR(100),
                source_price DECIMAL(15,2),
                target_price DECIMAL(15,2),
                price_difference DECIMAL(15,2),
                price_difference_percentage FLOAT,
                potential_profit DECIMAL(15,2),
                risk_score FLOAT,
                time_window_minutes INTEGER,
                detected_at TIMESTAMP,
                status VARCHAR(20),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            ''',
            '''
            CREATE INDEX IF NOT EXISTS idx_supply_demand_marketplace ON supply_demand_models(marketplace);
            CREATE INDEX IF NOT EXISTS idx_volatility_marketplace ON volatility_analysis(marketplace);
            CREATE INDEX IF NOT EXISTS idx_manipulation_alerts_type ON manipulation_alerts(manipulation_type);
            CREATE INDEX IF NOT EXISTS idx_economic_forecasts_marketplace ON economic_forecasts(marketplace);
            CREATE INDEX IF NOT EXISTS idx_arbitrage_opportunities_status ON arbitrage_opportunities(status);
            '''
        ]

        async with self.db_pool.acquire() as conn:
            for table_sql in tables:
                await conn.execute(table_sql)

    async def process_market_economics(self, marketplace: str, category: str,
                                     market_data: List[MarketDataPoint]) -> Dict[str, Any]:
        """Comprehensive market economics analysis"""

        try:
            logger.info(f"Processing market economics for {marketplace}/{category}")

            results = {
                'marketplace': marketplace,
                'category': category,
                'analysis_timestamp': datetime.now(),
                'supply_demand': None,
                'volatility': None,
                'manipulation_alerts': [],
                'forecast': None,
                'arbitrage_opportunities': []
            }

            # Supply/demand analysis
            if len(market_data) >= 20:
                supply_demand_model = await self.supply_demand_analyzer.analyze_supply_demand(
                    market_data, marketplace, category
                )
                results['supply_demand'] = asdict(supply_demand_model)
                await self._store_supply_demand_model(supply_demand_model)

            # Volatility analysis
            if len(market_data) >= 30:
                volatility_analysis = await self.volatility_analyzer.analyze_volatility(
                    market_data, marketplace, category
                )
                results['volatility'] = asdict(volatility_analysis)
                await self._store_volatility_analysis(volatility_analysis)

            # Manipulation detection
            manipulation_alerts = await self.manipulation_detector.detect_manipulation(
                market_data, []  # Would pass transaction data
            )
            results['manipulation_alerts'] = [asdict(alert) for alert in manipulation_alerts]

            for alert in manipulation_alerts:
                await self._store_manipulation_alert(alert)

            # Economic forecasting
            if len(market_data) >= 50:
                forecast = await self.economic_forecaster.generate_forecast(market_data)
                results['forecast'] = asdict(forecast)
                await self._store_economic_forecast(forecast)

            # Publish results to Kafka
            await self.kafka_producer.send(
                'market_economics_analysis',
                key=f"{marketplace}_{category}",
                value=results
            )

            logger.info(f"Completed market economics analysis for {marketplace}/{category}")
            return results

        except Exception as e:
            logger.error(f"Error processing market economics: {e}")
            raise

    async def detect_cross_market_arbitrage(self,
                                          multi_market_data: Dict[str, List[MarketDataPoint]]) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities across marketplaces"""

        opportunities = await self.arbitrage_detector.detect_arbitrage_opportunities(
            multi_market_data
        )

        # Store opportunities
        for opportunity in opportunities:
            await self._store_arbitrage_opportunity(opportunity)

        # Publish to Kafka
        if opportunities:
            await self.kafka_producer.send(
                'arbitrage_opportunities',
                value={
                    'timestamp': datetime.now(),
                    'opportunities': [asdict(opp) for opp in opportunities]
                }
            )

        return opportunities

    async def _store_supply_demand_model(self, model: SupplyDemandModel):
        """Store supply/demand model in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO supply_demand_models
                (model_id, marketplace, category, time_start, time_end,
                 supply_elasticity, demand_elasticity, equilibrium_price,
                 equilibrium_quantity, supply_curve_params, demand_curve_params,
                 r_squared, prediction_accuracy, last_updated)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ''',
                model.model_id, model.marketplace, model.category,
                model.time_period[0], model.time_period[1],
                model.supply_elasticity, model.demand_elasticity,
                model.equilibrium_price, model.equilibrium_quantity,
                json.dumps(model.supply_curve_params),
                json.dumps(model.demand_curve_params),
                model.r_squared, model.prediction_accuracy, model.last_updated
            )

    async def _store_volatility_analysis(self, analysis: PriceVolatilityAnalysis):
        """Store volatility analysis in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO volatility_analysis
                (analysis_id, marketplace, category, time_start, time_end,
                 price_volatility, volume_volatility, volatility_trend,
                 volatility_clusters, regime_changes, garch_parameters,
                 var_95, expected_shortfall)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ''',
                analysis.analysis_id, analysis.marketplace, analysis.category,
                analysis.time_period[0], analysis.time_period[1],
                analysis.price_volatility, analysis.volume_volatility,
                analysis.volatility_trend,
                json.dumps([(c[0].isoformat(), c[1].isoformat()) for c in analysis.volatility_clusters]),
                json.dumps([r.isoformat() for r in analysis.regime_changes]),
                json.dumps(analysis.garch_parameters),
                analysis.var_95, analysis.expected_shortfall
            )

    async def _store_manipulation_alert(self, alert: MarketManipulationAlert):
        """Store manipulation alert in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO manipulation_alerts
                (alert_id, marketplace, category, manipulation_type,
                 confidence_score, detection_timestamp, evidence,
                 affected_products, affected_vendors, estimated_impact, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ''',
                alert.alert_id, alert.marketplace, alert.category,
                alert.manipulation_type, alert.confidence_score,
                alert.detection_timestamp, json.dumps(alert.evidence),
                list(alert.affected_products), list(alert.affected_vendors),
                json.dumps(alert.estimated_impact), alert.status
            )

    async def _store_economic_forecast(self, forecast: EconomicForecast):
        """Store economic forecast in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO economic_forecasts
                (forecast_id, marketplace, category, forecast_horizon_days,
                 price_forecast, volume_forecast, trend_direction,
                 forecast_accuracy, model_type, generated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ''',
                forecast.forecast_id, forecast.marketplace, forecast.category,
                forecast.forecast_horizon_days,
                json.dumps([(f[0].isoformat(), f[1], f[2]) for f in forecast.price_forecast]),
                json.dumps([(f[0].isoformat(), f[1], f[2]) for f in forecast.volume_forecast]),
                forecast.trend_direction, forecast.forecast_accuracy,
                forecast.model_type, forecast.generated_at
            )

    async def _store_arbitrage_opportunity(self, opportunity: ArbitrageOpportunity):
        """Store arbitrage opportunity in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO arbitrage_opportunities
                (opportunity_id, product_category, source_marketplace,
                 target_marketplace, source_price, target_price,
                 price_difference, price_difference_percentage,
                 potential_profit, risk_score, time_window_minutes,
                 detected_at, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ''',
                opportunity.opportunity_id, opportunity.product_category,
                opportunity.source_marketplace, opportunity.target_marketplace,
                opportunity.source_price, opportunity.target_price,
                opportunity.price_difference, opportunity.price_difference_percentage,
                opportunity.potential_profit, opportunity.risk_score,
                opportunity.time_window_minutes, opportunity.detected_at,
                opportunity.status
            )

    async def get_market_statistics(self) -> Dict[str, Any]:
        """Get comprehensive market statistics"""

        async with self.db_pool.acquire() as conn:
            # Supply/demand statistics
            supply_demand_stats = await conn.fetchrow(
                '''
                SELECT
                    COUNT(*) as total_models,
                    AVG(supply_elasticity) as avg_supply_elasticity,
                    AVG(demand_elasticity) as avg_demand_elasticity,
                    AVG(prediction_accuracy) as avg_accuracy
                FROM supply_demand_models
                '''
            )

            # Volatility statistics
            volatility_stats = await conn.fetchrow(
                '''
                SELECT
                    COUNT(*) as total_analyses,
                    AVG(price_volatility) as avg_price_volatility,
                    AVG(var_95) as avg_var_95
                FROM volatility_analysis
                '''
            )

            # Manipulation statistics
            manipulation_stats = await conn.fetchrow(
                '''
                SELECT
                    COUNT(*) as total_alerts,
                    COUNT(*) FILTER (WHERE status = 'active') as active_alerts,
                    AVG(confidence_score) as avg_confidence
                FROM manipulation_alerts
                '''
            )

            # Arbitrage statistics
            arbitrage_stats = await conn.fetchrow(
                '''
                SELECT
                    COUNT(*) as total_opportunities,
                    COUNT(*) FILTER (WHERE status = 'active') as active_opportunities,
                    AVG(price_difference_percentage) as avg_price_difference_pct,
                    AVG(potential_profit) as avg_potential_profit
                FROM arbitrage_opportunities
                '''
            )

        return {
            'supply_demand': dict(supply_demand_stats) if supply_demand_stats else {},
            'volatility': dict(volatility_stats) if volatility_stats else {},
            'manipulation': dict(manipulation_stats) if manipulation_stats else {},
            'arbitrage': dict(arbitrage_stats) if arbitrage_stats else {}
        }

    async def cleanup(self):
        """Cleanup resources"""

        logger.info("Cleaning up Market Economics Processor...")

        if self.kafka_producer:
            await self.kafka_producer.stop()

        if self.redis_client:
            await self.redis_client.close()

        if self.db_pool:
            await self.db_pool.close()


# Example usage
async def main():
    """Example usage of Market Economics Processor"""

    # Configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'bev_osint',
        'user': 'bev_user',
        'password': os.getenv('DB_PASSWORD', 'dev_password')
    }

    redis_config = {
        'host': 'localhost',
        'port': 6379
    }

    kafka_config = {
        'bootstrap_servers': ['localhost:9092']
    }

    # Initialize processor
    processor = MarketEconomicsProcessor(db_config, redis_config, kafka_config)
    await processor.initialize()

    try:
        # Example market data
        market_data = [
            MarketDataPoint(
                timestamp=datetime.now() - timedelta(days=i),
                marketplace='example_market',
                category='electronics',
                product_id=f'product_{i}',
                price=Decimal(str(100 + np.random.normal(0, 10))),
                currency='USD',
                volume=int(50 + np.random.normal(0, 10)),
                listings_count=int(20 + np.random.normal(0, 5)),
                vendor_count=int(5 + np.random.normal(0, 2)),
                demand_indicators={'search_volume': np.random.random()},
                supply_indicators={'vendor_activity': np.random.random()}
            )
            for i in range(60)  # 60 days of data
        ]

        # Process market economics
        results = await processor.process_market_economics(
            'example_market', 'electronics', market_data
        )

        print(f"Market economics analysis results:")
        print(f"- Supply elasticity: {results.get('supply_demand', {}).get('supply_elasticity', 'N/A')}")
        print(f"- Demand elasticity: {results.get('supply_demand', {}).get('demand_elasticity', 'N/A')}")
        print(f"- Price volatility: {results.get('volatility', {}).get('price_volatility', 'N/A')}")
        print(f"- Manipulation alerts: {len(results.get('manipulation_alerts', []))}")

        # Get statistics
        stats = await processor.get_market_statistics()
        print(f"Processor statistics: {stats}")

    except KeyboardInterrupt:
        logger.info("Shutting down processor...")
    finally:
        await processor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())