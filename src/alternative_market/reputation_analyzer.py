import os
#!/usr/bin/env python3
"""
Vendor Reputation Framework - Phase 7 Alternative Market Intelligence Platform
Place in: /home/starlord/Projects/Bev/src/alternative_market/reputation_analyzer.py

Multi-source reputation aggregation with escrow monitoring and fraud detection.
"""

import asyncio
import json
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import asyncpg
import aioredis
from aiokafka import AIOKafkaProducer
import re
from textblob import TextBlob
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
import uuid
from decimal import Decimal, ROUND_HALF_UP
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VendorReputation:
    """Vendor reputation data structure"""
    vendor_id: str
    vendor_name: str
    marketplace: str
    overall_score: float  # 0.0 to 1.0
    trust_level: str  # 'untrusted', 'low', 'medium', 'high', 'verified'
    total_transactions: int
    successful_transactions: int
    disputed_transactions: int
    total_volume: Decimal
    average_rating: float
    rating_count: int
    response_time_hours: float
    uptime_percentage: float
    escrow_success_rate: float
    reputation_sources: List[str]
    risk_indicators: List[str]
    last_updated: datetime
    reputation_history: List[Dict[str, Any]]

@dataclass
class EscrowTransaction:
    """Escrow transaction monitoring structure"""
    escrow_id: str
    vendor_id: str
    buyer_id: str
    marketplace: str
    amount: Decimal
    currency: str
    product_category: str
    status: str  # 'pending', 'funded', 'disputed', 'released', 'refunded'
    created_at: datetime
    funded_at: Optional[datetime]
    disputed_at: Optional[datetime]
    resolved_at: Optional[datetime]
    dispute_reason: Optional[str]
    resolution_type: Optional[str]  # 'vendor_favor', 'buyer_favor', 'partial_refund'
    arbiter_notes: Optional[str]
    auto_release_hours: int
    fees: Dict[str, Decimal]
    metadata: Dict[str, Any]

@dataclass
class DisputeRecord:
    """Dispute resolution tracking"""
    dispute_id: str
    escrow_id: str
    vendor_id: str
    buyer_id: str
    marketplace: str
    dispute_type: str  # 'non_delivery', 'quality_issue', 'wrong_item', 'other'
    dispute_reason: str
    evidence_provided: List[str]
    vendor_response: Optional[str]
    arbiter_decision: Optional[str]
    resolution_time_hours: float
    satisfaction_rating: Optional[float]
    created_at: datetime
    resolved_at: Optional[datetime]

@dataclass
class FeedbackAnalysis:
    """Feedback sentiment and analysis"""
    feedback_id: str
    vendor_id: str
    buyer_id: str
    marketplace: str
    rating: float
    feedback_text: str
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: str  # 'negative', 'neutral', 'positive'
    authenticity_score: float  # 0.0 to 1.0 (fake review detection)
    themes_extracted: List[str]
    language_detected: str
    created_at: datetime
    verified_purchase: bool

@dataclass
class FraudPattern:
    """Detected fraud pattern"""
    pattern_id: str
    pattern_type: str  # 'fake_reviews', 'exit_scam', 'selective_scamming', 'identity_theft'
    confidence_score: float
    affected_vendors: Set[str]
    evidence: List[Dict[str, Any]]
    detection_method: str
    first_detected: datetime
    last_seen: datetime
    status: str  # 'active', 'resolved', 'investigating'


class ReputationSourceManager:
    """Manages multiple reputation data sources"""

    def __init__(self):
        self.sources = {
            'marketplace_feedback': {
                'weight': 0.4,
                'reliability': 0.8,
                'data_freshness_hours': 24
            },
            'escrow_performance': {
                'weight': 0.3,
                'reliability': 0.9,
                'data_freshness_hours': 1
            },
            'dispute_resolution': {
                'weight': 0.2,
                'reliability': 0.95,
                'data_freshness_hours': 1
            },
            'external_forums': {
                'weight': 0.1,
                'reliability': 0.6,
                'data_freshness_hours': 48
            }
        }

    async def aggregate_reputation(self, vendor_id: str,
                                 source_data: Dict[str, Any]) -> float:
        """Aggregate reputation from multiple sources with weights"""

        weighted_scores = []
        total_weight = 0

        for source, config in self.sources.items():
            if source in source_data:
                score = source_data[source].get('score', 0.0)
                freshness = source_data[source].get('freshness_hours', 0)

                # Apply freshness penalty
                freshness_factor = max(0.1, 1.0 - (freshness / config['data_freshness_hours']))

                # Apply reliability factor
                adjusted_score = score * config['reliability'] * freshness_factor

                weighted_scores.append(adjusted_score * config['weight'])
                total_weight += config['weight']

        if total_weight == 0:
            return 0.0

        return sum(weighted_scores) / total_weight

    def calculate_trust_level(self, reputation_score: float,
                            transaction_count: int,
                            dispute_rate: float) -> str:
        """Calculate trust level based on reputation metrics"""

        if reputation_score >= 0.9 and transaction_count >= 100 and dispute_rate <= 0.02:
            return 'verified'
        elif reputation_score >= 0.8 and transaction_count >= 50 and dispute_rate <= 0.05:
            return 'high'
        elif reputation_score >= 0.6 and transaction_count >= 20 and dispute_rate <= 0.1:
            return 'medium'
        elif reputation_score >= 0.4 and dispute_rate <= 0.2:
            return 'low'
        else:
            return 'untrusted'


class EscrowMonitor:
    """Monitors escrow transactions and performance"""

    def __init__(self):
        self.transaction_states = {
            'pending': 'Waiting for funding',
            'funded': 'Funds held in escrow',
            'disputed': 'Transaction disputed',
            'released': 'Funds released to vendor',
            'refunded': 'Funds returned to buyer'
        }

    async def monitor_escrow_transaction(self, escrow: EscrowTransaction) -> Dict[str, Any]:
        """Monitor individual escrow transaction"""

        monitoring_result = {
            'escrow_id': escrow.escrow_id,
            'current_status': escrow.status,
            'risk_level': 'low',
            'alerts': [],
            'recommendations': []
        }

        # Check for timeout risks
        if escrow.status == 'funded':
            hours_since_funding = (datetime.now() - escrow.funded_at).total_seconds() / 3600

            if hours_since_funding > escrow.auto_release_hours * 0.8:
                monitoring_result['alerts'].append({
                    'type': 'approaching_auto_release',
                    'severity': 'medium',
                    'message': f"Auto-release in {escrow.auto_release_hours - hours_since_funding:.1f} hours"
                })

        # Check for dispute patterns
        if escrow.status == 'disputed':
            hours_since_dispute = (datetime.now() - escrow.disputed_at).total_seconds() / 3600

            if hours_since_dispute > 72:  # 3 days
                monitoring_result['alerts'].append({
                    'type': 'long_dispute',
                    'severity': 'high',
                    'message': f"Dispute unresolved for {hours_since_dispute:.1f} hours"
                })
                monitoring_result['risk_level'] = 'high'

        # Analyze transaction patterns
        risk_factors = await self._analyze_transaction_risk(escrow)
        if risk_factors:
            monitoring_result['risk_level'] = 'medium'
            monitoring_result['alerts'].extend(risk_factors)

        return monitoring_result

    async def _analyze_transaction_risk(self, escrow: EscrowTransaction) -> List[Dict[str, Any]]:
        """Analyze transaction for risk factors"""

        risk_factors = []

        # High value transaction
        if escrow.amount > Decimal('1000'):
            risk_factors.append({
                'type': 'high_value',
                'severity': 'medium',
                'message': f"High value transaction: {escrow.amount} {escrow.currency}"
            })

        # New vendor risk
        vendor_age = escrow.metadata.get('vendor_age_days', 0)
        if vendor_age < 30:
            risk_factors.append({
                'type': 'new_vendor',
                'severity': 'medium',
                'message': f"Vendor account age: {vendor_age} days"
            })

        # Unusual product category
        high_risk_categories = ['electronics', 'pharmaceuticals', 'documents']
        if escrow.product_category in high_risk_categories:
            risk_factors.append({
                'type': 'high_risk_category',
                'severity': 'low',
                'message': f"High-risk category: {escrow.product_category}"
            })

        return risk_factors

    async def analyze_escrow_performance(self, vendor_id: str,
                                       escrow_history: List[EscrowTransaction]) -> Dict[str, Any]:
        """Analyze vendor's escrow performance"""

        if not escrow_history:
            return {
                'success_rate': 0.0,
                'average_resolution_time': 0.0,
                'dispute_rate': 0.0,
                'total_volume': Decimal('0')
            }

        total_transactions = len(escrow_history)
        successful = len([e for e in escrow_history if e.status == 'released'])
        disputed = len([e for e in escrow_history if e.status == 'disputed'])

        # Calculate resolution times
        resolved_transactions = [e for e in escrow_history if e.resolved_at]
        avg_resolution_time = 0.0

        if resolved_transactions:
            resolution_times = [
                (e.resolved_at - e.created_at).total_seconds() / 3600
                for e in resolved_transactions
            ]
            avg_resolution_time = np.mean(resolution_times)

        total_volume = sum(e.amount for e in escrow_history)

        return {
            'success_rate': successful / total_transactions if total_transactions > 0 else 0.0,
            'average_resolution_time': avg_resolution_time,
            'dispute_rate': disputed / total_transactions if total_transactions > 0 else 0.0,
            'total_volume': total_volume,
            'total_transactions': total_transactions
        }


class SentimentAnalyzer:
    """Analyzes feedback sentiment and authenticity"""

    def __init__(self):
        self.fake_review_patterns = [
            r'\b(best|amazing|perfect|excellent)\b.*\b(best|amazing|perfect|excellent)\b',  # Repetitive superlatives
            r'^.{0,20}$',  # Very short reviews
            r'\b(5 stars?|recommend|buy)\b.*\b(5 stars?|recommend|buy)\b',  # Repetitive promotional language
        ]

        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.authenticity_model = None

    async def analyze_feedback(self, feedback_text: str,
                             rating: float,
                             vendor_context: Dict[str, Any]) -> FeedbackAnalysis:
        """Comprehensive feedback analysis"""

        # Sentiment analysis
        blob = TextBlob(feedback_text)
        sentiment_score = blob.sentiment.polarity

        if sentiment_score > 0.1:
            sentiment_label = 'positive'
        elif sentiment_score < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'

        # Authenticity scoring
        authenticity_score = await self._calculate_authenticity(
            feedback_text, rating, vendor_context
        )

        # Theme extraction
        themes = await self._extract_themes(feedback_text)

        # Language detection
        try:
            language = blob.detect_language()
        except:
            language = 'unknown'

        return FeedbackAnalysis(
            feedback_id=str(uuid.uuid4()),
            vendor_id=vendor_context.get('vendor_id', 'unknown'),
            buyer_id=vendor_context.get('buyer_id', 'unknown'),
            marketplace=vendor_context.get('marketplace', 'unknown'),
            rating=rating,
            feedback_text=feedback_text,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            authenticity_score=authenticity_score,
            themes_extracted=themes,
            language_detected=language,
            created_at=datetime.now(),
            verified_purchase=vendor_context.get('verified_purchase', False)
        )

    async def _calculate_authenticity(self, text: str, rating: float,
                                    context: Dict[str, Any]) -> float:
        """Calculate authenticity score for feedback"""

        authenticity_score = 1.0

        # Check for fake review patterns
        for pattern in self.fake_review_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                authenticity_score -= 0.2

        # Length analysis
        if len(text) < 10:
            authenticity_score -= 0.3
        elif len(text) > 500:
            authenticity_score += 0.1

        # Rating-sentiment mismatch
        blob = TextBlob(text)
        expected_sentiment = (rating - 3) / 2  # Convert 1-5 rating to -1 to 1 sentiment
        sentiment_diff = abs(blob.sentiment.polarity - expected_sentiment)

        if sentiment_diff > 0.5:
            authenticity_score -= 0.2

        # Generic language detection
        generic_phrases = ['good product', 'fast shipping', 'as described', 'recommend']
        generic_count = sum(1 for phrase in generic_phrases if phrase in text.lower())

        if generic_count >= 2:
            authenticity_score -= 0.1

        # Verified purchase bonus
        if context.get('verified_purchase', False):
            authenticity_score += 0.2

        return max(0.0, min(1.0, authenticity_score))

    async def _extract_themes(self, text: str) -> List[str]:
        """Extract themes from feedback text"""

        themes = []

        # Define theme keywords
        theme_keywords = {
            'shipping': ['shipping', 'delivery', 'arrived', 'fast', 'slow', 'package'],
            'quality': ['quality', 'good', 'bad', 'excellent', 'poor', 'defective'],
            'communication': ['response', 'contact', 'message', 'helpful', 'rude'],
            'packaging': ['packaging', 'wrapped', 'box', 'damaged', 'secure'],
            'price': ['price', 'cheap', 'expensive', 'value', 'worth', 'cost']
        }

        text_lower = text.lower()

        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme)

        return themes

    async def detect_review_manipulation(self, vendor_feedbacks: List[FeedbackAnalysis]) -> Dict[str, Any]:
        """Detect potential review manipulation"""

        if len(vendor_feedbacks) < 10:
            return {'manipulation_detected': False, 'confidence': 0.0}

        manipulation_indicators = []

        # Rating distribution analysis
        ratings = [f.rating for f in vendor_feedbacks]
        rating_std = np.std(ratings)

        if rating_std < 0.5:  # Very low variance in ratings
            manipulation_indicators.append('uniform_ratings')

        # Sentiment-rating correlation
        sentiments = [f.sentiment_score for f in vendor_feedbacks]
        correlation = np.corrcoef(ratings, sentiments)[0, 1]

        if correlation < 0.3:  # Poor correlation between rating and sentiment
            manipulation_indicators.append('sentiment_rating_mismatch')

        # Time clustering analysis
        timestamps = [f.created_at for f in vendor_feedbacks[-20:]]  # Last 20 reviews
        if len(timestamps) >= 5:
            time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() / 3600
                         for i in range(1, len(timestamps))]

            if np.std(time_diffs) < 2:  # Reviews posted at very regular intervals
                manipulation_indicators.append('time_clustering')

        # Low authenticity scores
        avg_authenticity = np.mean([f.authenticity_score for f in vendor_feedbacks])
        if avg_authenticity < 0.6:
            manipulation_indicators.append('low_authenticity')

        # Calculate confidence
        confidence = len(manipulation_indicators) / 4.0  # 4 possible indicators

        return {
            'manipulation_detected': len(manipulation_indicators) >= 2,
            'confidence': confidence,
            'indicators': manipulation_indicators,
            'average_authenticity': avg_authenticity
        }


class FraudDetector:
    """Detects fraud patterns using machine learning"""

    def __init__(self):
        self.models = {
            'exit_scam': None,
            'selective_scam': None,
            'fake_vendor': None
        }
        self.feature_scalers = {}

    async def detect_fraud_patterns(self, vendor_data: Dict[str, Any],
                                  transaction_history: List[Dict[str, Any]]) -> List[FraudPattern]:
        """Detect various fraud patterns"""

        detected_patterns = []

        # Exit scam detection
        exit_scam = await self._detect_exit_scam(vendor_data, transaction_history)
        if exit_scam:
            detected_patterns.append(exit_scam)

        # Selective scamming detection
        selective_scam = await self._detect_selective_scamming(transaction_history)
        if selective_scam:
            detected_patterns.append(selective_scam)

        # Fake vendor detection
        fake_vendor = await self._detect_fake_vendor(vendor_data)
        if fake_vendor:
            detected_patterns.append(fake_vendor)

        return detected_patterns

    async def _detect_exit_scam(self, vendor_data: Dict[str, Any],
                              transactions: List[Dict[str, Any]]) -> Optional[FraudPattern]:
        """Detect exit scam patterns"""

        if len(transactions) < 10:
            return None

        # Analyze recent transaction patterns
        recent_transactions = sorted(transactions, key=lambda x: x['timestamp'])[-30:]

        # Check for sudden increase in transaction volume
        recent_volume = sum(t.get('amount', 0) for t in recent_transactions[-10:])
        historical_volume = sum(t.get('amount', 0) for t in recent_transactions[:-10]) / len(recent_transactions[:-10])

        volume_increase_ratio = recent_volume / (historical_volume * 10) if historical_volume > 0 else 0

        # Check for sudden stop in activity
        last_transaction_days = (datetime.now() -
                               datetime.fromisoformat(recent_transactions[-1]['timestamp'])).days

        # Check for reputation changes
        reputation_decline = vendor_data.get('reputation_decline_rate', 0)

        indicators = []
        confidence = 0.0

        if volume_increase_ratio > 3.0:
            indicators.append('sudden_volume_increase')
            confidence += 0.3

        if last_transaction_days > 14:
            indicators.append('sudden_activity_stop')
            confidence += 0.4

        if reputation_decline > 0.2:
            indicators.append('reputation_decline')
            confidence += 0.3

        if confidence > 0.6:
            return FraudPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type='exit_scam',
                confidence_score=confidence,
                affected_vendors={vendor_data['vendor_id']},
                evidence=indicators,
                detection_method='statistical_analysis',
                first_detected=datetime.now(),
                last_seen=datetime.now(),
                status='active'
            )

        return None

    async def _detect_selective_scamming(self, transactions: List[Dict[str, Any]]) -> Optional[FraudPattern]:
        """Detect selective scamming patterns"""

        if len(transactions) < 20:
            return None

        # Analyze transaction outcomes by value
        high_value_threshold = np.percentile([t.get('amount', 0) for t in transactions], 75)

        high_value_txs = [t for t in transactions if t.get('amount', 0) > high_value_threshold]
        low_value_txs = [t for t in transactions if t.get('amount', 0) <= high_value_threshold]

        # Calculate success rates
        high_value_success_rate = len([t for t in high_value_txs if t.get('status') == 'successful']) / len(high_value_txs)
        low_value_success_rate = len([t for t in low_value_txs if t.get('status') == 'successful']) / len(low_value_txs)

        # Check for significant difference
        success_rate_diff = low_value_success_rate - high_value_success_rate

        if success_rate_diff > 0.3 and len(high_value_txs) >= 5:
            return FraudPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type='selective_scamming',
                confidence_score=min(success_rate_diff, 1.0),
                affected_vendors=set(),  # Would be populated with actual vendor IDs
                evidence=[
                    f'High value success rate: {high_value_success_rate:.2f}',
                    f'Low value success rate: {low_value_success_rate:.2f}',
                    f'Difference: {success_rate_diff:.2f}'
                ],
                detection_method='statistical_analysis',
                first_detected=datetime.now(),
                last_seen=datetime.now(),
                status='active'
            )

        return None

    async def _detect_fake_vendor(self, vendor_data: Dict[str, Any]) -> Optional[FraudPattern]:
        """Detect fake vendor indicators"""

        indicators = []
        confidence = 0.0

        # Check account age vs activity
        account_age_days = vendor_data.get('account_age_days', 0)
        transaction_count = vendor_data.get('transaction_count', 0)

        if account_age_days < 30 and transaction_count > 100:
            indicators.append('suspicious_activity_velocity')
            confidence += 0.4

        # Check profile completeness
        profile_fields = ['description', 'pgp_key', 'shipping_info', 'contact_info']
        missing_fields = sum(1 for field in profile_fields if not vendor_data.get(field))

        if missing_fields >= 3:
            indicators.append('incomplete_profile')
            confidence += 0.2

        # Check review patterns
        if vendor_data.get('review_manipulation_detected', False):
            indicators.append('fake_reviews')
            confidence += 0.4

        if confidence > 0.5:
            return FraudPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type='fake_vendor',
                confidence_score=confidence,
                affected_vendors={vendor_data['vendor_id']},
                evidence=indicators,
                detection_method='heuristic_analysis',
                first_detected=datetime.now(),
                last_seen=datetime.now(),
                status='active'
            )

        return None


class VendorReputationFramework:
    """Main vendor reputation analysis framework"""

    def __init__(self, db_config: Dict[str, Any], redis_config: Dict[str, Any],
                 kafka_config: Dict[str, Any]):

        self.db_config = db_config
        self.redis_config = redis_config
        self.kafka_config = kafka_config

        # Core components
        self.reputation_manager = ReputationSourceManager()
        self.escrow_monitor = EscrowMonitor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.fraud_detector = FraudDetector()

        # Data storage
        self.db_pool = None
        self.redis_client = None
        self.kafka_producer = None

        # Monitoring state
        self.monitored_vendors: Set[str] = set()
        self.reputation_cache_ttl = 3600  # 1 hour

    async def initialize(self):
        """Initialize all components"""

        logger.info("Initializing Vendor Reputation Framework...")

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
            f"redis://{self.redis_config['host']}:{self.redis_config['port']}/2"
        )

        # Kafka producer
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_config['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )
        await self.kafka_producer.start()

        # Initialize database tables
        await self._initialize_database_tables()

        logger.info("Vendor Reputation Framework initialized successfully")

    async def _initialize_database_tables(self):
        """Create database tables for reputation analysis"""

        tables = [
            '''
            CREATE TABLE IF NOT EXISTS vendor_reputations (
                vendor_id VARCHAR(255) PRIMARY KEY,
                vendor_name VARCHAR(500),
                marketplace VARCHAR(100),
                overall_score FLOAT,
                trust_level VARCHAR(20),
                total_transactions INTEGER,
                successful_transactions INTEGER,
                disputed_transactions INTEGER,
                total_volume DECIMAL(15,2),
                average_rating FLOAT,
                rating_count INTEGER,
                response_time_hours FLOAT,
                uptime_percentage FLOAT,
                escrow_success_rate FLOAT,
                reputation_sources TEXT[],
                risk_indicators TEXT[],
                last_updated TIMESTAMP,
                reputation_history JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            ''',
            '''
            CREATE TABLE IF NOT EXISTS escrow_transactions (
                escrow_id VARCHAR(255) PRIMARY KEY,
                vendor_id VARCHAR(255),
                buyer_id VARCHAR(255),
                marketplace VARCHAR(100),
                amount DECIMAL(15,2),
                currency VARCHAR(10),
                product_category VARCHAR(100),
                status VARCHAR(50),
                created_at TIMESTAMP,
                funded_at TIMESTAMP,
                disputed_at TIMESTAMP,
                resolved_at TIMESTAMP,
                dispute_reason TEXT,
                resolution_type VARCHAR(50),
                arbiter_notes TEXT,
                auto_release_hours INTEGER,
                fees JSONB,
                metadata JSONB
            );
            ''',
            '''
            CREATE TABLE IF NOT EXISTS dispute_records (
                dispute_id VARCHAR(255) PRIMARY KEY,
                escrow_id VARCHAR(255),
                vendor_id VARCHAR(255),
                buyer_id VARCHAR(255),
                marketplace VARCHAR(100),
                dispute_type VARCHAR(50),
                dispute_reason TEXT,
                evidence_provided TEXT[],
                vendor_response TEXT,
                arbiter_decision TEXT,
                resolution_time_hours FLOAT,
                satisfaction_rating FLOAT,
                created_at TIMESTAMP,
                resolved_at TIMESTAMP
            );
            ''',
            '''
            CREATE TABLE IF NOT EXISTS feedback_analysis (
                feedback_id VARCHAR(255) PRIMARY KEY,
                vendor_id VARCHAR(255),
                buyer_id VARCHAR(255),
                marketplace VARCHAR(100),
                rating FLOAT,
                feedback_text TEXT,
                sentiment_score FLOAT,
                sentiment_label VARCHAR(20),
                authenticity_score FLOAT,
                themes_extracted TEXT[],
                language_detected VARCHAR(10),
                created_at TIMESTAMP,
                verified_purchase BOOLEAN
            );
            ''',
            '''
            CREATE TABLE IF NOT EXISTS fraud_patterns (
                pattern_id VARCHAR(255) PRIMARY KEY,
                pattern_type VARCHAR(50),
                confidence_score FLOAT,
                affected_vendors TEXT[],
                evidence JSONB,
                detection_method VARCHAR(50),
                first_detected TIMESTAMP,
                last_seen TIMESTAMP,
                status VARCHAR(20)
            );
            ''',
            '''
            CREATE INDEX IF NOT EXISTS idx_vendor_reputations_marketplace ON vendor_reputations(marketplace);
            CREATE INDEX IF NOT EXISTS idx_vendor_reputations_trust_level ON vendor_reputations(trust_level);
            CREATE INDEX IF NOT EXISTS idx_escrow_transactions_vendor ON escrow_transactions(vendor_id);
            CREATE INDEX IF NOT EXISTS idx_escrow_transactions_status ON escrow_transactions(status);
            CREATE INDEX IF NOT EXISTS idx_dispute_records_vendor ON dispute_records(vendor_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_analysis_vendor ON feedback_analysis(vendor_id);
            CREATE INDEX IF NOT EXISTS idx_fraud_patterns_type ON fraud_patterns(pattern_type);
            '''
        ]

        async with self.db_pool.acquire() as conn:
            for table_sql in tables:
                await conn.execute(table_sql)

    async def analyze_vendor_reputation(self, vendor_id: str) -> VendorReputation:
        """Comprehensive vendor reputation analysis"""

        try:
            # Check cache first
            cached_reputation = await self._get_cached_reputation(vendor_id)
            if cached_reputation:
                return cached_reputation

            # Gather data from multiple sources
            source_data = await self._gather_reputation_sources(vendor_id)

            # Aggregate reputation score
            overall_score = await self.reputation_manager.aggregate_reputation(vendor_id, source_data)

            # Calculate trust level
            transaction_count = source_data.get('escrow_performance', {}).get('total_transactions', 0)
            dispute_rate = source_data.get('escrow_performance', {}).get('dispute_rate', 0)
            trust_level = self.reputation_manager.calculate_trust_level(
                overall_score, transaction_count, dispute_rate
            )

            # Analyze feedback sentiment
            feedback_data = source_data.get('marketplace_feedback', {})

            # Detect fraud patterns
            fraud_patterns = await self._detect_vendor_fraud(vendor_id, source_data)

            # Create reputation object
            reputation = VendorReputation(
                vendor_id=vendor_id,
                vendor_name=source_data.get('vendor_name', 'Unknown'),
                marketplace=source_data.get('marketplace', 'Unknown'),
                overall_score=overall_score,
                trust_level=trust_level,
                total_transactions=transaction_count,
                successful_transactions=source_data.get('escrow_performance', {}).get('successful', 0),
                disputed_transactions=source_data.get('escrow_performance', {}).get('disputed', 0),
                total_volume=source_data.get('escrow_performance', {}).get('total_volume', Decimal('0')),
                average_rating=feedback_data.get('average_rating', 0.0),
                rating_count=feedback_data.get('rating_count', 0),
                response_time_hours=source_data.get('response_time_hours', 0.0),
                uptime_percentage=source_data.get('uptime_percentage', 0.0),
                escrow_success_rate=source_data.get('escrow_performance', {}).get('success_rate', 0.0),
                reputation_sources=list(source_data.keys()),
                risk_indicators=[p.pattern_type for p in fraud_patterns],
                last_updated=datetime.now(),
                reputation_history=[]
            )

            # Store reputation
            await self._store_reputation(reputation)

            # Cache reputation
            await self._cache_reputation(reputation)

            # Store fraud patterns
            for pattern in fraud_patterns:
                await self._store_fraud_pattern(pattern)

            # Publish to Kafka
            await self.kafka_producer.send(
                'vendor_reputation_analysis',
                key=vendor_id,
                value=asdict(reputation)
            )

            logger.info(f"Analyzed reputation for vendor {vendor_id}: {trust_level} ({overall_score:.2f})")
            return reputation

        except Exception as e:
            logger.error(f"Error analyzing vendor reputation {vendor_id}: {e}")
            raise

    async def _gather_reputation_sources(self, vendor_id: str) -> Dict[str, Any]:
        """Gather reputation data from all sources"""

        source_data = {}

        # Marketplace feedback
        feedback_data = await self._get_marketplace_feedback(vendor_id)
        if feedback_data:
            source_data['marketplace_feedback'] = feedback_data

        # Escrow performance
        escrow_data = await self._get_escrow_performance(vendor_id)
        if escrow_data:
            source_data['escrow_performance'] = escrow_data

        # Dispute resolution
        dispute_data = await self._get_dispute_history(vendor_id)
        if dispute_data:
            source_data['dispute_resolution'] = dispute_data

        return source_data

    async def _get_marketplace_feedback(self, vendor_id: str) -> Dict[str, Any]:
        """Get marketplace feedback data"""

        async with self.db_pool.acquire() as conn:
            feedbacks = await conn.fetch(
                '''
                SELECT rating, feedback_text, sentiment_score, authenticity_score, created_at
                FROM feedback_analysis
                WHERE vendor_id = $1
                ORDER BY created_at DESC
                LIMIT 100
                ''',
                vendor_id
            )

            if not feedbacks:
                return None

            ratings = [float(f['rating']) for f in feedbacks]
            authenticity_scores = [float(f['authenticity_score']) for f in feedbacks]

            return {
                'score': np.mean(ratings) / 5.0,  # Normalize to 0-1
                'average_rating': np.mean(ratings),
                'rating_count': len(ratings),
                'average_authenticity': np.mean(authenticity_scores),
                'freshness_hours': (datetime.now() - feedbacks[0]['created_at']).total_seconds() / 3600
            }

    async def _get_escrow_performance(self, vendor_id: str) -> Dict[str, Any]:
        """Get escrow performance data"""

        async with self.db_pool.acquire() as conn:
            escrows = await conn.fetch(
                '''
                SELECT status, amount, created_at, resolved_at
                FROM escrow_transactions
                WHERE vendor_id = $1
                ORDER BY created_at DESC
                LIMIT 200
                ''',
                vendor_id
            )

            if not escrows:
                return None

            total_transactions = len(escrows)
            successful = len([e for e in escrows if e['status'] == 'released'])
            disputed = len([e for e in escrows if e['status'] == 'disputed'])
            total_volume = sum(float(e['amount']) for e in escrows)

            return {
                'score': successful / total_transactions if total_transactions > 0 else 0.0,
                'total_transactions': total_transactions,
                'successful': successful,
                'disputed': disputed,
                'success_rate': successful / total_transactions if total_transactions > 0 else 0.0,
                'dispute_rate': disputed / total_transactions if total_transactions > 0 else 0.0,
                'total_volume': Decimal(str(total_volume)),
                'freshness_hours': (datetime.now() - escrows[0]['created_at']).total_seconds() / 3600
            }

    async def _get_dispute_history(self, vendor_id: str) -> Dict[str, Any]:
        """Get dispute resolution history"""

        async with self.db_pool.acquire() as conn:
            disputes = await conn.fetch(
                '''
                SELECT resolution_type, resolution_time_hours, satisfaction_rating, created_at
                FROM dispute_records
                WHERE vendor_id = $1 AND resolved_at IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 50
                ''',
                vendor_id
            )

            if not disputes:
                return None

            vendor_favorable = len([d for d in disputes if d['resolution_type'] == 'vendor_favor'])
            total_disputes = len(disputes)
            avg_resolution_time = np.mean([float(d['resolution_time_hours']) for d in disputes])

            satisfaction_scores = [float(d['satisfaction_rating']) for d in disputes
                                 if d['satisfaction_rating'] is not None]
            avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else 0.0

            return {
                'score': vendor_favorable / total_disputes if total_disputes > 0 else 0.0,
                'vendor_favorable_rate': vendor_favorable / total_disputes if total_disputes > 0 else 0.0,
                'average_resolution_time': avg_resolution_time,
                'average_satisfaction': avg_satisfaction,
                'freshness_hours': (datetime.now() - disputes[0]['created_at']).total_seconds() / 3600
            }

    async def _detect_vendor_fraud(self, vendor_id: str,
                                 source_data: Dict[str, Any]) -> List[FraudPattern]:
        """Detect fraud patterns for vendor"""

        # Prepare vendor data for fraud detection
        vendor_data = {
            'vendor_id': vendor_id,
            'account_age_days': 365,  # Would calculate from actual data
            'transaction_count': source_data.get('escrow_performance', {}).get('total_transactions', 0),
            'reputation_decline_rate': 0.0,  # Would calculate from historical data
            'review_manipulation_detected': False  # Would detect from feedback analysis
        }

        # Prepare transaction history
        transaction_history = []  # Would populate from database

        return await self.fraud_detector.detect_fraud_patterns(vendor_data, transaction_history)

    async def _get_cached_reputation(self, vendor_id: str) -> Optional[VendorReputation]:
        """Get cached reputation if available and fresh"""

        cached_data = await self.redis_client.get(f"reputation:{vendor_id}")
        if cached_data:
            try:
                data = json.loads(cached_data)
                last_updated = datetime.fromisoformat(data['last_updated'])

                if (datetime.now() - last_updated).total_seconds() < self.reputation_cache_ttl:
                    return VendorReputation(**data)
            except Exception as e:
                logger.warning(f"Error deserializing cached reputation: {e}")

        return None

    async def _cache_reputation(self, reputation: VendorReputation):
        """Cache reputation data"""

        reputation_dict = asdict(reputation)
        await self.redis_client.setex(
            f"reputation:{reputation.vendor_id}",
            self.reputation_cache_ttl,
            json.dumps(reputation_dict, default=str)
        )

    async def _store_reputation(self, reputation: VendorReputation):
        """Store reputation in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO vendor_reputations
                (vendor_id, vendor_name, marketplace, overall_score, trust_level,
                 total_transactions, successful_transactions, disputed_transactions,
                 total_volume, average_rating, rating_count, response_time_hours,
                 uptime_percentage, escrow_success_rate, reputation_sources,
                 risk_indicators, last_updated, reputation_history)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                ON CONFLICT (vendor_id) DO UPDATE SET
                    overall_score = EXCLUDED.overall_score,
                    trust_level = EXCLUDED.trust_level,
                    total_transactions = EXCLUDED.total_transactions,
                    successful_transactions = EXCLUDED.successful_transactions,
                    disputed_transactions = EXCLUDED.disputed_transactions,
                    total_volume = EXCLUDED.total_volume,
                    average_rating = EXCLUDED.average_rating,
                    rating_count = EXCLUDED.rating_count,
                    response_time_hours = EXCLUDED.response_time_hours,
                    uptime_percentage = EXCLUDED.uptime_percentage,
                    escrow_success_rate = EXCLUDED.escrow_success_rate,
                    reputation_sources = EXCLUDED.reputation_sources,
                    risk_indicators = EXCLUDED.risk_indicators,
                    last_updated = EXCLUDED.last_updated,
                    updated_at = NOW()
                ''',
                reputation.vendor_id, reputation.vendor_name, reputation.marketplace,
                reputation.overall_score, reputation.trust_level,
                reputation.total_transactions, reputation.successful_transactions,
                reputation.disputed_transactions, reputation.total_volume,
                reputation.average_rating, reputation.rating_count,
                reputation.response_time_hours, reputation.uptime_percentage,
                reputation.escrow_success_rate, reputation.reputation_sources,
                reputation.risk_indicators, reputation.last_updated,
                json.dumps(reputation.reputation_history)
            )

    async def _store_fraud_pattern(self, pattern: FraudPattern):
        """Store fraud pattern in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO fraud_patterns
                (pattern_id, pattern_type, confidence_score, affected_vendors,
                 evidence, detection_method, first_detected, last_seen, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (pattern_id) DO UPDATE SET
                    confidence_score = EXCLUDED.confidence_score,
                    last_seen = EXCLUDED.last_seen,
                    status = EXCLUDED.status
                ''',
                pattern.pattern_id, pattern.pattern_type, pattern.confidence_score,
                list(pattern.affected_vendors), json.dumps(pattern.evidence),
                pattern.detection_method, pattern.first_detected,
                pattern.last_seen, pattern.status
            )

    async def monitor_vendor(self, vendor_id: str) -> Dict[str, Any]:
        """Add vendor to monitoring list"""

        self.monitored_vendors.add(vendor_id)
        await self.redis_client.sadd('monitored_vendors', vendor_id)

        return {
            'status': 'success',
            'vendor_id': vendor_id,
            'monitoring': True
        }

    async def get_reputation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reputation statistics"""

        async with self.db_pool.acquire() as conn:
            # Reputation statistics
            reputation_stats = await conn.fetchrow(
                '''
                SELECT
                    COUNT(*) as total_vendors,
                    AVG(overall_score) as average_reputation,
                    COUNT(*) FILTER (WHERE trust_level = 'verified') as verified_vendors,
                    COUNT(*) FILTER (WHERE trust_level = 'high') as high_trust_vendors,
                    COUNT(*) FILTER (WHERE trust_level = 'untrusted') as untrusted_vendors
                FROM vendor_reputations
                '''
            )

            # Fraud statistics
            fraud_stats = await conn.fetchrow(
                '''
                SELECT
                    COUNT(*) as total_patterns,
                    COUNT(*) FILTER (WHERE status = 'active') as active_patterns,
                    AVG(confidence_score) as average_confidence
                FROM fraud_patterns
                '''
            )

            # Escrow statistics
            escrow_stats = await conn.fetchrow(
                '''
                SELECT
                    COUNT(*) as total_escrows,
                    COUNT(*) FILTER (WHERE status = 'disputed') as disputed_escrows,
                    AVG(amount) as average_amount
                FROM escrow_transactions
                '''
            )

        return {
            'reputation': dict(reputation_stats) if reputation_stats else {},
            'fraud': dict(fraud_stats) if fraud_stats else {},
            'escrow': dict(escrow_stats) if escrow_stats else {},
            'monitored_vendors': len(self.monitored_vendors)
        }

    async def cleanup(self):
        """Cleanup resources"""

        logger.info("Cleaning up Vendor Reputation Framework...")

        if self.kafka_producer:
            await self.kafka_producer.stop()

        if self.redis_client:
            await self.redis_client.close()

        if self.db_pool:
            await self.db_pool.close()


# Example usage
async def main():
    """Example usage of Vendor Reputation Framework"""

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

    # Initialize framework
    framework = VendorReputationFramework(db_config, redis_config, kafka_config)
    await framework.initialize()

    try:
        # Monitor a vendor
        await framework.monitor_vendor('vendor_123')

        # Analyze vendor reputation
        reputation = await framework.analyze_vendor_reputation('vendor_123')
        print(f"Vendor reputation: {reputation.trust_level} ({reputation.overall_score:.2f})")

        # Get statistics
        stats = await framework.get_reputation_statistics()
        print(f"Framework statistics: {stats}")

    except KeyboardInterrupt:
        logger.info("Shutting down framework...")
    finally:
        await framework.cleanup()


if __name__ == "__main__":
    asyncio.run(main())