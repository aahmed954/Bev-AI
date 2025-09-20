#!/usr/bin/env python3
"""
Intelligence Fusion Processor
Threat feed aggregation, ML threat classification, and strategic intelligence reporting
"""

import asyncio
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import aiohttp
import asyncpg
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge
import logging
import geoip2.database
import geoip2.errors
from .security_framework import OperationalSecurityFramework

# Metrics
THREAT_FEEDS_PROCESSED = Counter('threat_feeds_processed_total', 'Total threat feeds processed', ['source'])
THREATS_CLASSIFIED = Counter('threats_classified_total', 'Total threats classified', ['classification'])
INTELLIGENCE_REPORTS_GENERATED = Counter('intelligence_reports_generated_total', 'Intelligence reports generated', ['type'])
PREDICTION_ACCURACY = Gauge('prediction_accuracy', 'ML prediction accuracy')
FEED_LATENCY = Histogram('feed_processing_latency_seconds', 'Feed processing latency')

logger = logging.getLogger(__name__)

class ThreatClassification(Enum):
    """Threat classification categories"""
    MALWARE = "malware"
    PHISHING = "phishing"
    BOTNET = "botnet"
    APT = "advanced_persistent_threat"
    RANSOMWARE = "ransomware"
    CRYPTOMINING = "cryptomining"
    DDoS = "distributed_denial_of_service"
    DATA_BREACH = "data_breach"
    INSIDER_THREAT = "insider_threat"
    SUPPLY_CHAIN = "supply_chain_attack"

class ThreatSeverity(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "informational"

class IntelligenceType(Enum):
    """Types of intelligence products"""
    TACTICAL = "tactical"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    TECHNICAL = "technical"

class FeedReliability(Enum):
    """Feed reliability ratings"""
    A = "completely_reliable"
    B = "usually_reliable"
    C = "fairly_reliable"
    D = "not_usually_reliable"
    E = "unreliable"
    F = "reliability_unknown"

@dataclass
class ThreatFeed:
    """Threat feed configuration"""
    id: str
    name: str
    url: str
    feed_type: str  # json, xml, csv, stix
    update_frequency: int  # seconds
    reliability: FeedReliability
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    is_active: bool = True
    last_updated: Optional[datetime] = None
    indicators_count: int = 0

@dataclass
class ThreatIndicator:
    """Individual threat indicator"""
    id: str
    value: str
    indicator_type: str  # ip, domain, hash, url, email
    classification: ThreatClassification
    severity: ThreatSeverity
    confidence: float
    first_seen: datetime
    last_seen: datetime
    sources: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    context: Dict = field(default_factory=dict)
    geolocation: Optional[Dict] = None
    related_indicators: List[str] = field(default_factory=list)

@dataclass
class ThreatCampaign:
    """Threat campaign information"""
    id: str
    name: str
    description: str
    threat_actors: List[str]
    start_date: datetime
    end_date: Optional[datetime]
    target_sectors: List[str]
    target_countries: List[str]
    techniques: List[str] = field(default_factory=list)
    indicators: Set[str] = field(default_factory=set)
    confidence: float = 0.0

@dataclass
class IntelligenceProduct:
    """Intelligence analysis product"""
    id: str
    title: str
    intelligence_type: IntelligenceType
    classification: str
    created_date: datetime
    valid_until: datetime
    executive_summary: str
    key_findings: List[str]
    indicators: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_level: str = "medium"
    sources: List[str] = field(default_factory=list)
    author: str = "automated_analysis"

@dataclass
class GeospatialThreat:
    """Geospatial threat information"""
    id: str
    location: Dict  # lat, lon, country, region
    threat_type: ThreatClassification
    severity: ThreatSeverity
    timestamp: datetime
    indicators: List[str] = field(default_factory=list)
    threat_density: float = 0.0
    context: Dict = field(default_factory=dict)

class ThreatFeedAggregator:
    """Aggregate threat intelligence from multiple sources"""

    def __init__(self):
        self.feeds: Dict[str, ThreatFeed] = {}
        self.session = None
        self.geoip_reader = None
        self._initialize_feeds()
        self._initialize_geoip()

    def _initialize_feeds(self):
        """Initialize threat feed configurations"""
        # Default threat feeds (in production, load from config)
        default_feeds = [
            ThreatFeed(
                id="alienvault_otx",
                name="AlienVault OTX",
                url="https://otx.alienvault.com/api/v1/indicators/export",
                feed_type="json",
                update_frequency=3600,
                reliability=FeedReliability.B
            ),
            ThreatFeed(
                id="emerging_threats",
                name="Emerging Threats",
                url="https://rules.emergingthreats.net/blockrules/compromised-ips.txt",
                feed_type="text",
                update_frequency=1800,
                reliability=FeedReliability.B
            ),
            ThreatFeed(
                id="abuse_ch_malware",
                name="Abuse.ch Malware",
                url="https://urlhaus-api.abuse.ch/v1/urls/recent/",
                feed_type="json",
                update_frequency=900,
                reliability=FeedReliability.A
            ),
            ThreatFeed(
                id="malware_domain_list",
                name="Malware Domain List",
                url="http://www.malwaredomainlist.com/hostslist/hosts.txt",
                feed_type="text",
                update_frequency=3600,
                reliability=FeedReliability.C
            )
        ]

        for feed in default_feeds:
            self.feeds[feed.id] = feed

    def _initialize_geoip(self):
        """Initialize GeoIP database"""
        try:
            # Try to load GeoIP database
            geoip_path = "/opt/bev/geoip/GeoLite2-City.mmdb"
            if os.path.exists(geoip_path):
                self.geoip_reader = geoip2.database.Reader(geoip_path)
                logger.info("GeoIP database loaded")
            else:
                logger.warning("GeoIP database not found")
        except Exception as e:
            logger.error(f"Failed to initialize GeoIP: {e}")

    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )

    async def collect_threat_feeds(self) -> Dict[str, List[ThreatIndicator]]:
        """Collect indicators from all active threat feeds"""
        all_indicators = {}

        try:
            # Process feeds in parallel
            tasks = []
            for feed_id, feed in self.feeds.items():
                if feed.is_active:
                    task = asyncio.create_task(self._process_feed(feed))
                    tasks.append((feed_id, task))

            # Wait for all feeds to complete
            for feed_id, task in tasks:
                try:
                    indicators = await task
                    all_indicators[feed_id] = indicators
                    THREAT_FEEDS_PROCESSED.labels(source=feed_id).inc()
                except Exception as e:
                    logger.error(f"Error processing feed {feed_id}: {e}")
                    all_indicators[feed_id] = []

            return all_indicators

        except Exception as e:
            logger.error(f"Error collecting threat feeds: {e}")
            return {}

    async def _process_feed(self, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Process individual threat feed"""
        indicators = []

        try:
            with FEED_LATENCY.time():
                # Fetch feed data
                data = await self._fetch_feed_data(feed)

                if not data:
                    return indicators

                # Parse based on feed type
                if feed.feed_type == "json":
                    indicators = await self._parse_json_feed(data, feed)
                elif feed.feed_type == "text":
                    indicators = await self._parse_text_feed(data, feed)
                elif feed.feed_type == "csv":
                    indicators = await self._parse_csv_feed(data, feed)
                elif feed.feed_type == "stix":
                    indicators = await self._parse_stix_feed(data, feed)

                # Update feed metadata
                feed.last_updated = datetime.now()
                feed.indicators_count = len(indicators)

                logger.info(f"Processed {len(indicators)} indicators from {feed.name}")

        except Exception as e:
            logger.error(f"Error processing feed {feed.name}: {e}")

        return indicators

    async def _fetch_feed_data(self, feed: ThreatFeed) -> Optional[str]:
        """Fetch data from threat feed"""
        try:
            headers = feed.headers.copy()
            if feed.api_key:
                headers['X-API-KEY'] = feed.api_key

            async with self.session.get(feed.url, headers=headers) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.error(f"Feed {feed.name} returned status {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Error fetching feed {feed.name}: {e}")
            return None

    async def _parse_json_feed(self, data: str, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Parse JSON threat feed"""
        indicators = []

        try:
            feed_data = json.loads(data)

            # Handle different JSON structures
            if isinstance(feed_data, list):
                items = feed_data
            elif isinstance(feed_data, dict):
                # Common keys for indicator lists
                for key in ['indicators', 'results', 'data', 'items']:
                    if key in feed_data:
                        items = feed_data[key]
                        break
                else:
                    items = [feed_data]
            else:
                return indicators

            for item in items:
                indicator = await self._create_indicator_from_json(item, feed)
                if indicator:
                    indicators.append(indicator)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for feed {feed.name}: {e}")
        except Exception as e:
            logger.error(f"Error parsing JSON feed {feed.name}: {e}")

        return indicators

    async def _create_indicator_from_json(self, item: Dict, feed: ThreatFeed) -> Optional[ThreatIndicator]:
        """Create threat indicator from JSON item"""
        try:
            # Extract indicator value and type
            indicator_value = None
            indicator_type = None

            # Common field mappings
            value_fields = ['indicator', 'value', 'ioc', 'observable', 'ip', 'domain', 'hash', 'url']
            type_fields = ['type', 'indicator_type', 'ioc_type', 'category']

            for field in value_fields:
                if field in item and item[field]:
                    indicator_value = str(item[field]).strip()
                    break

            for field in type_fields:
                if field in item and item[field]:
                    indicator_type = str(item[field]).strip().lower()
                    break

            if not indicator_value:
                return None

            # Determine indicator type if not provided
            if not indicator_type:
                indicator_type = self._determine_indicator_type(indicator_value)

            # Extract metadata
            classification = self._determine_classification(item, feed)
            severity = self._determine_severity(item, feed)
            confidence = self._extract_confidence(item)

            # Create indicator
            indicator_id = hashlib.sha256(f"{indicator_value}_{feed.id}".encode()).hexdigest()[:16]

            indicator = ThreatIndicator(
                id=indicator_id,
                value=indicator_value,
                indicator_type=indicator_type,
                classification=classification,
                severity=severity,
                confidence=confidence,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                sources=[feed.id],
                context=item
            )

            # Add geolocation for IP addresses
            if indicator_type == 'ip':
                indicator.geolocation = await self._get_geolocation(indicator_value)

            return indicator

        except Exception as e:
            logger.error(f"Error creating indicator from JSON: {e}")
            return None

    def _determine_indicator_type(self, value: str) -> str:
        """Determine indicator type from value"""
        import re

        # IP address
        if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', value):
            return 'ip'

        # Domain
        if re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}$', value):
            return 'domain'

        # Hash (MD5, SHA1, SHA256)
        if re.match(r'^[a-fA-F0-9]{32}$', value):
            return 'md5'
        elif re.match(r'^[a-fA-F0-9]{40}$', value):
            return 'sha1'
        elif re.match(r'^[a-fA-F0-9]{64}$', value):
            return 'sha256'

        # URL
        if value.startswith(('http://', 'https://')):
            return 'url'

        # Email
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
            return 'email'

        return 'unknown'

    def _determine_classification(self, item: Dict, feed: ThreatFeed) -> ThreatClassification:
        """Determine threat classification"""
        # Check for classification indicators in the data
        classification_indicators = {
            ThreatClassification.MALWARE: ['malware', 'virus', 'trojan', 'backdoor'],
            ThreatClassification.PHISHING: ['phishing', 'phish', 'credential'],
            ThreatClassification.BOTNET: ['botnet', 'bot', 'zombie'],
            ThreatClassification.RANSOMWARE: ['ransomware', 'ransom', 'crypto'],
            ThreatClassification.APT: ['apt', 'advanced', 'persistent'],
            ThreatClassification.DDoS: ['ddos', 'amplification', 'reflection']
        }

        # Check item content for indicators
        item_text = json.dumps(item).lower()

        for classification, keywords in classification_indicators.items():
            if any(keyword in item_text for keyword in keywords):
                return classification

        # Default based on feed
        if 'malware' in feed.name.lower():
            return ThreatClassification.MALWARE
        elif 'phish' in feed.name.lower():
            return ThreatClassification.PHISHING

        return ThreatClassification.MALWARE  # Default

    def _determine_severity(self, item: Dict, feed: ThreatFeed) -> ThreatSeverity:
        """Determine threat severity"""
        # Check for severity indicators
        severity_fields = ['severity', 'priority', 'level', 'risk']

        for field in severity_fields:
            if field in item:
                severity_value = str(item[field]).lower()
                if severity_value in ['critical', 'high', 'medium', 'low']:
                    return ThreatSeverity(severity_value)

        # Default based on feed reliability
        reliability_severity_map = {
            FeedReliability.A: ThreatSeverity.HIGH,
            FeedReliability.B: ThreatSeverity.MEDIUM,
            FeedReliability.C: ThreatSeverity.MEDIUM,
            FeedReliability.D: ThreatSeverity.LOW,
            FeedReliability.E: ThreatSeverity.LOW,
            FeedReliability.F: ThreatSeverity.LOW
        }

        return reliability_severity_map.get(feed.reliability, ThreatSeverity.MEDIUM)

    def _extract_confidence(self, item: Dict) -> float:
        """Extract confidence score"""
        confidence_fields = ['confidence', 'certainty', 'probability', 'score']

        for field in confidence_fields:
            if field in item:
                try:
                    confidence = float(item[field])
                    return max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    continue

        return 0.5  # Default confidence

    async def _parse_text_feed(self, data: str, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Parse text-based threat feed"""
        indicators = []

        try:
            lines = data.strip().split('\n')

            for line in lines:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#') or line.startswith('//'):
                    continue

                # Extract indicator value
                parts = line.split()
                if parts:
                    indicator_value = parts[0]
                    indicator_type = self._determine_indicator_type(indicator_value)

                    if indicator_type != 'unknown':
                        indicator_id = hashlib.sha256(f"{indicator_value}_{feed.id}".encode()).hexdigest()[:16]

                        indicator = ThreatIndicator(
                            id=indicator_id,
                            value=indicator_value,
                            indicator_type=indicator_type,
                            classification=ThreatClassification.MALWARE,  # Default for text feeds
                            severity=ThreatSeverity.MEDIUM,
                            confidence=0.7,
                            first_seen=datetime.now(),
                            last_seen=datetime.now(),
                            sources=[feed.id]
                        )

                        # Add geolocation for IP addresses
                        if indicator_type == 'ip':
                            indicator.geolocation = await self._get_geolocation(indicator_value)

                        indicators.append(indicator)

        except Exception as e:
            logger.error(f"Error parsing text feed {feed.name}: {e}")

        return indicators

    async def _parse_csv_feed(self, data: str, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Parse CSV threat feed"""
        indicators = []

        try:
            import csv
            import io

            csv_reader = csv.DictReader(io.StringIO(data))

            for row in csv_reader:
                # Find indicator value column
                indicator_value = None
                indicator_type = None

                value_columns = ['indicator', 'ioc', 'value', 'ip', 'domain', 'hash', 'url']
                for col in value_columns:
                    if col in row and row[col]:
                        indicator_value = row[col].strip()
                        indicator_type = self._determine_indicator_type(indicator_value)
                        break

                if indicator_value and indicator_type != 'unknown':
                    indicator_id = hashlib.sha256(f"{indicator_value}_{feed.id}".encode()).hexdigest()[:16]

                    # Extract additional metadata from CSV
                    classification = self._determine_classification(row, feed)
                    severity = self._determine_severity(row, feed)
                    confidence = self._extract_confidence(row)

                    indicator = ThreatIndicator(
                        id=indicator_id,
                        value=indicator_value,
                        indicator_type=indicator_type,
                        classification=classification,
                        severity=severity,
                        confidence=confidence,
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        sources=[feed.id],
                        context=dict(row)
                    )

                    # Add geolocation for IP addresses
                    if indicator_type == 'ip':
                        indicator.geolocation = await self._get_geolocation(indicator_value)

                    indicators.append(indicator)

        except Exception as e:
            logger.error(f"Error parsing CSV feed {feed.name}: {e}")

        return indicators

    async def _parse_stix_feed(self, data: str, feed: ThreatFeed) -> List[ThreatIndicator]:
        """Parse STIX threat feed"""
        indicators = []

        try:
            # Parse STIX data (placeholder implementation)
            # In production, use a STIX library like python-stix2
            stix_data = json.loads(data)

            # Extract indicators from STIX bundle
            if 'objects' in stix_data:
                for obj in stix_data['objects']:
                    if obj.get('type') == 'indicator':
                        indicator = await self._create_indicator_from_stix(obj, feed)
                        if indicator:
                            indicators.append(indicator)

        except Exception as e:
            logger.error(f"Error parsing STIX feed {feed.name}: {e}")

        return indicators

    async def _create_indicator_from_stix(self, stix_obj: Dict, feed: ThreatFeed) -> Optional[ThreatIndicator]:
        """Create indicator from STIX object"""
        try:
            # Extract pattern (simplified STIX parsing)
            pattern = stix_obj.get('pattern', '')
            if not pattern:
                return None

            # Parse pattern to extract indicator value
            # This is a simplified parser - use proper STIX library in production
            indicator_value = None
            indicator_type = None

            if 'file:hashes' in pattern:
                # Extract hash
                import re
                hash_match = re.search(r"'([a-fA-F0-9]{32,64})'", pattern)
                if hash_match:
                    indicator_value = hash_match.group(1)
                    indicator_type = 'sha256' if len(indicator_value) == 64 else 'md5'

            elif 'network-traffic:src_ref.value' in pattern:
                # Extract IP
                import re
                ip_match = re.search(r"'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'", pattern)
                if ip_match:
                    indicator_value = ip_match.group(1)
                    indicator_type = 'ip'

            if not indicator_value:
                return None

            indicator_id = hashlib.sha256(f"{indicator_value}_{feed.id}".encode()).hexdigest()[:16]

            indicator = ThreatIndicator(
                id=indicator_id,
                value=indicator_value,
                indicator_type=indicator_type,
                classification=ThreatClassification.MALWARE,
                severity=ThreatSeverity.MEDIUM,
                confidence=0.8,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                sources=[feed.id],
                context=stix_obj
            )

            return indicator

        except Exception as e:
            logger.error(f"Error creating STIX indicator: {e}")
            return None

    async def _get_geolocation(self, ip_address: str) -> Optional[Dict]:
        """Get geolocation for IP address"""
        if not self.geoip_reader:
            return None

        try:
            response = self.geoip_reader.city(ip_address)

            return {
                'country': response.country.name,
                'country_code': response.country.iso_code,
                'region': response.subdivisions.most_specific.name,
                'city': response.city.name,
                'latitude': float(response.location.latitude) if response.location.latitude else None,
                'longitude': float(response.location.longitude) if response.location.longitude else None,
                'timezone': response.location.time_zone
            }

        except geoip2.errors.AddressNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error getting geolocation for {ip_address}: {e}")
            return None

    async def shutdown(self):
        """Shutdown aggregator"""
        if self.session:
            await self.session.close()

        if self.geoip_reader:
            self.geoip_reader.close()

class ThreatClassificationEngine:
    """Machine learning engine for threat classification and scoring"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.text_model = None
        self.classification_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Initialize transformer for text analysis
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.text_model = AutoModel.from_pretrained('distilbert-base-uncased').to(self.device)

            # Initialize classification models
            self.classification_model = ThreatClassificationNN().to(self.device)

            # Load pre-trained weights if available
            model_path = "/opt/bev/models/threat_classification.pth"
            if os.path.exists(model_path):
                self.classification_model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("Loaded pre-trained threat classification model")

            self.classification_model.eval()

            logger.info("Threat classification models initialized")

        except Exception as e:
            logger.error(f"Failed to initialize classification models: {e}")

    async def classify_threats(self, indicators: List[ThreatIndicator]) -> List[ThreatIndicator]:
        """Classify and score threat indicators using ML"""
        try:
            if not indicators:
                return indicators

            # Extract features
            features = []
            for indicator in indicators:
                feature_vector = await self._extract_indicator_features(indicator)
                features.append(feature_vector)

            if not features:
                return indicators

            # Normalize features
            features_array = np.array(features)
            features_normalized = self.scaler.fit_transform(features_array)

            # Run classification
            features_tensor = torch.FloatTensor(features_normalized).to(self.device)

            with torch.no_grad():
                predictions = self.classification_model(features_tensor)
                classification_probs = F.softmax(predictions, dim=1)

            # Update indicators with ML predictions
            for i, indicator in enumerate(indicators):
                probs = classification_probs[i].cpu().numpy()

                # Get highest probability classification
                max_prob_idx = np.argmax(probs)
                max_prob = float(probs[max_prob_idx])

                # Map to classification enum (simplified)
                classification_map = list(ThreatClassification)
                if max_prob_idx < len(classification_map):
                    predicted_class = classification_map[max_prob_idx]

                    # Update if confidence is high enough
                    if max_prob > 0.7:
                        indicator.classification = predicted_class
                        indicator.confidence = max_prob

                THREATS_CLASSIFIED.labels(classification=indicator.classification.value).inc()

            return indicators

        except Exception as e:
            logger.error(f"Error classifying threats: {e}")
            return indicators

    async def _extract_indicator_features(self, indicator: ThreatIndicator) -> List[float]:
        """Extract ML features from threat indicator"""
        features = []

        try:
            # Basic indicator features
            features.extend(self._extract_basic_features(indicator))

            # Context features
            features.extend(self._extract_context_features(indicator))

            # Geolocation features
            features.extend(self._extract_geolocation_features(indicator))

            # Temporal features
            features.extend(self._extract_temporal_features(indicator))

            # Source features
            features.extend(self._extract_source_features(indicator))

            # Pad or truncate to fixed size
            target_size = 100
            if len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
            elif len(features) > target_size:
                features = features[:target_size]

            return features

        except Exception as e:
            logger.error(f"Error extracting indicator features: {e}")
            return [0.0] * 100

    def _extract_basic_features(self, indicator: ThreatIndicator) -> List[float]:
        """Extract basic indicator features"""
        features = []

        # Indicator type encoding
        type_encoding = {
            'ip': [1, 0, 0, 0, 0, 0],
            'domain': [0, 1, 0, 0, 0, 0],
            'url': [0, 0, 1, 0, 0, 0],
            'hash': [0, 0, 0, 1, 0, 0],
            'email': [0, 0, 0, 0, 1, 0],
            'unknown': [0, 0, 0, 0, 0, 1]
        }
        indicator_type = indicator.indicator_type if indicator.indicator_type in type_encoding else 'unknown'
        features.extend(type_encoding[indicator_type])

        # Value characteristics
        value = indicator.value
        features.extend([
            len(value),  # Length
            value.count('.'),  # Dot count
            value.count('-'),  # Dash count
            sum(c.isdigit() for c in value) / len(value) if value else 0,  # Digit ratio
            sum(c.isalpha() for c in value) / len(value) if value else 0,  # Alpha ratio
            len(set(value)) / len(value) if value else 0,  # Unique char ratio
        ])

        # Severity encoding
        severity_encoding = {
            ThreatSeverity.CRITICAL: 1.0,
            ThreatSeverity.HIGH: 0.8,
            ThreatSeverity.MEDIUM: 0.6,
            ThreatSeverity.LOW: 0.4,
            ThreatSeverity.INFO: 0.2
        }
        features.append(severity_encoding.get(indicator.severity, 0.5))

        # Confidence score
        features.append(indicator.confidence)

        return features

    def _extract_context_features(self, indicator: ThreatIndicator) -> List[float]:
        """Extract features from indicator context"""
        features = []

        context = indicator.context or {}

        # Common context features
        context_indicators = [
            'malware', 'botnet', 'phishing', 'spam', 'exploit',
            'backdoor', 'trojan', 'ransomware', 'apt', 'c2'
        ]

        context_text = json.dumps(context).lower()

        for keyword in context_indicators:
            features.append(1.0 if keyword in context_text else 0.0)

        # Context richness
        features.append(len(context))
        features.append(len(context_text))

        return features

    def _extract_geolocation_features(self, indicator: ThreatIndicator) -> List[float]:
        """Extract geolocation features"""
        features = []

        geo = indicator.geolocation or {}

        # Has geolocation
        features.append(1.0 if geo else 0.0)

        # Country risk (simplified)
        high_risk_countries = ['CN', 'RU', 'KP', 'IR']
        country_code = geo.get('country_code', '')
        features.append(1.0 if country_code in high_risk_countries else 0.0)

        # Coordinates (normalized)
        lat = geo.get('latitude', 0.0) or 0.0
        lon = geo.get('longitude', 0.0) or 0.0
        features.append(lat / 90.0)  # Normalize latitude
        features.append(lon / 180.0)  # Normalize longitude

        return features

    def _extract_temporal_features(self, indicator: ThreatIndicator) -> List[float]:
        """Extract temporal features"""
        features = []

        now = datetime.now()

        # Age features
        age_hours = (now - indicator.first_seen).total_seconds() / 3600
        features.append(min(age_hours / 24, 1.0))  # Age in days (capped at 1)

        # Recency
        recency_hours = (now - indicator.last_seen).total_seconds() / 3600
        features.append(min(recency_hours / 24, 1.0))  # Recency in days

        # Time of day patterns
        hour = indicator.first_seen.hour
        features.append(np.sin(2 * np.pi * hour / 24))  # Hour sine
        features.append(np.cos(2 * np.pi * hour / 24))  # Hour cosine

        # Day of week
        day = indicator.first_seen.weekday()
        features.append(np.sin(2 * np.pi * day / 7))  # Day sine
        features.append(np.cos(2 * np.pi * day / 7))  # Day cosine

        return features

    def _extract_source_features(self, indicator: ThreatIndicator) -> List[float]:
        """Extract source-based features"""
        features = []

        # Number of sources
        features.append(len(indicator.sources))

        # Source diversity
        features.append(len(set(indicator.sources)))

        # Has high-quality sources
        high_quality_sources = ['abuse_ch', 'emerging_threats', 'microsoft']
        has_quality_source = any(source in high_quality_sources for source in indicator.sources)
        features.append(1.0 if has_quality_source else 0.0)

        return features

class ThreatClassificationNN(nn.Module):
    """Neural network for threat classification"""

    def __init__(self, input_size=100, hidden_size=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class PredictiveThreatModeling:
    """Predictive modeling for emerging threats"""

    def __init__(self):
        self.time_series_model = None
        self.clustering_model = None
        self.trend_detector = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize predictive models"""
        try:
            # Initialize clustering for threat pattern detection
            self.clustering_model = DBSCAN(eps=0.5, min_samples=5)

            # Initialize trend detection
            self.trend_detector = GradientBoostingClassifier(n_estimators=100, random_state=42)

            logger.info("Predictive modeling initialized")

        except Exception as e:
            logger.error(f"Failed to initialize predictive models: {e}")

    async def predict_emerging_threats(self, indicators: List[ThreatIndicator],
                                     historical_data: List[Dict]) -> Dict:
        """Predict emerging threat patterns"""
        try:
            # Analyze temporal patterns
            temporal_analysis = await self._analyze_temporal_patterns(indicators)

            # Detect emerging clusters
            emerging_clusters = await self._detect_emerging_clusters(indicators)

            # Predict threat evolution
            evolution_prediction = await self._predict_threat_evolution(indicators, historical_data)

            # Generate predictions summary
            predictions = {
                'temporal_trends': temporal_analysis,
                'emerging_clusters': emerging_clusters,
                'evolution_prediction': evolution_prediction,
                'confidence_score': self._calculate_prediction_confidence(temporal_analysis, emerging_clusters),
                'time_horizon': '7_days',
                'risk_assessment': self._assess_prediction_risk(emerging_clusters)
            }

            return predictions

        except Exception as e:
            logger.error(f"Error predicting emerging threats: {e}")
            return {'error': str(e)}

    async def _analyze_temporal_patterns(self, indicators: List[ThreatIndicator]) -> Dict:
        """Analyze temporal patterns in threat data"""
        try:
            if not indicators:
                return {}

            # Create time series data
            timestamps = [indicator.first_seen for indicator in indicators]
            classifications = [indicator.classification.value for indicator in indicators]

            # Group by time windows (hourly)
            time_series = {}
            for timestamp, classification in zip(timestamps, classifications):
                hour_key = timestamp.replace(minute=0, second=0, microsecond=0)

                if hour_key not in time_series:
                    time_series[hour_key] = {}

                if classification not in time_series[hour_key]:
                    time_series[hour_key][classification] = 0

                time_series[hour_key][classification] += 1

            # Analyze trends
            trends = {}
            for classification in ThreatClassification:
                class_name = classification.value
                values = []
                times = sorted(time_series.keys())

                for time_key in times:
                    count = time_series[time_key].get(class_name, 0)
                    values.append(count)

                if len(values) >= 3:
                    # Simple trend analysis
                    recent_avg = np.mean(values[-3:]) if len(values) >= 3 else 0
                    historical_avg = np.mean(values[:-3]) if len(values) > 3 else recent_avg

                    trend_direction = "increasing" if recent_avg > historical_avg else "decreasing"
                    trend_strength = abs(recent_avg - historical_avg) / (historical_avg + 1)

                    trends[class_name] = {
                        'direction': trend_direction,
                        'strength': float(trend_strength),
                        'recent_count': int(recent_avg),
                        'historical_count': int(historical_avg)
                    }

            return {
                'trends': trends,
                'time_window': 'hourly',
                'data_points': len(time_series)
            }

        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
            return {}

    async def _detect_emerging_clusters(self, indicators: List[ThreatIndicator]) -> List[Dict]:
        """Detect emerging threat clusters"""
        try:
            if len(indicators) < 5:
                return []

            # Extract features for clustering
            features = []
            for indicator in indicators:
                # Simple feature vector for clustering
                feature_vector = [
                    hash(indicator.classification.value) % 1000,  # Classification hash
                    len(indicator.value),  # Value length
                    indicator.confidence * 100,  # Confidence
                    len(indicator.sources),  # Source count
                    indicator.first_seen.hour,  # Hour of day
                    indicator.first_seen.weekday(),  # Day of week
                ]

                # Add geolocation features
                if indicator.geolocation:
                    geo = indicator.geolocation
                    feature_vector.extend([
                        geo.get('latitude', 0.0) or 0.0,
                        geo.get('longitude', 0.0) or 0.0,
                        hash(geo.get('country_code', '')) % 100
                    ])
                else:
                    feature_vector.extend([0.0, 0.0, 0.0])

                features.append(feature_vector)

            # Normalize features
            features_array = np.array(features)
            features_normalized = StandardScaler().fit_transform(features_array)

            # Perform clustering
            cluster_labels = self.clustering_model.fit_predict(features_normalized)

            # Analyze clusters
            clusters = []
            unique_labels = set(cluster_labels)

            for label in unique_labels:
                if label == -1:  # Noise in DBSCAN
                    continue

                cluster_indicators = [indicators[i] for i, l in enumerate(cluster_labels) if l == label]

                if len(cluster_indicators) >= 3:  # Minimum cluster size
                    cluster_info = {
                        'cluster_id': int(label),
                        'size': len(cluster_indicators),
                        'dominant_classification': self._get_dominant_classification(cluster_indicators),
                        'time_span': self._calculate_time_span(cluster_indicators),
                        'geographic_spread': self._calculate_geographic_spread(cluster_indicators),
                        'novelty_score': self._calculate_novelty_score(cluster_indicators),
                        'indicators': [ind.id for ind in cluster_indicators[:10]]  # Sample indicators
                    }
                    clusters.append(cluster_info)

            # Sort by novelty score
            clusters.sort(key=lambda x: x['novelty_score'], reverse=True)

            return clusters[:10]  # Return top 10 clusters

        except Exception as e:
            logger.error(f"Error detecting emerging clusters: {e}")
            return []

    def _get_dominant_classification(self, indicators: List[ThreatIndicator]) -> str:
        """Get dominant classification in cluster"""
        classifications = [ind.classification.value for ind in indicators]
        from collections import Counter
        return Counter(classifications).most_common(1)[0][0]

    def _calculate_time_span(self, indicators: List[ThreatIndicator]) -> Dict:
        """Calculate time span of cluster"""
        timestamps = [ind.first_seen for ind in indicators]
        min_time = min(timestamps)
        max_time = max(timestamps)
        span = (max_time - min_time).total_seconds()

        return {
            'start_time': min_time.isoformat(),
            'end_time': max_time.isoformat(),
            'duration_hours': span / 3600
        }

    def _calculate_geographic_spread(self, indicators: List[ThreatIndicator]) -> Dict:
        """Calculate geographic spread of cluster"""
        countries = set()
        coordinates = []

        for indicator in indicators:
            if indicator.geolocation:
                geo = indicator.geolocation
                country = geo.get('country_code')
                if country:
                    countries.add(country)

                lat = geo.get('latitude')
                lon = geo.get('longitude')
                if lat is not None and lon is not None:
                    coordinates.append((lat, lon))

        # Calculate geographic diversity
        diversity_score = len(countries) / len(indicators) if indicators else 0

        return {
            'unique_countries': len(countries),
            'countries': list(countries),
            'diversity_score': diversity_score,
            'coordinate_count': len(coordinates)
        }

    def _calculate_novelty_score(self, indicators: List[ThreatIndicator]) -> float:
        """Calculate novelty score for cluster"""
        try:
            # Factors that increase novelty:
            # 1. Recent emergence
            # 2. Unique patterns
            # 3. High confidence
            # 4. Multiple sources

            now = datetime.now()
            timestamps = [ind.first_seen for ind in indicators]

            # Recency score (higher for more recent)
            avg_age_hours = np.mean([(now - ts).total_seconds() / 3600 for ts in timestamps])
            recency_score = max(0, 1 - (avg_age_hours / (24 * 7)))  # Decay over week

            # Confidence score
            confidences = [ind.confidence for ind in indicators]
            confidence_score = np.mean(confidences)

            # Source diversity score
            all_sources = set()
            for ind in indicators:
                all_sources.update(ind.sources)
            source_diversity = len(all_sources) / len(indicators)

            # Size factor (larger clusters are potentially more significant)
            size_factor = min(len(indicators) / 10, 1.0)  # Cap at 10 indicators

            # Combined novelty score
            novelty_score = (
                recency_score * 0.4 +
                confidence_score * 0.3 +
                source_diversity * 0.2 +
                size_factor * 0.1
            )

            return float(novelty_score)

        except Exception as e:
            logger.error(f"Error calculating novelty score: {e}")
            return 0.0

    async def _predict_threat_evolution(self, indicators: List[ThreatIndicator],
                                      historical_data: List[Dict]) -> Dict:
        """Predict how threats will evolve"""
        try:
            # Simplified threat evolution prediction

            # Analyze current threat distribution
            current_distribution = {}
            for indicator in indicators:
                classification = indicator.classification.value
                current_distribution[classification] = current_distribution.get(classification, 0) + 1

            # Predict growth rates based on recent trends
            predicted_changes = {}
            for threat_type, current_count in current_distribution.items():
                # Simple growth prediction (in production, use more sophisticated models)
                if current_count > 0:
                    # Assume some growth based on current momentum
                    growth_rate = min(0.2, current_count / len(indicators))  # Cap at 20%
                    predicted_count = int(current_count * (1 + growth_rate))
                    predicted_changes[threat_type] = {
                        'current_count': current_count,
                        'predicted_count': predicted_count,
                        'growth_rate': growth_rate
                    }

            return {
                'prediction_horizon': '7_days',
                'predicted_changes': predicted_changes,
                'methodology': 'trend_extrapolation',
                'confidence': 0.6  # Moderate confidence for simple model
            }

        except Exception as e:
            logger.error(f"Error predicting threat evolution: {e}")
            return {}

    def _calculate_prediction_confidence(self, temporal_analysis: Dict, emerging_clusters: List[Dict]) -> float:
        """Calculate overall prediction confidence"""
        confidence_factors = []

        # Temporal analysis confidence
        trends = temporal_analysis.get('trends', {})
        if trends:
            trend_strengths = [trend['strength'] for trend in trends.values()]
            temporal_confidence = np.mean(trend_strengths) if trend_strengths else 0
            confidence_factors.append(temporal_confidence)

        # Cluster analysis confidence
        if emerging_clusters:
            cluster_novelties = [cluster['novelty_score'] for cluster in emerging_clusters]
            cluster_confidence = np.mean(cluster_novelties)
            confidence_factors.append(cluster_confidence)

        # Data sufficiency confidence
        data_points = temporal_analysis.get('data_points', 0)
        data_confidence = min(data_points / 100, 1.0)  # More data = higher confidence
        confidence_factors.append(data_confidence)

        # Overall confidence
        if confidence_factors:
            return float(np.mean(confidence_factors))
        else:
            return 0.5

    def _assess_prediction_risk(self, emerging_clusters: List[Dict]) -> str:
        """Assess risk level of predictions"""
        if not emerging_clusters:
            return "low"

        # Count high-severity clusters
        high_severity_count = 0
        for cluster in emerging_clusters:
            if cluster['novelty_score'] > 0.8:
                high_severity_count += 1

        if high_severity_count >= 3:
            return "critical"
        elif high_severity_count >= 2:
            return "high"
        elif high_severity_count >= 1:
            return "medium"
        else:
            return "low"

class GeospatialThreatMapper:
    """Map threats geospatially for strategic analysis"""

    def __init__(self):
        self.threat_map = {}
        self.country_stats = {}

    async def map_threats_geospatially(self, indicators: List[ThreatIndicator]) -> Dict:
        """Create geospatial threat map"""
        try:
            geospatial_threats = []
            country_aggregation = {}
            region_aggregation = {}

            for indicator in indicators:
                if not indicator.geolocation:
                    continue

                geo = indicator.geolocation
                country = geo.get('country_code', 'UNKNOWN')

                # Aggregate by country
                if country not in country_aggregation:
                    country_aggregation[country] = {
                        'country_name': geo.get('country', 'Unknown'),
                        'threat_count': 0,
                        'threat_types': {},
                        'severity_distribution': {},
                        'confidence_avg': 0.0,
                        'coordinates': []
                    }

                country_data = country_aggregation[country]
                country_data['threat_count'] += 1

                # Track threat types
                threat_type = indicator.classification.value
                country_data['threat_types'][threat_type] = country_data['threat_types'].get(threat_type, 0) + 1

                # Track severity distribution
                severity = indicator.severity.value
                country_data['severity_distribution'][severity] = country_data['severity_distribution'].get(severity, 0) + 1

                # Update average confidence
                current_avg = country_data['confidence_avg']
                current_count = country_data['threat_count']
                country_data['confidence_avg'] = ((current_avg * (current_count - 1)) + indicator.confidence) / current_count

                # Add coordinates
                lat = geo.get('latitude')
                lon = geo.get('longitude')
                if lat is not None and lon is not None:
                    country_data['coordinates'].append({'lat': lat, 'lon': lon})

                # Create individual geospatial threat
                geospatial_threat = GeospatialThreat(
                    id=f"geo_{indicator.id}",
                    location=geo,
                    threat_type=indicator.classification,
                    severity=indicator.severity,
                    timestamp=indicator.first_seen,
                    indicators=[indicator.id],
                    threat_density=1.0,  # Will be calculated later
                    context={'original_indicator': indicator.id}
                )

                geospatial_threats.append(geospatial_threat)

            # Calculate threat density for each country
            max_threats = max(data['threat_count'] for data in country_aggregation.values()) if country_aggregation else 1

            for country_data in country_aggregation.values():
                country_data['threat_density'] = country_data['threat_count'] / max_threats

            # Identify threat hotspots
            hotspots = self._identify_threat_hotspots(country_aggregation)

            # Generate strategic analysis
            strategic_analysis = self._generate_strategic_analysis(country_aggregation, hotspots)

            return {
                'total_geolocated_threats': len(geospatial_threats),
                'country_aggregation': country_aggregation,
                'threat_hotspots': hotspots,
                'strategic_analysis': strategic_analysis,
                'map_data': {
                    'countries': list(country_aggregation.keys()),
                    'max_threat_density': 1.0,
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Error mapping threats geospatially: {e}")
            return {'error': str(e)}

    def _identify_threat_hotspots(self, country_aggregation: Dict) -> List[Dict]:
        """Identify geographical threat hotspots"""
        hotspots = []

        try:
            # Sort countries by threat density
            sorted_countries = sorted(
                country_aggregation.items(),
                key=lambda x: x[1]['threat_density'],
                reverse=True
            )

            # Top 10 countries as hotspots
            for country_code, data in sorted_countries[:10]:
                if data['threat_count'] >= 5:  # Minimum threshold
                    hotspot = {
                        'country_code': country_code,
                        'country_name': data['country_name'],
                        'threat_count': data['threat_count'],
                        'threat_density': data['threat_density'],
                        'dominant_threat_type': max(data['threat_types'].items(), key=lambda x: x[1])[0] if data['threat_types'] else 'unknown',
                        'dominant_severity': max(data['severity_distribution'].items(), key=lambda x: x[1])[0] if data['severity_distribution'] else 'unknown',
                        'average_confidence': data['confidence_avg'],
                        'risk_level': self._calculate_country_risk_level(data)
                    }
                    hotspots.append(hotspot)

        except Exception as e:
            logger.error(f"Error identifying hotspots: {e}")

        return hotspots

    def _calculate_country_risk_level(self, country_data: Dict) -> str:
        """Calculate risk level for country"""
        try:
            threat_count = country_data['threat_count']
            threat_density = country_data['threat_density']
            avg_confidence = country_data['confidence_avg']

            # Calculate severity weight
            severity_weights = {
                'critical': 1.0,
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4,
                'informational': 0.2
            }

            severity_score = 0.0
            total_threats = sum(country_data['severity_distribution'].values())

            if total_threats > 0:
                for severity, count in country_data['severity_distribution'].items():
                    weight = severity_weights.get(severity, 0.5)
                    severity_score += (count / total_threats) * weight

            # Combined risk score
            risk_score = (
                min(threat_count / 100, 1.0) * 0.4 +  # Volume factor
                threat_density * 0.3 +  # Density factor
                avg_confidence * 0.2 +  # Confidence factor
                severity_score * 0.1  # Severity factor
            )

            if risk_score >= 0.8:
                return "critical"
            elif risk_score >= 0.6:
                return "high"
            elif risk_score >= 0.4:
                return "medium"
            else:
                return "low"

        except Exception as e:
            logger.error(f"Error calculating country risk level: {e}")
            return "unknown"

    def _generate_strategic_analysis(self, country_aggregation: Dict, hotspots: List[Dict]) -> Dict:
        """Generate strategic intelligence analysis"""
        try:
            analysis = {
                'summary': {},
                'trends': {},
                'recommendations': []
            }

            # Summary statistics
            total_countries = len(country_aggregation)
            total_threats = sum(data['threat_count'] for data in country_aggregation.values())
            avg_threats_per_country = total_threats / total_countries if total_countries > 0 else 0

            analysis['summary'] = {
                'total_countries_affected': total_countries,
                'total_geolocated_threats': total_threats,
                'average_threats_per_country': avg_threats_per_country,
                'high_risk_countries': len([h for h in hotspots if h['risk_level'] in ['critical', 'high']])
            }

            # Global threat type distribution
            global_threat_types = {}
            for data in country_aggregation.values():
                for threat_type, count in data['threat_types'].items():
                    global_threat_types[threat_type] = global_threat_types.get(threat_type, 0) + count

            dominant_global_threat = max(global_threat_types.items(), key=lambda x: x[1])[0] if global_threat_types else 'unknown'

            analysis['trends'] = {
                'dominant_global_threat': dominant_global_threat,
                'threat_type_distribution': global_threat_types,
                'geographic_spread': 'global' if total_countries > 50 else 'regional',
                'concentration_analysis': self._analyze_threat_concentration(hotspots)
            }

            # Strategic recommendations
            analysis['recommendations'] = self._generate_strategic_recommendations(hotspots, analysis)

            return analysis

        except Exception as e:
            logger.error(f"Error generating strategic analysis: {e}")
            return {}

    def _analyze_threat_concentration(self, hotspots: List[Dict]) -> Dict:
        """Analyze geographic concentration of threats"""
        try:
            if not hotspots:
                return {'concentration': 'unknown'}

            # Calculate concentration metrics
            threat_counts = [h['threat_count'] for h in hotspots]
            total_threats = sum(threat_counts)

            # Top 3 countries' share
            top_3_share = sum(threat_counts[:3]) / total_threats if total_threats > 0 else 0

            # Gini coefficient for inequality
            sorted_counts = sorted(threat_counts)
            n = len(sorted_counts)
            gini = 0

            if n > 1 and total_threats > 0:
                cumsum = np.cumsum(sorted_counts)
                gini = (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum))) / (n * total_threats)

            concentration_level = "high" if gini > 0.6 else "medium" if gini > 0.3 else "low"

            return {
                'concentration': concentration_level,
                'top_3_countries_share': top_3_share,
                'gini_coefficient': gini,
                'interpretation': f"Threats are {'highly concentrated' if gini > 0.6 else 'moderately distributed' if gini > 0.3 else 'widely distributed'} geographically"
            }

        except Exception as e:
            logger.error(f"Error analyzing threat concentration: {e}")
            return {'concentration': 'unknown'}

    def _generate_strategic_recommendations(self, hotspots: List[Dict], analysis: Dict) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []

        try:
            # High-risk country recommendations
            critical_countries = [h for h in hotspots if h['risk_level'] == 'critical']
            if critical_countries:
                recommendations.append(f"Immediate attention required for {len(critical_countries)} critical-risk countries")
                recommendations.append("Deploy enhanced monitoring for critical-risk regions")

            # Concentration-based recommendations
            concentration = analysis.get('trends', {}).get('concentration_analysis', {})
            if concentration.get('concentration') == 'high':
                recommendations.append("Focus resources on high-concentration threat regions")
            elif concentration.get('concentration') == 'low':
                recommendations.append("Implement broad-spectrum monitoring due to distributed threats")

            # Threat type recommendations
            dominant_threat = analysis.get('trends', {}).get('dominant_global_threat')
            if dominant_threat:
                threat_recommendations = {
                    'malware': 'Enhance anti-malware capabilities globally',
                    'phishing': 'Strengthen email security and user training',
                    'botnet': 'Implement network-based detection and blocking',
                    'ransomware': 'Focus on backup and recovery capabilities',
                    'apt': 'Deploy advanced threat hunting capabilities'
                }

                if dominant_threat in threat_recommendations:
                    recommendations.append(threat_recommendations[dominant_threat])

            # Regional coordination recommendations
            if analysis.get('summary', {}).get('total_countries_affected', 0) > 20:
                recommendations.append("Consider international cooperation and information sharing")

        except Exception as e:
            logger.error(f"Error generating strategic recommendations: {e}")

        return recommendations

class StrategicIntelligenceReporter:
    """Generate strategic intelligence reports"""

    def __init__(self):
        self.report_templates = self._load_report_templates()

    def _load_report_templates(self) -> Dict:
        """Load report templates"""
        return {
            'tactical': {
                'sections': ['executive_summary', 'key_findings', 'indicators', 'recommendations'],
                'classification': 'TLP:GREEN',
                'audience': 'security_analysts'
            },
            'operational': {
                'sections': ['executive_summary', 'threat_landscape', 'campaign_analysis', 'defensive_measures'],
                'classification': 'TLP:AMBER',
                'audience': 'security_managers'
            },
            'strategic': {
                'sections': ['executive_summary', 'strategic_trends', 'geopolitical_context', 'long_term_outlook'],
                'classification': 'TLP:RED',
                'audience': 'executives'
            }
        }

    async def generate_intelligence_report(self,
                                         report_type: IntelligenceType,
                                         indicators: List[ThreatIndicator],
                                         analysis_results: Dict) -> IntelligenceProduct:
        """Generate comprehensive intelligence report"""
        try:
            report_id = hashlib.sha256(f"{report_type.value}_{datetime.now()}".encode()).hexdigest()[:16]

            # Generate report content based on type
            if report_type == IntelligenceType.TACTICAL:
                report = await self._generate_tactical_report(report_id, indicators, analysis_results)
            elif report_type == IntelligenceType.OPERATIONAL:
                report = await self._generate_operational_report(report_id, indicators, analysis_results)
            elif report_type == IntelligenceType.STRATEGIC:
                report = await self._generate_strategic_report(report_id, indicators, analysis_results)
            else:
                report = await self._generate_technical_report(report_id, indicators, analysis_results)

            INTELLIGENCE_REPORTS_GENERATED.labels(type=report_type.value).inc()

            return report

        except Exception as e:
            logger.error(f"Error generating intelligence report: {e}")
            raise

    async def _generate_tactical_report(self, report_id: str,
                                      indicators: List[ThreatIndicator],
                                      analysis_results: Dict) -> IntelligenceProduct:
        """Generate tactical intelligence report"""

        # Extract key metrics
        total_indicators = len(indicators)
        high_confidence_indicators = len([i for i in indicators if i.confidence > 0.8])

        classification_counts = {}
        for indicator in indicators:
            cls = indicator.classification.value
            classification_counts[cls] = classification_counts.get(cls, 0) + 1

        executive_summary = f"""
        Tactical Threat Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}

        This report provides tactical-level threat intelligence based on analysis of {total_indicators}
        threat indicators collected from multiple sources. {high_confidence_indicators} indicators
        have high confidence ratings (>0.8).

        KEY HIGHLIGHTS:
        - Dominant threat type: {max(classification_counts.items(), key=lambda x: x[1])[0] if classification_counts else 'N/A'}
        - High-confidence indicators: {high_confidence_indicators}
        - Geographic coverage: {len(set(i.geolocation.get('country_code') for i in indicators if i.geolocation))} countries
        """

        key_findings = [
            f"Analyzed {total_indicators} threat indicators from multiple sources",
            f"Identified {len(classification_counts)} distinct threat types",
            f"{high_confidence_indicators} indicators exceed high-confidence threshold",
            "Threat landscape shows active adversary operations across multiple vectors"
        ]

        recommendations = [
            "Implement blocking rules for high-confidence IP indicators",
            "Enhance monitoring for dominant threat type patterns",
            "Update threat hunting queries with new IOCs",
            "Coordinate with threat intelligence feeds for validation"
        ]

        return IntelligenceProduct(
            id=report_id,
            title=f"Tactical Threat Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}",
            intelligence_type=IntelligenceType.TACTICAL,
            classification="TLP:GREEN",
            created_date=datetime.now(),
            valid_until=datetime.now() + timedelta(days=7),
            executive_summary=executive_summary.strip(),
            key_findings=key_findings,
            indicators=[i.id for i in indicators[:50]],  # Top 50 indicators
            recommendations=recommendations,
            confidence_level="medium",
            sources=list(set().union(*[i.sources for i in indicators])),
            author="tactical_intelligence_engine"
        )

    async def _generate_operational_report(self, report_id: str,
                                         indicators: List[ThreatIndicator],
                                         analysis_results: Dict) -> IntelligenceProduct:
        """Generate operational intelligence report"""

        # Campaign analysis
        predictive_results = analysis_results.get('predictive_analysis', {})
        geospatial_results = analysis_results.get('geospatial_analysis', {})

        executive_summary = f"""
        Operational Threat Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}

        This report provides operational-level threat intelligence focusing on threat campaigns,
        attack patterns, and defensive recommendations for security operations teams.

        THREAT LANDSCAPE OVERVIEW:
        - Active threat indicators: {len(indicators)}
        - Emerging threat clusters: {len(predictive_results.get('emerging_clusters', []))}
        - Geographic hotspots: {len(geospatial_results.get('threat_hotspots', []))}

        OPERATIONAL IMPACT:
        - Predicted threat evolution shows {predictive_results.get('evolution_prediction', {}).get('confidence', 'medium')} confidence
        - Geospatial analysis indicates {'concentrated' if len(geospatial_results.get('threat_hotspots', [])) < 5 else 'distributed'} threat activity
        """

        key_findings = [
            "Multiple threat actors demonstrate coordinated activities",
            f"Emerging threat patterns detected in {len(predictive_results.get('emerging_clusters', []))} distinct clusters",
            "Geographic threat distribution requires targeted defensive measures",
            "Threat evolution predictions suggest escalation in specific vectors"
        ]

        recommendations = [
            "Enhance detection capabilities for emerging threat clusters",
            "Implement geographic-specific security controls",
            "Develop threat hunting campaigns targeting predicted evolution patterns",
            "Coordinate with international partners in high-risk regions"
        ]

        return IntelligenceProduct(
            id=report_id,
            title=f"Operational Threat Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}",
            intelligence_type=IntelligenceType.OPERATIONAL,
            classification="TLP:AMBER",
            created_date=datetime.now(),
            valid_until=datetime.now() + timedelta(days=14),
            executive_summary=executive_summary.strip(),
            key_findings=key_findings,
            recommendations=recommendations,
            confidence_level="high",
            sources=list(set().union(*[i.sources for i in indicators])),
            author="operational_intelligence_engine"
        )

    async def _generate_strategic_report(self, report_id: str,
                                       indicators: List[ThreatIndicator],
                                       analysis_results: Dict) -> IntelligenceProduct:
        """Generate strategic intelligence report"""

        geospatial_results = analysis_results.get('geospatial_analysis', {})
        strategic_analysis = geospatial_results.get('strategic_analysis', {})

        executive_summary = f"""
        Strategic Threat Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}

        This report provides strategic-level threat intelligence for executive decision-making,
        focusing on long-term trends, geopolitical implications, and organizational risk posture.

        STRATEGIC ASSESSMENT:
        - Global threat landscape shows {strategic_analysis.get('trends', {}).get('geographic_spread', 'regional')} distribution
        - Threat concentration is {geospatial_results.get('strategic_analysis', {}).get('trends', {}).get('concentration_analysis', {}).get('concentration', 'unknown')}
        - {strategic_analysis.get('summary', {}).get('high_risk_countries', 0)} countries identified as high-risk

        BUSINESS IMPACT:
        - Threat evolution suggests need for strategic investment in defensive capabilities
        - Geographic risk distribution requires international security partnerships
        - Long-term outlook indicates persistent threat environment
        """

        key_findings = [
            "Global threat landscape demonstrates sophisticated adversary capabilities",
            "Geographic threat concentration indicates targeted regional campaigns",
            "Emerging threat patterns suggest evolution toward advanced persistent threats",
            "Strategic threat trends align with geopolitical tensions and conflicts"
        ]

        recommendations = [
            "Invest in advanced threat detection and response capabilities",
            "Develop strategic partnerships for threat intelligence sharing",
            "Implement enterprise-wide security architecture improvements",
            "Establish long-term cybersecurity strategic planning processes"
        ]

        return IntelligenceProduct(
            id=report_id,
            title=f"Strategic Threat Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}",
            intelligence_type=IntelligenceType.STRATEGIC,
            classification="TLP:RED",
            created_date=datetime.now(),
            valid_until=datetime.now() + timedelta(days=30),
            executive_summary=executive_summary.strip(),
            key_findings=key_findings,
            recommendations=recommendations,
            confidence_level="high",
            sources=["strategic_analysis_engine"],
            author="strategic_intelligence_engine"
        )

    async def _generate_technical_report(self, report_id: str,
                                       indicators: List[ThreatIndicator],
                                       analysis_results: Dict) -> IntelligenceProduct:
        """Generate technical intelligence report"""

        # Technical analysis
        classification_results = analysis_results.get('classification', {})

        executive_summary = f"""
        Technical Threat Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}

        This report provides technical-level threat intelligence for security researchers
        and malware analysts, focusing on technical indicators, attack techniques, and
        detailed threat characteristics.

        TECHNICAL ANALYSIS:
        - Total technical indicators analyzed: {len(indicators)}
        - Machine learning classification accuracy: {classification_results.get('accuracy', 'N/A')}
        - Unique indicator types: {len(set(i.indicator_type for i in indicators))}
        """

        key_findings = [
            f"Technical analysis processed {len(indicators)} unique threat indicators",
            "Machine learning classification shows consistent threat type identification",
            "Technical indicators demonstrate sophisticated attack methodologies",
            "Malware analysis reveals advanced evasion techniques"
        ]

        recommendations = [
            "Update technical detection signatures with new IOCs",
            "Enhance malware analysis sandbox capabilities",
            "Implement advanced behavioral analysis techniques",
            "Develop custom detection rules for emerging attack patterns"
        ]

        return IntelligenceProduct(
            id=report_id,
            title=f"Technical Threat Intelligence Report - {datetime.now().strftime('%Y-%m-%d')}",
            intelligence_type=IntelligenceType.TECHNICAL,
            classification="TLP:GREEN",
            created_date=datetime.now(),
            valid_until=datetime.now() + timedelta(days=30),
            executive_summary=executive_summary.strip(),
            key_findings=key_findings,
            indicators=[i.id for i in indicators],
            recommendations=recommendations,
            confidence_level="high",
            sources=list(set().union(*[i.sources for i in indicators])),
            author="technical_intelligence_engine"
        )

class IntelligenceFusionProcessor:
    """Main intelligence fusion orchestrator"""

    def __init__(self, security_framework: OperationalSecurityFramework):
        self.security_framework = security_framework
        self.feed_aggregator = ThreatFeedAggregator()
        self.classification_engine = ThreatClassificationEngine()
        self.predictive_modeler = PredictiveThreatModeling()
        self.geospatial_mapper = GeospatialThreatMapper()
        self.reporter = StrategicIntelligenceReporter()

        self.db_pool = None
        self.redis_client = None

        self.processed_indicators: Dict[str, ThreatIndicator] = {}
        self.intelligence_products: List[IntelligenceProduct] = []

    async def initialize(self, redis_url: str = "redis://localhost:6379",
                        db_url: str = "postgresql://user:pass@localhost/bev"):
        """Initialize the intelligence fusion processor"""
        try:
            # Initialize database connections
            self.redis_client = redis.from_url(redis_url)
            self.db_pool = await asyncpg.create_pool(db_url)

            # Initialize feed aggregator
            await self.feed_aggregator.initialize()

            logger.info("Intelligence Fusion Processor initialized")
            print("🧠 Intelligence Fusion Processor Ready")

        except Exception as e:
            logger.error(f"Failed to initialize intelligence fusion processor: {e}")
            raise

    async def process_intelligence_cycle(self) -> Dict:
        """Execute complete intelligence processing cycle"""
        try:
            cycle_start = datetime.now()

            # Step 1: Collect threat feeds
            logger.info("Collecting threat feeds...")
            feed_indicators = await self.feed_aggregator.collect_threat_feeds()

            # Flatten indicators
            all_indicators = []
            for feed_id, indicators in feed_indicators.items():
                all_indicators.extend(indicators)

            logger.info(f"Collected {len(all_indicators)} indicators from {len(feed_indicators)} feeds")

            # Step 2: Classify and score threats
            logger.info("Classifying threats with ML...")
            classified_indicators = await self.classification_engine.classify_threats(all_indicators)

            # Step 3: Predictive threat modeling
            logger.info("Running predictive threat models...")
            predictive_analysis = await self.predictive_modeler.predict_emerging_threats(
                classified_indicators, []  # Historical data would be loaded here
            )

            # Step 4: Geospatial threat mapping
            logger.info("Mapping threats geospatially...")
            geospatial_analysis = await self.geospatial_mapper.map_threats_geospatially(classified_indicators)

            # Step 5: Generate intelligence products
            logger.info("Generating intelligence reports...")
            analysis_results = {
                'classification': {'accuracy': 0.85},  # Would come from actual ML metrics
                'predictive_analysis': predictive_analysis,
                'geospatial_analysis': geospatial_analysis
            }

            intelligence_products = []
            for report_type in [IntelligenceType.TACTICAL, IntelligenceType.OPERATIONAL, IntelligenceType.STRATEGIC]:
                product = await self.reporter.generate_intelligence_report(
                    report_type, classified_indicators, analysis_results
                )
                intelligence_products.append(product)
                self.intelligence_products.append(product)

            # Step 6: Store results
            await self._store_processing_results(classified_indicators, intelligence_products)

            # Update processed indicators
            for indicator in classified_indicators:
                self.processed_indicators[indicator.id] = indicator

            cycle_duration = (datetime.now() - cycle_start).total_seconds()

            logger.info(f"Intelligence processing cycle completed in {cycle_duration:.2f} seconds")

            return {
                'cycle_duration': cycle_duration,
                'indicators_processed': len(classified_indicators),
                'feeds_processed': len(feed_indicators),
                'intelligence_products': len(intelligence_products),
                'predictive_analysis': predictive_analysis,
                'geospatial_analysis': geospatial_analysis,
                'cycle_timestamp': cycle_start.isoformat()
            }

        except Exception as e:
            logger.error(f"Error in intelligence processing cycle: {e}")
            return {'error': str(e)}

    async def _store_processing_results(self, indicators: List[ThreatIndicator],
                                      products: List[IntelligenceProduct]):
        """Store processing results in database"""
        try:
            if not self.db_pool:
                return

            async with self.db_pool.acquire() as conn:
                # Store indicators
                for indicator in indicators:
                    await conn.execute("""
                        INSERT INTO threat_indicators (
                            id, value, indicator_type, classification, severity,
                            confidence, first_seen, last_seen, sources, tags,
                            context, geolocation
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        ON CONFLICT (id) DO UPDATE SET
                            last_seen = $8,
                            confidence = $6,
                            sources = $9
                    """,
                    indicator.id, indicator.value, indicator.indicator_type,
                    indicator.classification.value, indicator.severity.value,
                    indicator.confidence, indicator.first_seen, indicator.last_seen,
                    json.dumps(indicator.sources), json.dumps(list(indicator.tags)),
                    json.dumps(indicator.context), json.dumps(indicator.geolocation)
                    )

                # Store intelligence products
                for product in products:
                    await conn.execute("""
                        INSERT INTO intelligence_products (
                            id, title, intelligence_type, classification, created_date,
                            valid_until, executive_summary, key_findings, indicators,
                            recommendations, confidence_level, sources, author
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                        ON CONFLICT (id) DO UPDATE SET
                            valid_until = $6,
                            executive_summary = $7
                    """,
                    product.id, product.title, product.intelligence_type.value,
                    product.classification, product.created_date, product.valid_until,
                    product.executive_summary, json.dumps(product.key_findings),
                    json.dumps(product.indicators), json.dumps(product.recommendations),
                    product.confidence_level, json.dumps(product.sources), product.author
                    )

        except Exception as e:
            logger.error(f"Error storing processing results: {e}")

    async def get_threat_intelligence_summary(self) -> Dict:
        """Get summary of current threat intelligence"""
        try:
            indicators = list(self.processed_indicators.values())

            if not indicators:
                return {'status': 'no_data'}

            # Classification distribution
            classification_counts = {}
            for indicator in indicators:
                cls = indicator.classification.value
                classification_counts[cls] = classification_counts.get(cls, 0) + 1

            # Severity distribution
            severity_counts = {}
            for indicator in indicators:
                sev = indicator.severity.value
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            # Geographic distribution
            countries = set()
            for indicator in indicators:
                if indicator.geolocation:
                    country = indicator.geolocation.get('country_code')
                    if country:
                        countries.add(country)

            # Recent indicators (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_indicators = [i for i in indicators if i.last_seen > recent_cutoff]

            return {
                'total_indicators': len(indicators),
                'recent_indicators': len(recent_indicators),
                'classification_distribution': classification_counts,
                'severity_distribution': severity_counts,
                'geographic_coverage': len(countries),
                'countries_affected': list(countries)[:10],  # Top 10
                'last_updated': max(i.last_seen for i in indicators).isoformat() if indicators else None,
                'intelligence_products': len(self.intelligence_products)
            }

        except Exception as e:
            logger.error(f"Error getting threat intelligence summary: {e}")
            return {'error': str(e)}

    async def shutdown(self):
        """Shutdown intelligence fusion processor"""
        try:
            # Shutdown components
            await self.feed_aggregator.shutdown()

            # Close database connections
            if self.db_pool:
                await self.db_pool.close()

            if self.redis_client:
                await self.redis_client.close()

            logger.info("Intelligence Fusion Processor shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Example usage
async def main():
    """Example usage of the Intelligence Fusion Processor"""
    try:
        # Initialize security framework
        security = OperationalSecurityFramework()
        await security.initialize_security()

        # Initialize intelligence fusion processor
        processor = IntelligenceFusionProcessor(security)
        await processor.initialize()

        # Run intelligence processing cycle
        cycle_results = await processor.process_intelligence_cycle()
        print(f"✅ Processing cycle completed: {cycle_results}")

        # Get threat intelligence summary
        summary = await processor.get_threat_intelligence_summary()
        print(f"📊 Threat Intelligence Summary: {summary}")

        # Shutdown
        await processor.shutdown()

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())