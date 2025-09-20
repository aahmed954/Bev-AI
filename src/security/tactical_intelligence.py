#!/usr/bin/env python3
"""
Tactical Intelligence Platform
Multi-INT fusion (SIGINT, HUMINT, OSINT, TECHINT) with advanced threat analysis
"""

import asyncio
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import networkx as nx
from collections import defaultdict, deque
import aiohttp
import asyncpg
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge
import logging
from .security_framework import OperationalSecurityFramework

# Metrics
INTEL_PROCESSING_TIME = Histogram('intel_processing_seconds', 'Time spent processing intelligence')
THREAT_DETECTIONS = Counter('threat_detections_total', 'Total threat detections', ['source', 'severity'])
ACTIVE_CAMPAIGNS = Gauge('active_campaigns', 'Number of active threat campaigns')
IOC_ENRICHMENTS = Counter('ioc_enrichments_total', 'Total IoC enrichments', ['type'])

logger = logging.getLogger(__name__)

class IntelType(Enum):
    """Intelligence collection types"""
    SIGINT = "signals_intelligence"
    HUMINT = "human_intelligence"
    OSINT = "open_source_intelligence"
    TECHINT = "technical_intelligence"
    GEOINT = "geospatial_intelligence"
    FININT = "financial_intelligence"

class ThreatSeverity(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "informational"

class CampaignStatus(Enum):
    """Campaign tracking statuses"""
    ACTIVE = "active"
    DORMANT = "dormant"
    CONCLUDED = "concluded"
    INVESTIGATING = "investigating"

@dataclass
class IntelligenceReport:
    """Structured intelligence report"""
    id: str
    source: str
    intel_type: IntelType
    content: str
    confidence: float
    timestamp: datetime
    classification: str = "TLP:WHITE"
    tags: Set[str] = field(default_factory=set)
    indicators: List[str] = field(default_factory=list)
    threat_actors: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    geolocation: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class ThreatActor:
    """Threat actor profile"""
    id: str
    name: str
    aliases: List[str]
    motivation: str
    sophistication: str
    attribution_confidence: float
    first_seen: datetime
    last_seen: datetime
    tools: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    campaigns: List[str] = field(default_factory=list)
    behavioral_patterns: Dict = field(default_factory=dict)

@dataclass
class ThreatCampaign:
    """Threat campaign tracking"""
    id: str
    name: str
    status: CampaignStatus
    threat_actors: List[str]
    start_date: datetime
    end_date: Optional[datetime]
    objectives: List[str]
    techniques: Set[str] = field(default_factory=set)
    indicators: Set[str] = field(default_factory=set)
    victims: List[str] = field(default_factory=list)
    timeline: List[Dict] = field(default_factory=list)
    confidence: float = 0.0

@dataclass
class IOCEnrichment:
    """Indicator of Compromise enrichment data"""
    ioc: str
    ioc_type: str
    first_seen: datetime
    last_seen: datetime
    reputation_score: float
    threat_associations: List[str]
    geolocation: Optional[Dict]
    whois_data: Optional[Dict]
    malware_families: List[str] = field(default_factory=list)
    campaigns: List[str] = field(default_factory=list)
    confidence: float = 0.0

class ThreatIntelligenceML:
    """Machine learning models for threat intelligence analysis"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.scaler = StandardScaler()
        self.clustering_model = None
        self.anomaly_detector = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Initialize transformer for text analysis
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = AutoModel.from_pretrained('distilbert-base-uncased').to(self.device)

            # Initialize clustering
            self.clustering_model = DBSCAN(eps=0.5, min_samples=3)

            # Initialize anomaly detection
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)

            logger.info("ML models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")

    async def analyze_threat_text(self, text: str) -> Dict:
        """Analyze threat intelligence text using transformer models"""
        try:
            # Tokenize and encode
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            # Extract features
            features = {
                'embedding': embeddings.tolist()[0],
                'threat_indicators': self._extract_threat_indicators(text),
                'sentiment_score': self._analyze_sentiment(embeddings),
                'complexity_score': self._calculate_complexity(text)
            }

            return features
        except Exception as e:
            logger.error(f"Error analyzing threat text: {e}")
            return {}

    def _extract_threat_indicators(self, text: str) -> List[str]:
        """Extract threat indicators from text"""
        indicators = []

        # Common threat keywords
        threat_keywords = [
            'malware', 'ransomware', 'phishing', 'backdoor', 'trojan',
            'botnet', 'apt', 'exploit', 'vulnerability', 'breach',
            'exfiltration', 'lateral movement', 'privilege escalation'
        ]

        text_lower = text.lower()
        for keyword in threat_keywords:
            if keyword in text_lower:
                indicators.append(keyword)

        return indicators

    def _analyze_sentiment(self, embeddings: np.ndarray) -> float:
        """Analyze sentiment of threat intelligence"""
        # Simple sentiment analysis based on embedding magnitude
        return float(np.linalg.norm(embeddings))

    def _calculate_complexity(self, text: str) -> float:
        """Calculate complexity score of threat description"""
        # Basic complexity metrics
        words = text.split()
        unique_words = set(words)

        if len(words) == 0:
            return 0.0

        complexity = len(unique_words) / len(words)
        return min(complexity * 2, 1.0)  # Normalize to 0-1

    async def cluster_threats(self, threat_features: List[np.ndarray]) -> List[int]:
        """Cluster similar threats using DBSCAN"""
        if not threat_features:
            return []

        try:
            features_array = np.array(threat_features)
            features_scaled = self.scaler.fit_transform(features_array)

            clusters = self.clustering_model.fit_predict(features_scaled)
            return clusters.tolist()
        except Exception as e:
            logger.error(f"Error clustering threats: {e}")
            return []

    async def detect_anomalies(self, threat_features: List[np.ndarray]) -> List[bool]:
        """Detect anomalous threat patterns"""
        if not threat_features:
            return []

        try:
            features_array = np.array(threat_features)
            features_scaled = self.scaler.fit_transform(features_array)

            anomalies = self.anomaly_detector.fit_predict(features_scaled)
            return [anomaly == -1 for anomaly in anomalies]  # -1 indicates anomaly
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []

class AdversaryInfrastructureMapper:
    """Map and analyze adversary infrastructure using graph analysis"""

    def __init__(self):
        self.infrastructure_graph = nx.DiGraph()
        self.redis_client = None
        self.db_pool = None

    async def initialize(self, redis_url: str, db_url: str):
        """Initialize connections"""
        try:
            self.redis_client = redis.from_url(redis_url)
            self.db_pool = await asyncpg.create_pool(db_url)
            logger.info("Infrastructure mapper initialized")
        except Exception as e:
            logger.error(f"Failed to initialize infrastructure mapper: {e}")

    async def add_infrastructure_node(self, node_id: str, node_type: str, metadata: Dict):
        """Add infrastructure node to graph"""
        try:
            # Add to graph
            self.infrastructure_graph.add_node(
                node_id,
                type=node_type,
                metadata=metadata,
                first_seen=datetime.now(),
                last_seen=datetime.now()
            )

            # Store in database
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO infrastructure_nodes (id, type, metadata, first_seen, last_seen)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (id) DO UPDATE SET
                            last_seen = $5,
                            metadata = $3
                    """, node_id, node_type, json.dumps(metadata), datetime.now(), datetime.now())

            logger.debug(f"Added infrastructure node: {node_id}")
        except Exception as e:
            logger.error(f"Error adding infrastructure node: {e}")

    async def add_infrastructure_relationship(self, source: str, target: str, relationship_type: str, confidence: float):
        """Add relationship between infrastructure nodes"""
        try:
            # Add edge to graph
            self.infrastructure_graph.add_edge(
                source,
                target,
                type=relationship_type,
                confidence=confidence,
                first_seen=datetime.now(),
                last_seen=datetime.now()
            )

            # Store in database
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO infrastructure_relationships (source, target, type, confidence, first_seen, last_seen)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (source, target, type) DO UPDATE SET
                            confidence = $4,
                            last_seen = $6
                    """, source, target, relationship_type, confidence, datetime.now(), datetime.now())

            logger.debug(f"Added relationship: {source} -> {target} ({relationship_type})")
        except Exception as e:
            logger.error(f"Error adding infrastructure relationship: {e}")

    async def analyze_infrastructure_clusters(self) -> List[Dict]:
        """Identify infrastructure clusters using community detection"""
        try:
            # Convert to undirected for community detection
            undirected_graph = self.infrastructure_graph.to_undirected()

            # Find communities using Louvain algorithm
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(undirected_graph)

            clusters = []
            for i, community in enumerate(communities):
                cluster_info = {
                    'id': f"cluster_{i}",
                    'nodes': list(community),
                    'size': len(community),
                    'centrality': self._calculate_cluster_centrality(community),
                    'density': nx.density(undirected_graph.subgraph(community))
                }
                clusters.append(cluster_info)

            return sorted(clusters, key=lambda x: x['size'], reverse=True)
        except Exception as e:
            logger.error(f"Error analyzing infrastructure clusters: {e}")
            return []

    def _calculate_cluster_centrality(self, community: Set[str]) -> Dict:
        """Calculate centrality metrics for cluster"""
        subgraph = self.infrastructure_graph.subgraph(community)

        try:
            betweenness = nx.betweenness_centrality(subgraph)
            closeness = nx.closeness_centrality(subgraph)
            degree = nx.degree_centrality(subgraph)

            return {
                'betweenness': dict(betweenness),
                'closeness': dict(closeness),
                'degree': dict(degree)
            }
        except Exception as e:
            logger.error(f"Error calculating centrality: {e}")
            return {}

    async def get_infrastructure_paths(self, source: str, target: str) -> List[List[str]]:
        """Find paths between infrastructure nodes"""
        try:
            if source not in self.infrastructure_graph or target not in self.infrastructure_graph:
                return []

            # Find all simple paths
            paths = list(nx.all_simple_paths(
                self.infrastructure_graph,
                source,
                target,
                cutoff=5  # Limit path length
            ))

            return paths[:10]  # Return top 10 paths
        except Exception as e:
            logger.error(f"Error finding infrastructure paths: {e}")
            return []

class CampaignTracker:
    """Track and analyze threat campaigns using ML"""

    def __init__(self, ml_analyzer: ThreatIntelligenceML):
        self.ml_analyzer = ml_analyzer
        self.campaigns: Dict[str, ThreatCampaign] = {}
        self.campaign_features: Dict[str, np.ndarray] = {}
        self.db_pool = None

    async def initialize(self, db_url: str):
        """Initialize database connection"""
        try:
            self.db_pool = await asyncpg.create_pool(db_url)
            await self._load_existing_campaigns()
            logger.info("Campaign tracker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize campaign tracker: {e}")

    async def _load_existing_campaigns(self):
        """Load existing campaigns from database"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM threat_campaigns WHERE status = 'active'")

                for row in rows:
                    campaign = ThreatCampaign(
                        id=row['id'],
                        name=row['name'],
                        status=CampaignStatus(row['status']),
                        threat_actors=json.loads(row['threat_actors']),
                        start_date=row['start_date'],
                        end_date=row['end_date'],
                        objectives=json.loads(row['objectives']),
                        techniques=set(json.loads(row['techniques'])),
                        indicators=set(json.loads(row['indicators'])),
                        victims=json.loads(row['victims']),
                        timeline=json.loads(row['timeline']),
                        confidence=row['confidence']
                    )
                    self.campaigns[campaign.id] = campaign

            logger.info(f"Loaded {len(self.campaigns)} active campaigns")
        except Exception as e:
            logger.error(f"Error loading campaigns: {e}")

    async def analyze_campaign_attribution(self, intelligence_reports: List[IntelligenceReport]) -> Dict:
        """Analyze campaign attribution using ML clustering"""
        try:
            # Extract features from reports
            features = []
            for report in intelligence_reports:
                text_features = await self.ml_analyzer.analyze_threat_text(report.content)
                if text_features and 'embedding' in text_features:
                    features.append(np.array(text_features['embedding']))

            if not features:
                return {'error': 'No features extracted from reports'}

            # Cluster reports
            clusters = await self.ml_analyzer.cluster_threats(features)

            # Analyze clusters for campaign patterns
            campaign_analysis = self._analyze_campaign_clusters(intelligence_reports, clusters)

            return campaign_analysis
        except Exception as e:
            logger.error(f"Error analyzing campaign attribution: {e}")
            return {'error': str(e)}

    def _analyze_campaign_clusters(self, reports: List[IntelligenceReport], clusters: List[int]) -> Dict:
        """Analyze clusters to identify potential campaigns"""
        cluster_groups = defaultdict(list)

        # Group reports by cluster
        for report, cluster_id in zip(reports, clusters):
            if cluster_id != -1:  # -1 indicates noise in DBSCAN
                cluster_groups[cluster_id].append(report)

        campaign_candidates = []
        for cluster_id, cluster_reports in cluster_groups.items():
            if len(cluster_reports) >= 3:  # Minimum reports for campaign
                candidate = self._create_campaign_candidate(cluster_id, cluster_reports)
                campaign_candidates.append(candidate)

        return {
            'total_clusters': len(cluster_groups),
            'campaign_candidates': campaign_candidates,
            'noise_reports': sum(1 for c in clusters if c == -1)
        }

    def _create_campaign_candidate(self, cluster_id: int, reports: List[IntelligenceReport]) -> Dict:
        """Create campaign candidate from clustered reports"""
        # Extract common indicators
        all_indicators = set()
        all_techniques = set()
        all_actors = set()

        for report in reports:
            all_indicators.update(report.indicators)
            all_techniques.update(report.techniques)
            all_actors.update(report.threat_actors)

        # Calculate temporal patterns
        timestamps = [report.timestamp for report in reports]
        time_span = max(timestamps) - min(timestamps)

        # Calculate confidence based on overlap
        confidence = self._calculate_campaign_confidence(reports)

        return {
            'cluster_id': cluster_id,
            'report_count': len(reports),
            'common_indicators': list(all_indicators),
            'common_techniques': list(all_techniques),
            'potential_actors': list(all_actors),
            'time_span_days': time_span.days,
            'confidence': confidence,
            'first_seen': min(timestamps),
            'last_seen': max(timestamps)
        }

    def _calculate_campaign_confidence(self, reports: List[IntelligenceReport]) -> float:
        """Calculate confidence score for campaign attribution"""
        if len(reports) < 2:
            return 0.0

        # Factor 1: Indicator overlap
        indicator_overlap = self._calculate_overlap([set(r.indicators) for r in reports])

        # Factor 2: Technique overlap
        technique_overlap = self._calculate_overlap([set(r.techniques) for r in reports])

        # Factor 3: Source diversity
        sources = set(report.source for report in reports)
        source_diversity = len(sources) / len(reports)

        # Factor 4: Temporal clustering
        timestamps = [report.timestamp for report in reports]
        temporal_score = self._calculate_temporal_clustering(timestamps)

        # Weighted average
        confidence = (
            indicator_overlap * 0.3 +
            technique_overlap * 0.3 +
            source_diversity * 0.2 +
            temporal_score * 0.2
        )

        return min(confidence, 1.0)

    def _calculate_overlap(self, sets: List[Set]) -> float:
        """Calculate overlap score between sets"""
        if len(sets) < 2:
            return 0.0

        intersection = sets[0]
        union = sets[0]

        for s in sets[1:]:
            intersection = intersection.intersection(s)
            union = union.union(s)

        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)

    def _calculate_temporal_clustering(self, timestamps: List[datetime]) -> float:
        """Calculate temporal clustering score"""
        if len(timestamps) < 2:
            return 0.0

        # Sort timestamps
        sorted_times = sorted(timestamps)

        # Calculate intervals
        intervals = []
        for i in range(1, len(sorted_times)):
            interval = (sorted_times[i] - sorted_times[i-1]).total_seconds()
            intervals.append(interval)

        # Calculate coefficient of variation (lower = more clustered)
        if len(intervals) == 0:
            return 0.0

        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        if mean_interval == 0:
            return 1.0

        cv = std_interval / mean_interval
        # Convert to clustering score (0-1, higher = more clustered)
        clustering_score = 1.0 / (1.0 + cv)

        return clustering_score

class IOCEnricher:
    """Enrich Indicators of Compromise with threat intelligence"""

    def __init__(self):
        self.enrichment_cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.redis_client = None
        self.db_pool = None

    async def initialize(self, redis_url: str, db_url: str):
        """Initialize connections"""
        try:
            self.redis_client = redis.from_url(redis_url)
            self.db_pool = await asyncpg.create_pool(db_url)
            logger.info("IOC enricher initialized")
        except Exception as e:
            logger.error(f"Failed to initialize IOC enricher: {e}")

    async def enrich_ioc(self, ioc: str, ioc_type: str) -> IOCEnrichment:
        """Enrich IoC with threat intelligence"""
        try:
            # Check cache first
            cache_key = f"ioc:{hashlib.sha256(ioc.encode()).hexdigest()}"
            cached = await self._get_cached_enrichment(cache_key)
            if cached:
                return cached

            # Perform enrichment
            enrichment = IOCEnrichment(
                ioc=ioc,
                ioc_type=ioc_type,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                reputation_score=0.0,
                threat_associations=[],
                geolocation=None,
                whois_data=None
            )

            # Enrich based on type
            if ioc_type == 'ip':
                await self._enrich_ip(enrichment)
            elif ioc_type == 'domain':
                await self._enrich_domain(enrichment)
            elif ioc_type == 'hash':
                await self._enrich_hash(enrichment)
            elif ioc_type == 'url':
                await self._enrich_url(enrichment)

            # Cache enrichment
            await self._cache_enrichment(cache_key, enrichment)

            # Store in database
            await self._store_enrichment(enrichment)

            IOC_ENRICHMENTS.labels(type=ioc_type).inc()
            return enrichment

        except Exception as e:
            logger.error(f"Error enriching IoC {ioc}: {e}")
            return IOCEnrichment(
                ioc=ioc,
                ioc_type=ioc_type,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                reputation_score=0.0,
                threat_associations=[],
                geolocation=None,
                whois_data=None
            )

    async def _enrich_ip(self, enrichment: IOCEnrichment):
        """Enrich IP address"""
        try:
            # Geolocation lookup
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://ip-api.com/json/{enrichment.ioc}') as resp:
                    if resp.status == 200:
                        geo_data = await resp.json()
                        enrichment.geolocation = geo_data

            # Calculate reputation score
            enrichment.reputation_score = await self._calculate_ip_reputation(enrichment.ioc)

            # Check threat feeds
            enrichment.threat_associations = await self._check_threat_feeds(enrichment.ioc, 'ip')

        except Exception as e:
            logger.error(f"Error enriching IP {enrichment.ioc}: {e}")

    async def _enrich_domain(self, enrichment: IOCEnrichment):
        """Enrich domain name"""
        try:
            # WHOIS lookup
            enrichment.whois_data = await self._whois_lookup(enrichment.ioc)

            # Calculate reputation score
            enrichment.reputation_score = await self._calculate_domain_reputation(enrichment.ioc)

            # Check threat feeds
            enrichment.threat_associations = await self._check_threat_feeds(enrichment.ioc, 'domain')

        except Exception as e:
            logger.error(f"Error enriching domain {enrichment.ioc}: {e}")

    async def _enrich_hash(self, enrichment: IOCEnrichment):
        """Enrich file hash"""
        try:
            # Calculate reputation score
            enrichment.reputation_score = await self._calculate_hash_reputation(enrichment.ioc)

            # Check malware databases
            enrichment.malware_families = await self._check_malware_families(enrichment.ioc)

            # Check threat feeds
            enrichment.threat_associations = await self._check_threat_feeds(enrichment.ioc, 'hash')

        except Exception as e:
            logger.error(f"Error enriching hash {enrichment.ioc}: {e}")

    async def _enrich_url(self, enrichment: IOCEnrichment):
        """Enrich URL"""
        try:
            # Calculate reputation score
            enrichment.reputation_score = await self._calculate_url_reputation(enrichment.ioc)

            # Check threat feeds
            enrichment.threat_associations = await self._check_threat_feeds(enrichment.ioc, 'url')

        except Exception as e:
            logger.error(f"Error enriching URL {enrichment.ioc}: {e}")

    async def _calculate_ip_reputation(self, ip: str) -> float:
        """Calculate IP reputation score"""
        # Placeholder implementation
        # In production, integrate with threat feeds like VirusTotal, AbuseIPDB, etc.
        return 0.5

    async def _calculate_domain_reputation(self, domain: str) -> float:
        """Calculate domain reputation score"""
        # Placeholder implementation
        return 0.5

    async def _calculate_hash_reputation(self, hash_value: str) -> float:
        """Calculate hash reputation score"""
        # Placeholder implementation
        return 0.5

    async def _calculate_url_reputation(self, url: str) -> float:
        """Calculate URL reputation score"""
        # Placeholder implementation
        return 0.5

    async def _check_threat_feeds(self, ioc: str, ioc_type: str) -> List[str]:
        """Check IoC against threat feeds"""
        # Placeholder implementation
        # In production, integrate with multiple threat feeds
        return []

    async def _check_malware_families(self, hash_value: str) -> List[str]:
        """Check hash against malware family databases"""
        # Placeholder implementation
        return []

    async def _whois_lookup(self, domain: str) -> Optional[Dict]:
        """Perform WHOIS lookup"""
        # Placeholder implementation
        return None

    async def _get_cached_enrichment(self, cache_key: str) -> Optional[IOCEnrichment]:
        """Get cached enrichment"""
        if not self.redis_client:
            return None

        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return IOCEnrichment(**data)
        except Exception as e:
            logger.error(f"Error getting cached enrichment: {e}")

        return None

    async def _cache_enrichment(self, cache_key: str, enrichment: IOCEnrichment):
        """Cache enrichment data"""
        if not self.redis_client:
            return

        try:
            data = {
                'ioc': enrichment.ioc,
                'ioc_type': enrichment.ioc_type,
                'first_seen': enrichment.first_seen.isoformat(),
                'last_seen': enrichment.last_seen.isoformat(),
                'reputation_score': enrichment.reputation_score,
                'threat_associations': enrichment.threat_associations,
                'geolocation': enrichment.geolocation,
                'whois_data': enrichment.whois_data,
                'malware_families': enrichment.malware_families,
                'campaigns': enrichment.campaigns,
                'confidence': enrichment.confidence
            }

            await self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.error(f"Error caching enrichment: {e}")

    async def _store_enrichment(self, enrichment: IOCEnrichment):
        """Store enrichment in database"""
        if not self.db_pool:
            return

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ioc_enrichments (
                        ioc, ioc_type, first_seen, last_seen, reputation_score,
                        threat_associations, geolocation, whois_data, malware_families,
                        campaigns, confidence
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (ioc) DO UPDATE SET
                        last_seen = $4,
                        reputation_score = $5,
                        threat_associations = $6,
                        confidence = $11
                """,
                enrichment.ioc, enrichment.ioc_type, enrichment.first_seen,
                enrichment.last_seen, enrichment.reputation_score,
                json.dumps(enrichment.threat_associations),
                json.dumps(enrichment.geolocation),
                json.dumps(enrichment.whois_data),
                json.dumps(enrichment.malware_families),
                json.dumps(enrichment.campaigns),
                enrichment.confidence
                )
        except Exception as e:
            logger.error(f"Error storing enrichment: {e}")

class TacticalIntelligencePlatform:
    """Main tactical intelligence platform orchestrator"""

    def __init__(self, security_framework: OperationalSecurityFramework):
        self.security_framework = security_framework
        self.ml_analyzer = ThreatIntelligenceML()
        self.infrastructure_mapper = AdversaryInfrastructureMapper()
        self.campaign_tracker = CampaignTracker(self.ml_analyzer)
        self.ioc_enricher = IOCEnricher()

        self.intelligence_reports: Dict[str, IntelligenceReport] = {}
        self.threat_actors: Dict[str, ThreatActor] = {}

        self.processing_queue = asyncio.Queue(maxsize=1000)
        self.workers = []
        self.num_workers = 5

    async def initialize(self, redis_url: str = "redis://localhost:6379",
                        db_url: str = "postgresql://user:pass@localhost/bev"):
        """Initialize the tactical intelligence platform"""
        try:
            # Initialize components
            await self.infrastructure_mapper.initialize(redis_url, db_url)
            await self.campaign_tracker.initialize(db_url)
            await self.ioc_enricher.initialize(redis_url, db_url)

            # Start processing workers
            for i in range(self.num_workers):
                worker = asyncio.create_task(self._intelligence_worker(i))
                self.workers.append(worker)

            logger.info("Tactical Intelligence Platform initialized")
            print("🎯 Tactical Intelligence Platform Ready")

        except Exception as e:
            logger.error(f"Failed to initialize tactical intelligence platform: {e}")
            raise

    async def _intelligence_worker(self, worker_id: int):
        """Process intelligence reports from queue"""
        while True:
            try:
                report = await self.processing_queue.get()

                with INTEL_PROCESSING_TIME.time():
                    await self._process_intelligence_report(report)

                self.processing_queue.task_done()

            except Exception as e:
                logger.error(f"Intelligence worker {worker_id} error: {e}")
                await asyncio.sleep(1)

    async def submit_intelligence(self, report: IntelligenceReport) -> str:
        """Submit intelligence report for processing"""
        try:
            # Store report
            self.intelligence_reports[report.id] = report

            # Queue for processing
            await self.processing_queue.put(report)

            logger.info(f"Intelligence report {report.id} queued for processing")
            return report.id

        except Exception as e:
            logger.error(f"Error submitting intelligence: {e}")
            raise

    async def _process_intelligence_report(self, report: IntelligenceReport):
        """Process individual intelligence report"""
        try:
            # Analyze content with ML
            text_features = await self.ml_analyzer.analyze_threat_text(report.content)

            # Extract and enrich IoCs
            for indicator in report.indicators:
                ioc_type = self._determine_ioc_type(indicator)
                enrichment = await self.ioc_enricher.enrich_ioc(indicator, ioc_type)

                # Update campaigns with new intelligence
                for campaign_id in enrichment.campaigns:
                    if campaign_id in self.campaign_tracker.campaigns:
                        campaign = self.campaign_tracker.campaigns[campaign_id]
                        campaign.indicators.add(indicator)
                        campaign.last_seen = datetime.now()

            # Update threat actor profiles
            for actor_name in report.threat_actors:
                await self._update_threat_actor_profile(actor_name, report)

            # Add infrastructure nodes
            for indicator in report.indicators:
                if self._is_infrastructure_indicator(indicator):
                    await self.infrastructure_mapper.add_infrastructure_node(
                        indicator,
                        self._get_infrastructure_type(indicator),
                        {'source': report.source, 'confidence': report.confidence}
                    )

            # Record metrics
            severity = self._determine_threat_severity(text_features)
            THREAT_DETECTIONS.labels(source=report.source, severity=severity.value).inc()

            logger.debug(f"Processed intelligence report {report.id}")

        except Exception as e:
            logger.error(f"Error processing intelligence report {report.id}: {e}")

    def _determine_ioc_type(self, indicator: str) -> str:
        """Determine the type of IoC"""
        import re

        # IP address
        if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', indicator):
            return 'ip'

        # Domain
        if re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}$', indicator):
            return 'domain'

        # Hash (MD5, SHA1, SHA256)
        if re.match(r'^[a-fA-F0-9]{32}$', indicator):
            return 'md5'
        elif re.match(r'^[a-fA-F0-9]{40}$', indicator):
            return 'sha1'
        elif re.match(r'^[a-fA-F0-9]{64}$', indicator):
            return 'sha256'

        # URL
        if indicator.startswith(('http://', 'https://')):
            return 'url'

        return 'unknown'

    def _is_infrastructure_indicator(self, indicator: str) -> bool:
        """Check if indicator represents infrastructure"""
        ioc_type = self._determine_ioc_type(indicator)
        return ioc_type in ['ip', 'domain', 'url']

    def _get_infrastructure_type(self, indicator: str) -> str:
        """Get infrastructure type for indicator"""
        ioc_type = self._determine_ioc_type(indicator)
        if ioc_type == 'ip':
            return 'server'
        elif ioc_type == 'domain':
            return 'domain'
        elif ioc_type == 'url':
            return 'web_resource'
        return 'unknown'

    def _determine_threat_severity(self, features: Dict) -> ThreatSeverity:
        """Determine threat severity from features"""
        if not features:
            return ThreatSeverity.LOW

        threat_indicators = features.get('threat_indicators', [])
        complexity = features.get('complexity_score', 0.0)

        # High severity indicators
        high_severity_keywords = ['apt', 'ransomware', 'breach', 'exfiltration']
        if any(keyword in threat_indicators for keyword in high_severity_keywords):
            return ThreatSeverity.CRITICAL

        # Medium severity indicators
        medium_severity_keywords = ['malware', 'phishing', 'backdoor', 'exploit']
        if any(keyword in threat_indicators for keyword in medium_severity_keywords):
            return ThreatSeverity.HIGH if complexity > 0.7 else ThreatSeverity.MEDIUM

        return ThreatSeverity.LOW

    async def _update_threat_actor_profile(self, actor_name: str, report: IntelligenceReport):
        """Update threat actor profile with new intelligence"""
        try:
            if actor_name not in self.threat_actors:
                # Create new threat actor profile
                self.threat_actors[actor_name] = ThreatActor(
                    id=hashlib.sha256(actor_name.encode()).hexdigest()[:16],
                    name=actor_name,
                    aliases=[],
                    motivation="unknown",
                    sophistication="unknown",
                    attribution_confidence=report.confidence,
                    first_seen=report.timestamp,
                    last_seen=report.timestamp
                )

            actor = self.threat_actors[actor_name]

            # Update profile
            actor.last_seen = max(actor.last_seen, report.timestamp)
            actor.tools.extend([tool for tool in report.metadata.get('tools', []) if tool not in actor.tools])
            actor.techniques.extend([tech for tech in report.techniques if tech not in actor.techniques])

            # Update behavioral patterns
            if 'behavioral_patterns' not in actor.behavioral_patterns:
                actor.behavioral_patterns['behavioral_patterns'] = {}

            # Analyze timing patterns
            if 'timing' not in actor.behavioral_patterns:
                actor.behavioral_patterns['timing'] = []
            actor.behavioral_patterns['timing'].append(report.timestamp.isoformat())

            logger.debug(f"Updated threat actor profile: {actor_name}")

        except Exception as e:
            logger.error(f"Error updating threat actor profile {actor_name}: {e}")

    async def get_campaign_analysis(self) -> Dict:
        """Get current campaign analysis"""
        try:
            active_campaigns = [c for c in self.campaign_tracker.campaigns.values()
                              if c.status == CampaignStatus.ACTIVE]

            ACTIVE_CAMPAIGNS.set(len(active_campaigns))

            return {
                'active_campaigns': len(active_campaigns),
                'total_campaigns': len(self.campaign_tracker.campaigns),
                'campaigns': [
                    {
                        'id': c.id,
                        'name': c.name,
                        'status': c.status.value,
                        'confidence': c.confidence,
                        'indicators_count': len(c.indicators),
                        'techniques_count': len(c.techniques),
                        'duration_days': (c.end_date or datetime.now() - c.start_date).days
                    }
                    for c in active_campaigns
                ]
            }
        except Exception as e:
            logger.error(f"Error getting campaign analysis: {e}")
            return {'error': str(e)}

    async def get_infrastructure_analysis(self) -> Dict:
        """Get infrastructure analysis"""
        try:
            clusters = await self.infrastructure_mapper.analyze_infrastructure_clusters()

            return {
                'total_nodes': self.infrastructure_mapper.infrastructure_graph.number_of_nodes(),
                'total_edges': self.infrastructure_mapper.infrastructure_graph.number_of_edges(),
                'clusters': clusters[:10],  # Top 10 clusters
                'graph_density': nx.density(self.infrastructure_mapper.infrastructure_graph)
            }
        except Exception as e:
            logger.error(f"Error getting infrastructure analysis: {e}")
            return {'error': str(e)}

    async def get_threat_actor_profiles(self) -> Dict:
        """Get threat actor profiles"""
        try:
            return {
                'total_actors': len(self.threat_actors),
                'actors': [
                    {
                        'id': actor.id,
                        'name': actor.name,
                        'sophistication': actor.sophistication,
                        'attribution_confidence': actor.attribution_confidence,
                        'tools_count': len(actor.tools),
                        'techniques_count': len(actor.techniques),
                        'first_seen': actor.first_seen.isoformat(),
                        'last_seen': actor.last_seen.isoformat()
                    }
                    for actor in self.threat_actors.values()
                ]
            }
        except Exception as e:
            logger.error(f"Error getting threat actor profiles: {e}")
            return {'error': str(e)}

    async def shutdown(self):
        """Shutdown the platform"""
        try:
            # Cancel workers
            for worker in self.workers:
                worker.cancel()

            # Wait for queue to finish
            await self.processing_queue.join()

            logger.info("Tactical Intelligence Platform shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Example usage and testing
async def main():
    """Example usage of the Tactical Intelligence Platform"""
    try:
        # Initialize security framework
        security = OperationalSecurityFramework()
        await security.initialize_security()

        # Initialize tactical intelligence platform
        platform = TacticalIntelligencePlatform(security)
        await platform.initialize()

        # Example intelligence report
        report = IntelligenceReport(
            id="report_001",
            source="analyst_001",
            intel_type=IntelType.OSINT,
            content="APT29 observed using new malware variant targeting government entities",
            confidence=0.85,
            timestamp=datetime.now(),
            indicators=["192.168.1.100", "malicious.example.com", "a1b2c3d4e5f6"],
            threat_actors=["APT29"],
            techniques=["T1055", "T1071"]
        )

        # Submit for processing
        report_id = await platform.submit_intelligence(report)
        print(f"✅ Intelligence report submitted: {report_id}")

        # Wait for processing
        await asyncio.sleep(2)

        # Get analysis
        campaign_analysis = await platform.get_campaign_analysis()
        infrastructure_analysis = await platform.get_infrastructure_analysis()
        actor_profiles = await platform.get_threat_actor_profiles()

        print(f"📊 Campaign Analysis: {campaign_analysis}")
        print(f"🏗️ Infrastructure Analysis: {infrastructure_analysis}")
        print(f"👤 Threat Actor Profiles: {actor_profiles}")

        # Shutdown
        await platform.shutdown()

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())