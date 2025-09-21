#!/usr/bin/env python3
"""
Cryptocurrency Intelligence Engine - Phase 7 Alternative Market Intelligence Platform
Place in: /home/starlord/Projects/Bev/src/alternative_market/crypto_analyzer.py

Multi-chain transaction monitoring with wallet clustering and mixing service detection.
"""

import asyncio
import aiohttp
import json
import time
import hashlib
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import asyncpg
import aioredis
from aiokafka import AIOKafkaProducer
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import re
import uuid
from decimal import Decimal, ROUND_HALF_UP
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Transaction:
    """Cryptocurrency transaction structure"""
    tx_id: str
    blockchain: str  # 'bitcoin', 'ethereum', 'monero', 'zcash'
    block_height: int
    timestamp: datetime
    from_addresses: List[str]
    to_addresses: List[str]
    amount: Decimal
    currency: str
    fee: Decimal
    confirmations: int
    transaction_type: str  # 'standard', 'mixing', 'exchange', 'suspicious'
    risk_score: float
    metadata: Dict[str, Any]

@dataclass
class WalletCluster:
    """Wallet clustering analysis result"""
    cluster_id: str
    addresses: Set[str]
    blockchain: str
    cluster_type: str  # 'individual', 'service', 'exchange', 'mixer'
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    first_seen: datetime
    last_activity: datetime
    total_volume: Decimal
    transaction_count: int
    behavioral_patterns: Dict[str, Any]
    connected_clusters: Set[str]

@dataclass
class MixingService:
    """Mixing service detection result"""
    service_id: str
    name: Optional[str]
    blockchain: str
    addresses: Set[str]
    detection_confidence: float
    mixing_patterns: Dict[str, Any]
    volume_mixed: Decimal
    active_period: Tuple[datetime, datetime]
    clients_detected: int

@dataclass
class ExchangeFlow:
    """Exchange transaction flow analysis"""
    exchange_id: str
    exchange_name: str
    blockchain: str
    deposit_addresses: Set[str]
    withdrawal_addresses: Set[str]
    flow_direction: str  # 'deposit', 'withdrawal', 'internal'
    amount: Decimal
    currency: str
    timestamp: datetime
    user_cluster: Optional[str]


class BlockchainAPI:
    """Multi-blockchain API interface"""

    def __init__(self):
        self.api_endpoints = {
            'bitcoin': {
                'blockstream': 'https://blockstream.info/api',
                'blockcypher': 'https://api.blockcypher.com/v1/btc/main',
                'blockchain_info': 'https://blockchain.info'
            },
            'ethereum': {
                'etherscan': 'https://api.etherscan.io/api',
                'infura': 'https://mainnet.infura.io/v3',
                'alchemy': 'https://eth-mainnet.alchemyapi.io/v2'
            },
            'monero': {
                'xmrchain': 'https://xmrchain.net/api',
                'monerod': 'http://localhost:18081'  # Local node
            },
            'zcash': {
                'zcashd': 'http://localhost:8232',  # Local node
                'explorer': 'https://explorer.zcha.in/api'
            }
        }
        self.api_keys = {}
        self.rate_limits = {
            'bitcoin': 10,  # requests per second
            'ethereum': 5,
            'monero': 2,
            'zcash': 2
        }
        self.session = None

    async def initialize(self):
        """Initialize HTTP session and load API keys"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )

        # Load API keys from environment
        import os
        self.api_keys = {
            'etherscan': os.getenv('ETHERSCAN_API_KEY', ''),
            'blockcypher': os.getenv('BLOCKCYPHER_API_KEY', ''),
            'infura': os.getenv('INFURA_PROJECT_ID', ''),
            'alchemy': os.getenv('ALCHEMY_API_KEY', '')
        }

    async def get_transaction(self, tx_id: str, blockchain: str) -> Optional[Transaction]:
        """Get transaction details from blockchain"""

        try:
            if blockchain == 'bitcoin':
                return await self._get_bitcoin_transaction(tx_id)
            elif blockchain == 'ethereum':
                return await self._get_ethereum_transaction(tx_id)
            elif blockchain == 'monero':
                return await self._get_monero_transaction(tx_id)
            elif blockchain == 'zcash':
                return await self._get_zcash_transaction(tx_id)
            else:
                logger.error(f"Unsupported blockchain: {blockchain}")
                return None

        except Exception as e:
            logger.error(f"Error fetching transaction {tx_id} on {blockchain}: {e}")
            return None

    async def _get_bitcoin_transaction(self, tx_id: str) -> Optional[Transaction]:
        """Get Bitcoin transaction via Blockstream API"""

        url = f"{self.api_endpoints['bitcoin']['blockstream']}/tx/{tx_id}"

        async with self.session.get(url) as response:
            if response.status != 200:
                return None

            data = await response.json()

            # Parse inputs and outputs
            from_addresses = []
            to_addresses = []
            total_amount = Decimal('0')

            for vin in data.get('vin', []):
                if 'prevout' in vin:
                    from_addresses.append(vin['prevout'].get('scriptpubkey_address', ''))

            for vout in data.get('vout', []):
                to_addresses.append(vout.get('scriptpubkey_address', ''))
                total_amount += Decimal(str(vout.get('value', 0))) / Decimal('100000000')  # Convert from satoshis

            return Transaction(
                tx_id=tx_id,
                blockchain='bitcoin',
                block_height=data.get('status', {}).get('block_height', 0),
                timestamp=datetime.fromtimestamp(data.get('status', {}).get('block_time', 0)),
                from_addresses=from_addresses,
                to_addresses=to_addresses,
                amount=total_amount,
                currency='BTC',
                fee=Decimal(str(data.get('fee', 0))) / Decimal('100000000'),
                confirmations=data.get('status', {}).get('confirmations', 0),
                transaction_type='standard',
                risk_score=0.0,
                metadata=data
            )

    async def _get_ethereum_transaction(self, tx_id: str) -> Optional[Transaction]:
        """Get Ethereum transaction via Etherscan API"""

        if not self.api_keys.get('etherscan'):
            logger.warning("Etherscan API key not configured")
            return None

        url = f"{self.api_endpoints['ethereum']['etherscan']}"
        params = {
            'module': 'proxy',
            'action': 'eth_getTransactionByHash',
            'txhash': tx_id,
            'apikey': self.api_keys['etherscan']
        }

        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                return None

            data = await response.json()
            tx_data = data.get('result')

            if not tx_data:
                return None

            # Convert hex values
            amount = Decimal(int(tx_data.get('value', '0x0'), 16)) / Decimal('10') ** 18
            gas_price = Decimal(int(tx_data.get('gasPrice', '0x0'), 16))
            gas_used = Decimal(int(tx_data.get('gas', '0x0'), 16))
            fee = (gas_price * gas_used) / Decimal('10') ** 18

            return Transaction(
                tx_id=tx_id,
                blockchain='ethereum',
                block_height=int(tx_data.get('blockNumber', '0x0'), 16),
                timestamp=datetime.fromtimestamp(int(tx_data.get('timeStamp', '0'), 16)),
                from_addresses=[tx_data.get('from', '')],
                to_addresses=[tx_data.get('to', '')],
                amount=amount,
                currency='ETH',
                fee=fee,
                confirmations=1,  # Would need current block height to calculate
                transaction_type='standard',
                risk_score=0.0,
                metadata=tx_data
            )

    async def _get_monero_transaction(self, tx_id: str) -> Optional[Transaction]:
        """Get Monero transaction (limited due to privacy features)"""

        # Monero transactions are private by default
        # Limited information available through public APIs
        url = f"{self.api_endpoints['monero']['xmrchain']}/transaction/{tx_id}"

        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None

                data = await response.json()

                return Transaction(
                    tx_id=tx_id,
                    blockchain='monero',
                    block_height=data.get('block_height', 0),
                    timestamp=datetime.fromtimestamp(data.get('timestamp', 0)),
                    from_addresses=['private'],  # Monero addresses are private
                    to_addresses=['private'],
                    amount=Decimal('0'),  # Amount is private
                    currency='XMR',
                    fee=Decimal(str(data.get('fee', 0))) / Decimal('10') ** 12,
                    confirmations=data.get('confirmations', 0),
                    transaction_type='private',
                    risk_score=0.0,
                    metadata=data
                )

        except Exception:
            return None

    async def _get_zcash_transaction(self, tx_id: str) -> Optional[Transaction]:
        """Get Zcash transaction"""

        url = f"{self.api_endpoints['zcash']['explorer']}/transactions/{tx_id}"

        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None

                data = await response.json()

                # Parse transaction data
                from_addresses = []
                to_addresses = []
                total_amount = Decimal('0')

                for vin in data.get('vin', []):
                    from_addresses.append(vin.get('addr', 'shielded'))

                for vout in data.get('vout', []):
                    to_addresses.append(vout.get('scriptPubKey', {}).get('addresses', ['shielded'])[0])
                    total_amount += Decimal(str(vout.get('value', 0)))

                return Transaction(
                    tx_id=tx_id,
                    blockchain='zcash',
                    block_height=data.get('height', 0),
                    timestamp=datetime.fromtimestamp(data.get('time', 0)),
                    from_addresses=from_addresses,
                    to_addresses=to_addresses,
                    amount=total_amount,
                    currency='ZEC',
                    fee=Decimal(str(data.get('fee', 0))),
                    confirmations=data.get('confirmations', 0),
                    transaction_type='standard',
                    risk_score=0.0,
                    metadata=data
                )

        except Exception:
            return None

    async def cleanup(self):
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()


class WalletClusteringEngine:
    """Advanced wallet clustering using multiple heuristics"""

    def __init__(self):
        self.clustering_algorithms = ['common_input', 'change_address', 'behavioral']
        self.min_cluster_size = 2
        self.max_cluster_size = 10000
        self.similarity_threshold = 0.7

    async def cluster_addresses(self, transactions: List[Transaction]) -> List[WalletCluster]:
        """Perform wallet clustering analysis"""

        logger.info(f"Clustering {len(transactions)} transactions...")

        # Build transaction graph
        address_graph = self._build_address_graph(transactions)

        # Apply clustering heuristics
        clusters = []

        # Common input heuristic
        common_input_clusters = self._common_input_clustering(transactions)
        clusters.extend(common_input_clusters)

        # Change address heuristic
        change_clusters = self._change_address_clustering(transactions)
        clusters.extend(change_clusters)

        # Behavioral clustering
        behavioral_clusters = await self._behavioral_clustering(transactions)
        clusters.extend(behavioral_clusters)

        # Merge overlapping clusters
        merged_clusters = self._merge_clusters(clusters)

        # Analyze cluster characteristics
        analyzed_clusters = []
        for cluster in merged_clusters:
            analyzed_cluster = await self._analyze_cluster(cluster, transactions)
            analyzed_clusters.append(analyzed_cluster)

        logger.info(f"Identified {len(analyzed_clusters)} wallet clusters")
        return analyzed_clusters

    def _build_address_graph(self, transactions: List[Transaction]) -> nx.Graph:
        """Build graph of address relationships"""

        graph = nx.Graph()

        for tx in transactions:
            # Add nodes for all addresses
            for addr in tx.from_addresses + tx.to_addresses:
                if addr and addr != 'private' and addr != 'shielded':
                    graph.add_node(addr)

            # Add edges between input addresses (common input heuristic)
            for i, addr1 in enumerate(tx.from_addresses):
                for addr2 in tx.from_addresses[i+1:]:
                    if addr1 and addr2 and addr1 != addr2:
                        if graph.has_edge(addr1, addr2):
                            graph[addr1][addr2]['weight'] += 1
                        else:
                            graph.add_edge(addr1, addr2, weight=1, relation='common_input')

        return graph

    def _common_input_clustering(self, transactions: List[Transaction]) -> List[WalletCluster]:
        """Cluster addresses using common input heuristic"""

        clusters = []
        processed_addresses = set()

        for tx in transactions:
            if len(tx.from_addresses) > 1:
                # All input addresses likely belong to same entity
                cluster_addresses = set(addr for addr in tx.from_addresses
                                      if addr and addr not in processed_addresses)

                if len(cluster_addresses) >= self.min_cluster_size:
                    cluster = WalletCluster(
                        cluster_id=str(uuid.uuid4()),
                        addresses=cluster_addresses,
                        blockchain=tx.blockchain,
                        cluster_type='individual',
                        risk_level='low',
                        first_seen=tx.timestamp,
                        last_activity=tx.timestamp,
                        total_volume=tx.amount,
                        transaction_count=1,
                        behavioral_patterns={'common_input': True},
                        connected_clusters=set()
                    )

                    clusters.append(cluster)
                    processed_addresses.update(cluster_addresses)

        return clusters

    def _change_address_clustering(self, transactions: List[Transaction]) -> List[WalletCluster]:
        """Cluster addresses using change address heuristic"""

        clusters = []
        change_candidates = defaultdict(list)

        # Identify potential change addresses
        for tx in transactions:
            if len(tx.to_addresses) == 2 and len(tx.from_addresses) == 1:
                # Likely has one recipient and one change address
                amounts = self._get_output_amounts(tx)
                if amounts and len(amounts) == 2:
                    # Smaller amount is likely change
                    min_amount = min(amounts)
                    change_addr = tx.to_addresses[amounts.index(min_amount)]
                    sender = tx.from_addresses[0]

                    change_candidates[sender].append(change_addr)

        # Create clusters for change addresses
        for sender, change_addrs in change_candidates.items():
            if len(change_addrs) >= 2:
                cluster_addresses = {sender} | set(change_addrs)

                cluster = WalletCluster(
                    cluster_id=str(uuid.uuid4()),
                    addresses=cluster_addresses,
                    blockchain='bitcoin',  # Heuristic mainly for Bitcoin
                    cluster_type='individual',
                    risk_level='low',
                    first_seen=datetime.now(),
                    last_activity=datetime.now(),
                    total_volume=Decimal('0'),
                    transaction_count=len(change_addrs),
                    behavioral_patterns={'change_address': True},
                    connected_clusters=set()
                )

                clusters.append(cluster)

        return clusters

    async def _behavioral_clustering(self, transactions: List[Transaction]) -> List[WalletCluster]:
        """Cluster addresses using behavioral patterns"""

        # Group transactions by address
        address_behaviors = defaultdict(list)

        for tx in transactions:
            for addr in tx.from_addresses + tx.to_addresses:
                if addr and addr != 'private' and addr != 'shielded':
                    behavior = {
                        'amount': float(tx.amount),
                        'hour': tx.timestamp.hour,
                        'weekday': tx.timestamp.weekday(),
                        'fee_rate': float(tx.fee / tx.amount) if tx.amount > 0 else 0
                    }
                    address_behaviors[addr].append(behavior)

        # Extract behavioral features
        features = []
        addresses = []

        for addr, behaviors in address_behaviors.items():
            if len(behaviors) >= 3:  # Minimum transactions for behavioral analysis
                df = pd.DataFrame(behaviors)

                feature_vector = [
                    df['amount'].mean(),
                    df['amount'].std(),
                    df['hour'].mean(),
                    df['hour'].std(),
                    df['weekday'].mode().iloc[0] if not df['weekday'].mode().empty else 0,
                    df['fee_rate'].mean()
                ]

                features.append(feature_vector)
                addresses.append(addr)

        if len(features) < 2:
            return []

        # Perform clustering
        features_array = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)

        clustering = DBSCAN(eps=0.5, min_samples=2)
        cluster_labels = clustering.fit_predict(features_scaled)

        # Create clusters
        clusters = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise points
                continue

            cluster_addresses = {addresses[i] for i, label in enumerate(cluster_labels)
                               if label == cluster_id}

            if len(cluster_addresses) >= self.min_cluster_size:
                cluster = WalletCluster(
                    cluster_id=str(uuid.uuid4()),
                    addresses=cluster_addresses,
                    blockchain='multi',
                    cluster_type='behavioral',
                    risk_level='medium',
                    first_seen=datetime.now(),
                    last_activity=datetime.now(),
                    total_volume=Decimal('0'),
                    transaction_count=0,
                    behavioral_patterns={'behavioral_clustering': True},
                    connected_clusters=set()
                )

                clusters.append(cluster)

        return clusters

    def _merge_clusters(self, clusters: List[WalletCluster]) -> List[WalletCluster]:
        """Merge overlapping clusters"""

        merged = []
        processed = set()

        for i, cluster1 in enumerate(clusters):
            if i in processed:
                continue

            merged_addresses = cluster1.addresses.copy()
            merged_cluster_ids = {cluster1.cluster_id}

            for j, cluster2 in enumerate(clusters[i+1:], i+1):
                if j in processed:
                    continue

                # Check for overlap
                overlap = cluster1.addresses & cluster2.addresses
                if len(overlap) > 0:
                    merged_addresses |= cluster2.addresses
                    merged_cluster_ids.add(cluster2.cluster_id)
                    processed.add(j)

            # Create merged cluster
            merged_cluster = WalletCluster(
                cluster_id=str(uuid.uuid4()),
                addresses=merged_addresses,
                blockchain=cluster1.blockchain,
                cluster_type=cluster1.cluster_type,
                risk_level=cluster1.risk_level,
                first_seen=cluster1.first_seen,
                last_activity=cluster1.last_activity,
                total_volume=cluster1.total_volume,
                transaction_count=cluster1.transaction_count,
                behavioral_patterns=cluster1.behavioral_patterns,
                connected_clusters=merged_cluster_ids - {cluster1.cluster_id}
            )

            merged.append(merged_cluster)
            processed.add(i)

        return merged

    async def _analyze_cluster(self, cluster: WalletCluster,
                             transactions: List[Transaction]) -> WalletCluster:
        """Analyze cluster characteristics and risk"""

        # Get transactions involving cluster addresses
        cluster_transactions = [
            tx for tx in transactions
            if any(addr in cluster.addresses for addr in tx.from_addresses + tx.to_addresses)
        ]

        if not cluster_transactions:
            return cluster

        # Calculate statistics
        cluster.transaction_count = len(cluster_transactions)
        cluster.total_volume = sum(tx.amount for tx in cluster_transactions)
        cluster.first_seen = min(tx.timestamp for tx in cluster_transactions)
        cluster.last_activity = max(tx.timestamp for tx in cluster_transactions)

        # Analyze patterns
        time_patterns = self._analyze_time_patterns(cluster_transactions)
        amount_patterns = self._analyze_amount_patterns(cluster_transactions)
        geographic_patterns = await self._analyze_geographic_patterns(cluster.addresses)

        cluster.behavioral_patterns.update({
            'time_patterns': time_patterns,
            'amount_patterns': amount_patterns,
            'geographic_patterns': geographic_patterns
        })

        # Assess risk level
        cluster.risk_level = self._assess_cluster_risk(cluster, cluster_transactions)

        return cluster

    def _analyze_time_patterns(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Analyze temporal patterns in transactions"""

        if not transactions:
            return {}

        timestamps = [tx.timestamp for tx in transactions]
        hours = [ts.hour for ts in timestamps]
        weekdays = [ts.weekday() for ts in timestamps]

        return {
            'most_active_hour': max(set(hours), key=hours.count),
            'most_active_weekday': max(set(weekdays), key=weekdays.count),
            'time_span_days': (max(timestamps) - min(timestamps)).days,
            'average_interval_hours': np.mean([
                (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
                for i in range(1, len(timestamps))
            ]) if len(timestamps) > 1 else 0
        }

    def _analyze_amount_patterns(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Analyze transaction amount patterns"""

        if not transactions:
            return {}

        amounts = [float(tx.amount) for tx in transactions]

        return {
            'total_volume': sum(amounts),
            'average_amount': np.mean(amounts),
            'median_amount': np.median(amounts),
            'amount_std': np.std(amounts),
            'max_amount': max(amounts),
            'min_amount': min(amounts),
            'round_amounts_ratio': sum(1 for amt in amounts if amt == round(amt)) / len(amounts)
        }

    async def _analyze_geographic_patterns(self, addresses: Set[str]) -> Dict[str, Any]:
        """Analyze geographic patterns (placeholder)"""

        # In production, this would use IP geolocation or other techniques
        return {
            'estimated_regions': ['unknown'],
            'confidence': 0.0
        }

    def _assess_cluster_risk(self, cluster: WalletCluster,
                           transactions: List[Transaction]) -> str:
        """Assess risk level of cluster"""

        risk_factors = 0

        # High transaction volume
        if cluster.total_volume > 1000:
            risk_factors += 1

        # High transaction frequency
        if cluster.transaction_count > 100:
            risk_factors += 1

        # Round number transactions (suspicious)
        round_ratio = cluster.behavioral_patterns.get('amount_patterns', {}).get('round_amounts_ratio', 0)
        if round_ratio > 0.5:
            risk_factors += 1

        # Active during unusual hours
        active_hour = cluster.behavioral_patterns.get('time_patterns', {}).get('most_active_hour', 12)
        if active_hour < 6 or active_hour > 22:
            risk_factors += 1

        # Determine risk level
        if risk_factors >= 3:
            return 'critical'
        elif risk_factors >= 2:
            return 'high'
        elif risk_factors >= 1:
            return 'medium'
        else:
            return 'low'

    def _get_output_amounts(self, transaction: Transaction) -> List[float]:
        """Extract output amounts from transaction metadata"""

        try:
            if 'vout' in transaction.metadata:
                return [vout.get('value', 0) for vout in transaction.metadata['vout']]
            elif transaction.amount:
                return [float(transaction.amount)]
            else:
                return []
        except Exception:
            return []


class MixingServiceDetector:
    """Detect cryptocurrency mixing services"""

    def __init__(self):
        self.known_mixers = {
            'bitcoin': {
                'tornado_cash_btc': {
                    'patterns': ['equal_amounts', 'timing_delays'],
                    'addresses': set()
                },
                'classic_mixer': {
                    'patterns': ['multiple_inputs', 'multiple_outputs'],
                    'addresses': set()
                }
            },
            'ethereum': {
                'tornado_cash': {
                    'patterns': ['fixed_amounts', 'zero_knowledge'],
                    'addresses': {
                        '0x12D66f87A04A9E220743712cE6d9bB1B5616B8Fc',  # Tornado Cash 0.1 ETH
                        '0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936',  # Tornado Cash 1 ETH
                        '0x910Cbd523D972eb0a6f4cAe4618aD62622b39DbF'   # Tornado Cash 10 ETH
                    }
                }
            }
        }
        self.mixing_patterns = {
            'equal_amounts': self._detect_equal_amounts,
            'timing_delays': self._detect_timing_delays,
            'multiple_inputs': self._detect_multiple_inputs,
            'fixed_amounts': self._detect_fixed_amounts
        }

    async def detect_mixing_services(self, transactions: List[Transaction]) -> List[MixingService]:
        """Detect mixing services in transaction data"""

        logger.info(f"Analyzing {len(transactions)} transactions for mixing services...")

        detected_services = []

        # Group transactions by blockchain
        blockchain_txs = defaultdict(list)
        for tx in transactions:
            blockchain_txs[tx.blockchain].append(tx)

        # Analyze each blockchain
        for blockchain, txs in blockchain_txs.items():
            # Known mixer detection
            known_services = await self._detect_known_mixers(txs, blockchain)
            detected_services.extend(known_services)

            # Pattern-based detection
            pattern_services = await self._detect_pattern_based_mixing(txs, blockchain)
            detected_services.extend(pattern_services)

        logger.info(f"Detected {len(detected_services)} mixing services")
        return detected_services

    async def _detect_known_mixers(self, transactions: List[Transaction],
                                 blockchain: str) -> List[MixingService]:
        """Detect known mixing services"""

        detected = []

        if blockchain not in self.known_mixers:
            return detected

        for service_name, config in self.known_mixers[blockchain].items():
            service_transactions = []

            for tx in transactions:
                # Check if transaction involves known mixer addresses
                all_addresses = set(tx.from_addresses + tx.to_addresses)
                if all_addresses & config['addresses']:
                    service_transactions.append(tx)

            if service_transactions:
                total_volume = sum(tx.amount for tx in service_transactions)

                service = MixingService(
                    service_id=f"{blockchain}_{service_name}",
                    name=service_name,
                    blockchain=blockchain,
                    addresses=config['addresses'],
                    detection_confidence=0.95,  # High confidence for known mixers
                    mixing_patterns=config['patterns'],
                    volume_mixed=total_volume,
                    active_period=(
                        min(tx.timestamp for tx in service_transactions),
                        max(tx.timestamp for tx in service_transactions)
                    ),
                    clients_detected=len(set(
                        addr for tx in service_transactions
                        for addr in tx.from_addresses + tx.to_addresses
                        if addr not in config['addresses']
                    ))
                )

                detected.append(service)

        return detected

    async def _detect_pattern_based_mixing(self, transactions: List[Transaction],
                                         blockchain: str) -> List[MixingService]:
        """Detect unknown mixing services using patterns"""

        detected = []

        # Analyze transaction patterns
        for pattern_name, detector_func in self.mixing_patterns.items():
            pattern_results = await detector_func(transactions)

            for result in pattern_results:
                service = MixingService(
                    service_id=f"{blockchain}_pattern_{pattern_name}_{uuid.uuid4()}",
                    name=f"Unknown {pattern_name} mixer",
                    blockchain=blockchain,
                    addresses=result['addresses'],
                    detection_confidence=result['confidence'],
                    mixing_patterns=[pattern_name],
                    volume_mixed=result['volume'],
                    active_period=result['period'],
                    clients_detected=result['clients']
                )

                detected.append(service)

        return detected

    async def _detect_equal_amounts(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """Detect mixing based on equal amounts"""

        results = []
        amount_groups = defaultdict(list)

        # Group transactions by amount
        for tx in transactions:
            amount_key = float(tx.amount)
            amount_groups[amount_key].append(tx)

        # Look for suspicious equal amount patterns
        for amount, txs in amount_groups.items():
            if len(txs) >= 5 and amount > 0:  # At least 5 transactions of same amount
                # Check time distribution
                times = [tx.timestamp for tx in txs]
                time_span = (max(times) - min(times)).total_seconds()

                if time_span < 3600:  # Within 1 hour
                    all_addresses = set()
                    for tx in txs:
                        all_addresses.update(tx.from_addresses + tx.to_addresses)

                    results.append({
                        'addresses': all_addresses,
                        'confidence': 0.7,
                        'volume': Decimal(str(amount * len(txs))),
                        'period': (min(times), max(times)),
                        'clients': len(all_addresses)
                    })

        return results

    async def _detect_timing_delays(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """Detect mixing based on timing delays"""

        results = []

        # Look for patterns where funds are received and sent after delays
        address_flows = defaultdict(list)

        for tx in transactions:
            for addr in tx.to_addresses:
                address_flows[addr].append(('receive', tx))
            for addr in tx.from_addresses:
                address_flows[addr].append(('send', tx))

        # Analyze each address for mixing patterns
        for addr, flows in address_flows.items():
            flows.sort(key=lambda x: x[1].timestamp)

            # Look for receive -> delay -> send patterns
            for i in range(len(flows) - 1):
                if flows[i][0] == 'receive' and flows[i+1][0] == 'send':
                    delay = (flows[i+1][1].timestamp - flows[i][1].timestamp).total_seconds()

                    # Suspicious delays (10 minutes to 24 hours)
                    if 600 <= delay <= 86400:
                        results.append({
                            'addresses': {addr},
                            'confidence': 0.6,
                            'volume': flows[i][1].amount,
                            'period': (flows[i][1].timestamp, flows[i+1][1].timestamp),
                            'clients': 1
                        })

        return results

    async def _detect_multiple_inputs(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """Detect mixing based on multiple inputs pattern"""

        results = []

        # Look for transactions with many inputs and outputs
        for tx in transactions:
            if len(tx.from_addresses) >= 10 and len(tx.to_addresses) >= 10:
                # This pattern suggests potential mixing
                all_addresses = set(tx.from_addresses + tx.to_addresses)

                results.append({
                    'addresses': all_addresses,
                    'confidence': 0.8,
                    'volume': tx.amount,
                    'period': (tx.timestamp, tx.timestamp),
                    'clients': len(all_addresses)
                })

        return results

    async def _detect_fixed_amounts(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """Detect mixing based on fixed amounts (like Tornado Cash)"""

        results = []
        fixed_amounts = [0.1, 1.0, 10.0, 100.0]  # Common fixed amounts

        for fixed_amount in fixed_amounts:
            matching_txs = [
                tx for tx in transactions
                if abs(float(tx.amount) - fixed_amount) < 0.001
            ]

            if len(matching_txs) >= 3:
                all_addresses = set()
                for tx in matching_txs:
                    all_addresses.update(tx.from_addresses + tx.to_addresses)

                times = [tx.timestamp for tx in matching_txs]

                results.append({
                    'addresses': all_addresses,
                    'confidence': 0.85,
                    'volume': Decimal(str(fixed_amount * len(matching_txs))),
                    'period': (min(times), max(times)),
                    'clients': len(all_addresses)
                })

        return results


class ExchangeFlowAnalyzer:
    """Analyze cryptocurrency exchange flows"""

    def __init__(self):
        self.known_exchanges = {
            'binance': {
                'bitcoin': {'1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s'},
                'ethereum': {'0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE'}
            },
            'coinbase': {
                'bitcoin': {'bc1q3j3j3j3j3j3j3j3j3j3j3j3j3j3j3j3j3j3j3j3'},
                'ethereum': {'0x503828976D22510aad0201ac7EC88293211D23Da'}
            }
        }

    async def analyze_exchange_flows(self, transactions: List[Transaction]) -> List[ExchangeFlow]:
        """Analyze flows to/from exchanges"""

        flows = []

        for tx in transactions:
            # Check if transaction involves known exchange addresses
            exchange_flows = self._identify_exchange_flows(tx)
            flows.extend(exchange_flows)

        return flows

    def _identify_exchange_flows(self, transaction: Transaction) -> List[ExchangeFlow]:
        """Identify exchange flows in a transaction"""

        flows = []

        for exchange_name, chains in self.known_exchanges.items():
            exchange_addresses = chains.get(transaction.blockchain, set())

            # Check for deposits (to exchange)
            for to_addr in transaction.to_addresses:
                if to_addr in exchange_addresses:
                    flow = ExchangeFlow(
                        exchange_id=f"{exchange_name}_{transaction.blockchain}",
                        exchange_name=exchange_name,
                        blockchain=transaction.blockchain,
                        deposit_addresses={to_addr},
                        withdrawal_addresses=set(),
                        flow_direction='deposit',
                        amount=transaction.amount,
                        currency=transaction.currency,
                        timestamp=transaction.timestamp,
                        user_cluster=None
                    )
                    flows.append(flow)

            # Check for withdrawals (from exchange)
            for from_addr in transaction.from_addresses:
                if from_addr in exchange_addresses:
                    flow = ExchangeFlow(
                        exchange_id=f"{exchange_name}_{transaction.blockchain}",
                        exchange_name=exchange_name,
                        blockchain=transaction.blockchain,
                        deposit_addresses=set(),
                        withdrawal_addresses={from_addr},
                        flow_direction='withdrawal',
                        amount=transaction.amount,
                        currency=transaction.currency,
                        timestamp=transaction.timestamp,
                        user_cluster=None
                    )
                    flows.append(flow)

        return flows


class CryptocurrencyIntelligenceEngine:
    """Main cryptocurrency intelligence engine"""

    def __init__(self, db_config: Dict[str, Any], redis_config: Dict[str, Any],
                 kafka_config: Dict[str, Any]):

        self.db_config = db_config
        self.redis_config = redis_config
        self.kafka_config = kafka_config

        # Core components
        self.blockchain_api = BlockchainAPI()
        self.clustering_engine = WalletClusteringEngine()
        self.mixing_detector = MixingServiceDetector()
        self.exchange_analyzer = ExchangeFlowAnalyzer()

        # Data storage
        self.db_pool = None
        self.redis_client = None
        self.kafka_producer = None

        # Monitoring state
        self.monitored_addresses: Set[str] = set()
        self.alert_thresholds = {
            'large_transaction': Decimal('10'),  # 10 BTC/ETH equivalent
            'rapid_transactions': 10,  # 10 transactions per hour
            'mixing_confidence': 0.8
        }

    async def initialize(self):
        """Initialize all components"""

        logger.info("Initializing Cryptocurrency Intelligence Engine...")

        # Initialize database connection
        self.db_pool = await asyncpg.create_pool(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            min_size=5,
            max_size=20
        )

        # Initialize Redis
        self.redis_client = aioredis.from_url(
            f"redis://{self.redis_config['host']}:{self.redis_config['port']}/1"
        )

        # Initialize Kafka
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_config['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )
        await self.kafka_producer.start()

        # Initialize blockchain API
        await self.blockchain_api.initialize()

        # Create database tables
        await self._initialize_database_tables()

        logger.info("Cryptocurrency Intelligence Engine initialized successfully")

    async def _initialize_database_tables(self):
        """Create database tables for crypto intelligence"""

        tables = [
            '''
            CREATE TABLE IF NOT EXISTS crypto_transactions (
                tx_id VARCHAR(255) PRIMARY KEY,
                blockchain VARCHAR(50),
                block_height BIGINT,
                timestamp TIMESTAMP,
                from_addresses TEXT[],
                to_addresses TEXT[],
                amount DECIMAL(30,8),
                currency VARCHAR(10),
                fee DECIMAL(30,8),
                confirmations INTEGER,
                transaction_type VARCHAR(50),
                risk_score FLOAT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
            ''',
            '''
            CREATE TABLE IF NOT EXISTS wallet_clusters (
                cluster_id VARCHAR(255) PRIMARY KEY,
                addresses TEXT[],
                blockchain VARCHAR(50),
                cluster_type VARCHAR(50),
                risk_level VARCHAR(20),
                first_seen TIMESTAMP,
                last_activity TIMESTAMP,
                total_volume DECIMAL(30,8),
                transaction_count INTEGER,
                behavioral_patterns JSONB,
                connected_clusters TEXT[],
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            ''',
            '''
            CREATE TABLE IF NOT EXISTS mixing_services (
                service_id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(200),
                blockchain VARCHAR(50),
                addresses TEXT[],
                detection_confidence FLOAT,
                mixing_patterns TEXT[],
                volume_mixed DECIMAL(30,8),
                active_start TIMESTAMP,
                active_end TIMESTAMP,
                clients_detected INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            );
            ''',
            '''
            CREATE TABLE IF NOT EXISTS exchange_flows (
                id SERIAL PRIMARY KEY,
                exchange_id VARCHAR(100),
                exchange_name VARCHAR(100),
                blockchain VARCHAR(50),
                flow_direction VARCHAR(20),
                amount DECIMAL(30,8),
                currency VARCHAR(10),
                timestamp TIMESTAMP,
                user_cluster VARCHAR(255),
                created_at TIMESTAMP DEFAULT NOW()
            );
            ''',
            '''
            CREATE INDEX IF NOT EXISTS idx_crypto_transactions_blockchain ON crypto_transactions(blockchain);
            CREATE INDEX IF NOT EXISTS idx_crypto_transactions_timestamp ON crypto_transactions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_wallet_clusters_risk_level ON wallet_clusters(risk_level);
            CREATE INDEX IF NOT EXISTS idx_mixing_services_blockchain ON mixing_services(blockchain);
            CREATE INDEX IF NOT EXISTS idx_exchange_flows_timestamp ON exchange_flows(timestamp);
            '''
        ]

        async with self.db_pool.acquire() as conn:
            for table_sql in tables:
                await conn.execute(table_sql)

    async def monitor_address(self, address: str, blockchain: str) -> Dict[str, Any]:
        """Add address to monitoring list"""

        self.monitored_addresses.add(f"{blockchain}:{address}")

        # Store in Redis for persistence
        await self.redis_client.sadd('monitored_addresses', f"{blockchain}:{address}")

        return {
            'status': 'success',
            'address': address,
            'blockchain': blockchain,
            'monitoring': True
        }

    async def analyze_transaction(self, tx_id: str, blockchain: str) -> Dict[str, Any]:
        """Comprehensive transaction analysis"""

        try:
            # Fetch transaction
            transaction = await self.blockchain_api.get_transaction(tx_id, blockchain)
            if not transaction:
                return {'error': 'Transaction not found'}

            # Store transaction
            await self._store_transaction(transaction)

            # Perform analysis
            analysis_results = {
                'transaction': asdict(transaction),
                'risk_assessment': await self._assess_transaction_risk(transaction),
                'clustering_analysis': None,
                'mixing_analysis': None,
                'exchange_analysis': None
            }

            # Wallet clustering analysis
            clusters = await self.clustering_engine.cluster_addresses([transaction])
            if clusters:
                analysis_results['clustering_analysis'] = [asdict(cluster) for cluster in clusters]
                # Store clusters
                for cluster in clusters:
                    await self._store_cluster(cluster)

            # Mixing service detection
            mixing_services = await self.mixing_detector.detect_mixing_services([transaction])
            if mixing_services:
                analysis_results['mixing_analysis'] = [asdict(service) for service in mixing_services]
                # Store mixing services
                for service in mixing_services:
                    await self._store_mixing_service(service)

            # Exchange flow analysis
            exchange_flows = await self.exchange_analyzer.analyze_exchange_flows([transaction])
            if exchange_flows:
                analysis_results['exchange_analysis'] = [asdict(flow) for flow in exchange_flows]
                # Store exchange flows
                for flow in exchange_flows:
                    await self._store_exchange_flow(flow)

            # Check alerts
            await self._check_alerts(transaction, analysis_results)

            # Publish to Kafka
            await self.kafka_producer.send(
                'crypto_transaction_analysis',
                key=tx_id,
                value=analysis_results
            )

            return analysis_results

        except Exception as e:
            logger.error(f"Error analyzing transaction {tx_id}: {e}")
            return {'error': str(e)}

    async def _assess_transaction_risk(self, transaction: Transaction) -> Dict[str, Any]:
        """Assess risk score for transaction"""

        risk_factors = []
        risk_score = 0.0

        # Large amount
        btc_equivalent = float(transaction.amount)  # Simplified
        if btc_equivalent > 10:
            risk_factors.append('large_amount')
            risk_score += 0.3

        # Multiple inputs/outputs
        if len(transaction.from_addresses) > 5 or len(transaction.to_addresses) > 5:
            risk_factors.append('multiple_addresses')
            risk_score += 0.2

        # Round amounts (often suspicious)
        if float(transaction.amount) == round(float(transaction.amount)):
            risk_factors.append('round_amount')
            risk_score += 0.1

        # Low fee (potential mixing)
        if transaction.fee < transaction.amount * Decimal('0.001'):
            risk_factors.append('low_fee')
            risk_score += 0.1

        # Privacy coin usage
        if transaction.blockchain in ['monero', 'zcash']:
            risk_factors.append('privacy_coin')
            risk_score += 0.4

        return {
            'risk_score': min(risk_score, 1.0),
            'risk_level': 'low' if risk_score < 0.3 else 'medium' if risk_score < 0.7 else 'high',
            'risk_factors': risk_factors
        }

    async def _check_alerts(self, transaction: Transaction, analysis: Dict[str, Any]):
        """Check if transaction triggers any alerts"""

        alerts = []

        # Large transaction alert
        if transaction.amount > self.alert_thresholds['large_transaction']:
            alerts.append({
                'type': 'large_transaction',
                'severity': 'medium',
                'message': f"Large {transaction.currency} transaction: {transaction.amount}"
            })

        # High risk transaction
        risk_score = analysis['risk_assessment']['risk_score']
        if risk_score > 0.7:
            alerts.append({
                'type': 'high_risk_transaction',
                'severity': 'high',
                'message': f"High risk transaction detected (score: {risk_score:.2f})"
            })

        # Mixing service detection
        if analysis['mixing_analysis']:
            for service in analysis['mixing_analysis']:
                if service['detection_confidence'] > self.alert_thresholds['mixing_confidence']:
                    alerts.append({
                        'type': 'mixing_service_detected',
                        'severity': 'high',
                        'message': f"Mixing service detected: {service['name']}"
                    })

        # Send alerts if any
        if alerts:
            for alert in alerts:
                await self.kafka_producer.send(
                    'crypto_alerts',
                    key=transaction.tx_id,
                    value={
                        'transaction_id': transaction.tx_id,
                        'alert': alert,
                        'timestamp': datetime.now().isoformat()
                    }
                )

    async def _store_transaction(self, transaction: Transaction):
        """Store transaction in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO crypto_transactions
                (tx_id, blockchain, block_height, timestamp, from_addresses, to_addresses,
                 amount, currency, fee, confirmations, transaction_type, risk_score, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (tx_id) DO UPDATE SET
                    confirmations = EXCLUDED.confirmations,
                    metadata = EXCLUDED.metadata
                ''',
                transaction.tx_id, transaction.blockchain, transaction.block_height,
                transaction.timestamp, transaction.from_addresses, transaction.to_addresses,
                transaction.amount, transaction.currency, transaction.fee,
                transaction.confirmations, transaction.transaction_type,
                transaction.risk_score, json.dumps(transaction.metadata)
            )

    async def _store_cluster(self, cluster: WalletCluster):
        """Store wallet cluster in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO wallet_clusters
                (cluster_id, addresses, blockchain, cluster_type, risk_level,
                 first_seen, last_activity, total_volume, transaction_count,
                 behavioral_patterns, connected_clusters)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (cluster_id) DO UPDATE SET
                    last_activity = EXCLUDED.last_activity,
                    total_volume = EXCLUDED.total_volume,
                    transaction_count = EXCLUDED.transaction_count,
                    behavioral_patterns = EXCLUDED.behavioral_patterns,
                    updated_at = NOW()
                ''',
                cluster.cluster_id, list(cluster.addresses), cluster.blockchain,
                cluster.cluster_type, cluster.risk_level, cluster.first_seen,
                cluster.last_activity, cluster.total_volume, cluster.transaction_count,
                json.dumps(cluster.behavioral_patterns), list(cluster.connected_clusters)
            )

    async def _store_mixing_service(self, service: MixingService):
        """Store mixing service in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO mixing_services
                (service_id, name, blockchain, addresses, detection_confidence,
                 mixing_patterns, volume_mixed, active_start, active_end, clients_detected)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (service_id) DO UPDATE SET
                    volume_mixed = EXCLUDED.volume_mixed,
                    active_end = EXCLUDED.active_end,
                    clients_detected = EXCLUDED.clients_detected
                ''',
                service.service_id, service.name, service.blockchain,
                list(service.addresses), service.detection_confidence,
                service.mixing_patterns, service.volume_mixed,
                service.active_period[0], service.active_period[1],
                service.clients_detected
            )

    async def _store_exchange_flow(self, flow: ExchangeFlow):
        """Store exchange flow in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO exchange_flows
                (exchange_id, exchange_name, blockchain, flow_direction, amount,
                 currency, timestamp, user_cluster)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ''',
                flow.exchange_id, flow.exchange_name, flow.blockchain,
                flow.flow_direction, flow.amount, flow.currency,
                flow.timestamp, flow.user_cluster
            )

    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""

        async with self.db_pool.acquire() as conn:
            # Transaction statistics
            tx_stats = await conn.fetchrow(
                '''
                SELECT
                    COUNT(*) as total_transactions,
                    COUNT(DISTINCT blockchain) as blockchains_monitored,
                    SUM(amount) as total_volume,
                    AVG(risk_score) as average_risk_score
                FROM crypto_transactions
                '''
            )

            # Cluster statistics
            cluster_stats = await conn.fetchrow(
                '''
                SELECT
                    COUNT(*) as total_clusters,
                    COUNT(*) FILTER (WHERE risk_level = 'high') as high_risk_clusters,
                    AVG(transaction_count) as avg_cluster_size
                FROM wallet_clusters
                '''
            )

            # Mixing service statistics
            mixing_stats = await conn.fetchrow(
                '''
                SELECT
                    COUNT(*) as mixing_services_detected,
                    SUM(volume_mixed) as total_mixed_volume,
                    AVG(detection_confidence) as avg_confidence
                FROM mixing_services
                '''
            )

        return {
            'transactions': dict(tx_stats) if tx_stats else {},
            'clusters': dict(cluster_stats) if cluster_stats else {},
            'mixing_services': dict(mixing_stats) if mixing_stats else {},
            'monitored_addresses': len(self.monitored_addresses)
        }

    async def cleanup(self):
        """Cleanup resources"""

        logger.info("Cleaning up Cryptocurrency Intelligence Engine...")

        if self.kafka_producer:
            await self.kafka_producer.stop()

        if self.redis_client:
            await self.redis_client.close()

        if self.db_pool:
            await self.db_pool.close()

        await self.blockchain_api.cleanup()


# Example usage
async def main():
    """Example usage of Cryptocurrency Intelligence Engine"""

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

    # Initialize engine
    engine = CryptocurrencyIntelligenceEngine(db_config, redis_config, kafka_config)
    await engine.initialize()

    try:
        # Monitor an address
        await engine.monitor_address(
            '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',  # Genesis address
            'bitcoin'
        )

        # Analyze a transaction (example)
        # analysis = await engine.analyze_transaction('tx_id_here', 'bitcoin')

        # Get statistics
        stats = await engine.get_statistics()
        print(f"Engine statistics: {stats}")

    except KeyboardInterrupt:
        logger.info("Shutting down engine...")
    finally:
        await engine.cleanup()


if __name__ == "__main__":
    asyncio.run(main())