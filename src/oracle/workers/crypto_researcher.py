#!/usr/bin/env python3
"""
Cryptocurrency Intelligence Worker for ORACLE1
Advanced cryptocurrency analysis and blockchain intelligence gathering
Supports Bitcoin, Ethereum, Monero, and other major cryptocurrencies
"""

import asyncio
import json
import time
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import aiohttp
import sqlite3
import threading
from urllib.parse import urljoin

# Crypto libraries
import ecdsa
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
import bech32

# Data analysis
import numpy as np
import pandas as pd
from collections import defaultdict, deque

# ORACLE integration
import redis
from influxdb_client import InfluxDBClient, Point, WritePrecision

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CryptocurrencyType(Enum):
    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"
    MONERO = "monero"
    LITECOIN = "litecoin"
    BITCOIN_CASH = "bitcoin_cash"
    ZCASH = "zcash"
    DASH = "dash"
    RIPPLE = "ripple"
    CHAINLINK = "chainlink"
    UNKNOWN = "unknown"

class TransactionType(Enum):
    STANDARD = "standard"
    MULTISIG = "multisig"
    COINJOIN = "coinjoin"
    MIXING = "mixing"
    EXCHANGE = "exchange"
    SMART_CONTRACT = "smart_contract"
    PRIVACY_COIN = "privacy_coin"
    ATOMIC_SWAP = "atomic_swap"
    LIGHTNING = "lightning"
    UNKNOWN = "unknown"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class WalletInfo:
    """Cryptocurrency wallet information"""
    address: str
    cryptocurrency: CryptocurrencyType
    balance: float
    transaction_count: int
    first_seen: datetime
    last_seen: datetime
    labels: List[str]
    risk_score: float
    cluster_id: Optional[str]

@dataclass
class TransactionInfo:
    """Cryptocurrency transaction information"""
    tx_hash: str
    cryptocurrency: CryptocurrencyType
    block_height: int
    timestamp: datetime
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    fee: float
    amount: float
    transaction_type: TransactionType
    privacy_features: List[str]
    risk_indicators: List[str]

@dataclass
class CryptoAnalysisResult:
    """Result of cryptocurrency analysis"""
    timestamp: datetime
    query_type: str  # address, transaction, cluster
    target: str
    cryptocurrency: CryptocurrencyType
    wallet_info: Optional[WalletInfo]
    transactions: List[TransactionInfo]
    risk_assessment: RiskLevel
    intelligence_summary: Dict[str, Any]
    recommendations: List[str]
    processing_time: float

class BlockchainAPIClient:
    """Blockchain API client with multiple provider support"""

    def __init__(self):
        self.providers = {
            'blockchair': {
                'base_url': 'https://api.blockchair.com',
                'rate_limit': 30  # requests per minute
            },
            'blockchain_info': {
                'base_url': 'https://blockchain.info',
                'rate_limit': 300  # requests per 5 minutes
            },
            'etherscan': {
                'base_url': 'https://api.etherscan.io/api',
                'rate_limit': 5  # requests per second
            }
        }

        self.rate_limiters = {}
        for provider in self.providers:
            self.rate_limiters[provider] = deque()

    async def _check_rate_limit(self, provider: str) -> bool:
        """Check if we can make a request to the provider"""
        current_time = time.time()
        rate_limit = self.providers[provider]['rate_limit']
        rate_limiter = self.rate_limiters[provider]

        # Clean old requests
        while rate_limiter and current_time - rate_limiter[0] > 60:
            rate_limiter.popleft()

        # Check if we can make a request
        if len(rate_limiter) < rate_limit:
            rate_limiter.append(current_time)
            return True

        return False

    async def get_bitcoin_address_info(self, address: str) -> Dict[str, Any]:
        """Get Bitcoin address information"""
        try:
            # Try Blockchair first
            if await self._check_rate_limit('blockchair'):
                url = f"{self.providers['blockchair']['base_url']}/bitcoin/dashboards/address/{address}"

                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            return self._parse_blockchair_address(data)

            # Fallback to blockchain.info
            if await self._check_rate_limit('blockchain_info'):
                url = f"{self.providers['blockchain_info']['base_url']}/rawaddr/{address}"

                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            return self._parse_blockchain_info_address(data)

        except Exception as e:
            logger.error(f"Bitcoin address lookup failed: {e}")

        return {}

    async def get_ethereum_address_info(self, address: str) -> Dict[str, Any]:
        """Get Ethereum address information"""
        try:
            if await self._check_rate_limit('etherscan'):
                # Get balance
                balance_url = f"{self.providers['etherscan']['base_url']}?module=account&action=balance&address={address}&tag=latest&apikey=YourApiKeyToken"

                # Get transaction list
                tx_url = f"{self.providers['etherscan']['base_url']}?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=desc&apikey=YourApiKeyToken"

                async with aiohttp.ClientSession() as session:
                    balance_task = session.get(balance_url)
                    tx_task = session.get(tx_url)

                    balance_response, tx_response = await asyncio.gather(balance_task, tx_task)

                    balance_data = await balance_response.json() if balance_response.status == 200 else {}
                    tx_data = await tx_response.json() if tx_response.status == 200 else {}

                return self._parse_etherscan_data(balance_data, tx_data)

        except Exception as e:
            logger.error(f"Ethereum address lookup failed: {e}")

        return {}

    async def get_transaction_info(self, tx_hash: str, cryptocurrency: CryptocurrencyType) -> Dict[str, Any]:
        """Get transaction information"""
        try:
            if cryptocurrency == CryptocurrencyType.BITCOIN:
                return await self._get_bitcoin_transaction(tx_hash)
            elif cryptocurrency == CryptocurrencyType.ETHEREUM:
                return await self._get_ethereum_transaction(tx_hash)
            else:
                logger.warning(f"Transaction lookup not implemented for {cryptocurrency}")

        except Exception as e:
            logger.error(f"Transaction lookup failed: {e}")

        return {}

    async def _get_bitcoin_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """Get Bitcoin transaction details"""
        if await self._check_rate_limit('blockchair'):
            url = f"{self.providers['blockchair']['base_url']}/bitcoin/dashboards/transaction/{tx_hash}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_blockchair_transaction(data)

        return {}

    async def _get_ethereum_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """Get Ethereum transaction details"""
        if await self._check_rate_limit('etherscan'):
            url = f"{self.providers['etherscan']['base_url']}?module=proxy&action=eth_getTransactionByHash&txhash={tx_hash}&apikey=YourApiKeyToken"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_etherscan_transaction(data)

        return {}

    def _parse_blockchair_address(self, data: Dict) -> Dict:
        """Parse Blockchair address response"""
        if 'data' not in data:
            return {}

        address_data = list(data['data'].values())[0]
        address_info = address_data.get('address', {})

        return {
            'balance': address_info.get('balance', 0) / 100000000,  # Convert satoshi to BTC
            'transaction_count': address_info.get('transaction_count', 0),
            'first_seen': address_info.get('first_seen_receiving'),
            'last_seen': address_info.get('last_seen_receiving'),
            'received': address_info.get('received', 0) / 100000000,
            'spent': address_info.get('spent', 0) / 100000000,
            'transactions': address_data.get('transactions', [])
        }

    def _parse_blockchain_info_address(self, data: Dict) -> Dict:
        """Parse blockchain.info address response"""
        return {
            'balance': data.get('final_balance', 0) / 100000000,
            'transaction_count': data.get('n_tx', 0),
            'received': data.get('total_received', 0) / 100000000,
            'spent': data.get('total_sent', 0) / 100000000,
            'transactions': data.get('txs', [])
        }

    def _parse_etherscan_data(self, balance_data: Dict, tx_data: Dict) -> Dict:
        """Parse Etherscan API responses"""
        balance = 0
        if balance_data.get('status') == '1':
            balance = int(balance_data.get('result', '0')) / 10**18  # Convert wei to ETH

        transactions = []
        if tx_data.get('status') == '1':
            transactions = tx_data.get('result', [])

        return {
            'balance': balance,
            'transaction_count': len(transactions),
            'transactions': transactions
        }

    def _parse_blockchair_transaction(self, data: Dict) -> Dict:
        """Parse Blockchair transaction response"""
        if 'data' not in data:
            return {}

        tx_data = list(data['data'].values())[0]
        transaction = tx_data.get('transaction', {})

        return {
            'hash': transaction.get('hash'),
            'block_id': transaction.get('block_id'),
            'time': transaction.get('time'),
            'inputs': tx_data.get('inputs', []),
            'outputs': tx_data.get('outputs', []),
            'fee': transaction.get('fee', 0) / 100000000,
            'size': transaction.get('size')
        }

    def _parse_etherscan_transaction(self, data: Dict) -> Dict:
        """Parse Etherscan transaction response"""
        if data.get('result'):
            result = data['result']
            return {
                'hash': result.get('hash'),
                'block_number': int(result.get('blockNumber', '0'), 16),
                'from': result.get('from'),
                'to': result.get('to'),
                'value': int(result.get('value', '0'), 16) / 10**18,
                'gas': int(result.get('gas', '0'), 16),
                'gas_price': int(result.get('gasPrice', '0'), 16),
                'gas_used': result.get('gasUsed')
            }

        return {}

class CryptocurrencyAnalyzer:
    """Cryptocurrency analysis engine"""

    def __init__(self):
        self.api_client = BlockchainAPIClient()
        self.address_clusters = defaultdict(set)
        self.known_entities = self._load_known_entities()

    def _load_known_entities(self) -> Dict[str, str]:
        """Load known cryptocurrency entities (exchanges, mixers, etc.)"""
        # In production, this would load from a comprehensive database
        return {
            # Known exchange addresses (examples)
            '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa': 'Genesis Block',
            '3FpjdDTdRGBMJU1VCS8Edc7Wh3mJ6T3G9g': 'Exchange Cold Wallet',

            # Known mixer services
            '1tumbler123...': 'Bitcoin Mixer',

            # Known gambling sites
            '1dice8EMZmqKvrGE4Qc9bUFf9PX3xaYDp': 'SatoshiDice'
        }

    async def analyze_address(self, address: str, cryptocurrency: CryptocurrencyType) -> WalletInfo:
        """Analyze a cryptocurrency address"""
        logger.info(f"Analyzing {cryptocurrency.value} address: {address}")

        try:
            # Get address information from blockchain
            if cryptocurrency == CryptocurrencyType.BITCOIN:
                api_data = await self.api_client.get_bitcoin_address_info(address)
            elif cryptocurrency == CryptocurrencyType.ETHEREUM:
                api_data = await self.api_client.get_ethereum_address_info(address)
            else:
                raise ValueError(f"Unsupported cryptocurrency: {cryptocurrency}")

            if not api_data:
                logger.warning(f"No data found for address: {address}")
                return None

            # Extract basic information
            balance = api_data.get('balance', 0)
            tx_count = api_data.get('transaction_count', 0)

            # Parse timestamps
            first_seen = self._parse_timestamp(api_data.get('first_seen'))
            last_seen = self._parse_timestamp(api_data.get('last_seen'))

            # Determine labels and cluster
            labels = await self._determine_address_labels(address, api_data)
            cluster_id = await self._find_address_cluster(address, api_data)

            # Calculate risk score
            risk_score = await self._calculate_risk_score(address, api_data, labels)

            return WalletInfo(
                address=address,
                cryptocurrency=cryptocurrency,
                balance=balance,
                transaction_count=tx_count,
                first_seen=first_seen or datetime.now(),
                last_seen=last_seen or datetime.now(),
                labels=labels,
                risk_score=risk_score,
                cluster_id=cluster_id
            )

        except Exception as e:
            logger.error(f"Address analysis failed: {e}")
            raise

    async def analyze_transaction(self, tx_hash: str, cryptocurrency: CryptocurrencyType) -> TransactionInfo:
        """Analyze a cryptocurrency transaction"""
        logger.info(f"Analyzing {cryptocurrency.value} transaction: {tx_hash}")

        try:
            # Get transaction data
            tx_data = await self.api_client.get_transaction_info(tx_hash, cryptocurrency)

            if not tx_data:
                logger.warning(f"No data found for transaction: {tx_hash}")
                return None

            # Parse transaction details
            timestamp = self._parse_timestamp(tx_data.get('time'))
            block_height = tx_data.get('block_id', 0)

            # Analyze inputs and outputs
            inputs = self._parse_transaction_inputs(tx_data.get('inputs', []), cryptocurrency)
            outputs = self._parse_transaction_outputs(tx_data.get('outputs', []), cryptocurrency)

            # Calculate amounts and fees
            total_amount = sum(output.get('value', 0) for output in outputs)
            fee = tx_data.get('fee', 0)

            # Determine transaction type
            tx_type = await self._classify_transaction_type(inputs, outputs, tx_data)

            # Identify privacy features
            privacy_features = await self._identify_privacy_features(inputs, outputs, tx_data)

            # Check for risk indicators
            risk_indicators = await self._identify_risk_indicators(inputs, outputs, tx_data)

            return TransactionInfo(
                tx_hash=tx_hash,
                cryptocurrency=cryptocurrency,
                block_height=block_height,
                timestamp=timestamp or datetime.now(),
                inputs=inputs,
                outputs=outputs,
                fee=fee,
                amount=total_amount,
                transaction_type=tx_type,
                privacy_features=privacy_features,
                risk_indicators=risk_indicators
            )

        except Exception as e:
            logger.error(f"Transaction analysis failed: {e}")
            raise

    def _parse_timestamp(self, timestamp: Union[str, int, None]) -> Optional[datetime]:
        """Parse timestamp from various formats"""
        if not timestamp:
            return None

        try:
            if isinstance(timestamp, str):
                # Try ISO format first
                try:
                    return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    pass

                # Try parsing as Unix timestamp
                try:
                    return datetime.fromtimestamp(int(timestamp))
                except:
                    pass

            elif isinstance(timestamp, (int, float)):
                return datetime.fromtimestamp(timestamp)

        except Exception as e:
            logger.error(f"Timestamp parsing failed: {e}")

        return None

    async def _determine_address_labels(self, address: str, api_data: Dict) -> List[str]:
        """Determine labels for an address"""
        labels = []

        # Check against known entities
        if address in self.known_entities:
            labels.append(self.known_entities[address])

        # Analyze transaction patterns
        tx_count = api_data.get('transaction_count', 0)
        balance = api_data.get('balance', 0)

        if tx_count > 1000:
            labels.append('High Activity')

        if balance > 100:  # Adjust threshold based on currency
            labels.append('Large Balance')

        if tx_count > 100 and balance < 0.1:
            labels.append('Possible Mixer')

        # Check for exchange patterns
        if await self._is_exchange_pattern(api_data):
            labels.append('Exchange')

        return labels

    async def _find_address_cluster(self, address: str, api_data: Dict) -> Optional[str]:
        """Find cluster ID for an address"""
        # Simplified clustering based on common spending patterns
        # In production, this would use sophisticated clustering algorithms

        cluster_id = hashlib.md5(address.encode()).hexdigest()[:8]
        return f"cluster_{cluster_id}"

    async def _calculate_risk_score(self, address: str, api_data: Dict, labels: List[str]) -> float:
        """Calculate risk score for an address"""
        risk_score = 0.0

        # Base risk from labels
        risk_labels = {
            'Mixer': 0.8,
            'Gambling': 0.6,
            'Darknet Market': 0.9,
            'Ransomware': 1.0,
            'Exchange': 0.2,
            'High Activity': 0.3
        }

        for label in labels:
            risk_score = max(risk_score, risk_labels.get(label, 0.0))

        # Additional risk factors
        tx_count = api_data.get('transaction_count', 0)
        balance = api_data.get('balance', 0)

        # Rapid turnover (high tx count, low balance)
        if tx_count > 50 and balance < 1.0:
            risk_score += 0.2

        # Large round numbers might indicate automated systems
        if balance > 0 and balance == int(balance):
            risk_score += 0.1

        return min(risk_score, 1.0)

    def _parse_transaction_inputs(self, inputs: List[Dict], cryptocurrency: CryptocurrencyType) -> List[Dict]:
        """Parse transaction inputs"""
        parsed_inputs = []

        for inp in inputs:
            if cryptocurrency == CryptocurrencyType.BITCOIN:
                parsed_inputs.append({
                    'address': inp.get('recipient'),
                    'value': inp.get('value', 0) / 100000000,
                    'index': inp.get('index')
                })
            elif cryptocurrency == CryptocurrencyType.ETHEREUM:
                parsed_inputs.append({
                    'address': inp.get('from'),
                    'value': inp.get('value', 0),
                    'index': inp.get('index')
                })

        return parsed_inputs

    def _parse_transaction_outputs(self, outputs: List[Dict], cryptocurrency: CryptocurrencyType) -> List[Dict]:
        """Parse transaction outputs"""
        parsed_outputs = []

        for out in outputs:
            if cryptocurrency == CryptocurrencyType.BITCOIN:
                parsed_outputs.append({
                    'address': out.get('recipient'),
                    'value': out.get('value', 0) / 100000000,
                    'index': out.get('index')
                })
            elif cryptocurrency == CryptocurrencyType.ETHEREUM:
                parsed_outputs.append({
                    'address': out.get('to'),
                    'value': out.get('value', 0),
                    'index': out.get('index')
                })

        return parsed_outputs

    async def _classify_transaction_type(self, inputs: List[Dict], outputs: List[Dict], tx_data: Dict) -> TransactionType:
        """Classify transaction type"""

        # Check for CoinJoin patterns (multiple inputs/outputs with similar values)
        if len(inputs) > 1 and len(outputs) > 1:
            output_values = [out.get('value', 0) for out in outputs]
            if len(set(output_values)) < len(output_values) / 2:  # Many similar values
                return TransactionType.COINJOIN

        # Check for mixing patterns
        if len(outputs) > 10 and all(out.get('value', 0) < 1.0 for out in outputs):
            return TransactionType.MIXING

        # Check for multisig
        # This would require script analysis in a real implementation
        if any('multisig' in str(out) for out in outputs):
            return TransactionType.MULTISIG

        # Check for exchange patterns
        if len(outputs) == 2 and abs(outputs[0].get('value', 0) - outputs[1].get('value', 0)) > 10:
            return TransactionType.EXCHANGE

        return TransactionType.STANDARD

    async def _identify_privacy_features(self, inputs: List[Dict], outputs: List[Dict], tx_data: Dict) -> List[str]:
        """Identify privacy features in transaction"""
        features = []

        # Multiple inputs/outputs for privacy
        if len(inputs) > 2:
            features.append('Multiple Inputs')

        if len(outputs) > 2:
            features.append('Multiple Outputs')

        # Change address detection (simplified)
        if len(outputs) == 2:
            values = [out.get('value', 0) for out in outputs]
            if min(values) < max(values) * 0.1:  # One much smaller output
                features.append('Change Address')

        # Round number avoidance
        if any(out.get('value', 0) % 1 != 0 for out in outputs):
            features.append('Non-Round Amounts')

        return features

    async def _identify_risk_indicators(self, inputs: List[Dict], outputs: List[Dict], tx_data: Dict) -> List[str]:
        """Identify risk indicators in transaction"""
        indicators = []

        # Very large amounts
        total_value = sum(out.get('value', 0) for out in outputs)
        if total_value > 1000:  # Adjust threshold
            indicators.append('Large Amount')

        # Many small outputs (possible tumbler)
        if len(outputs) > 20 and all(out.get('value', 0) < 0.1 for out in outputs):
            indicators.append('Possible Tumbling')

        # Known bad addresses (simplified check)
        all_addresses = [inp.get('address') for inp in inputs] + [out.get('address') for out in outputs]
        bad_keywords = ['mixer', 'tumbler', 'dark', 'ransom']

        for addr in all_addresses:
            if addr and any(keyword in addr.lower() for keyword in bad_keywords):
                indicators.append('Suspicious Address Pattern')

        return indicators

    async def _is_exchange_pattern(self, api_data: Dict) -> bool:
        """Check if address shows exchange patterns"""
        tx_count = api_data.get('transaction_count', 0)
        balance = api_data.get('balance', 0)

        # High transaction count with moderate balance retention
        if tx_count > 500 and 0.1 < balance < 1000:
            return True

        return False

class CryptocurrencyResearchWorker:
    """Main cryptocurrency research worker for ORACLE1"""

    def __init__(self):
        self.analyzer = CryptocurrencyAnalyzer()

        # Data storage
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.influx_client = InfluxDBClient(
            url="http://localhost:8086",
            token="oracle-research-token",
            org="bev-research"
        )

        # Local database for caching
        self._init_local_db()

        self.running = True

    def _init_local_db(self):
        """Initialize local SQLite database for caching"""
        self.db_path = "/tmp/crypto_research.db"
        conn = sqlite3.connect(self.db_path)

        conn.execute('''
            CREATE TABLE IF NOT EXISTS address_cache (
                address TEXT PRIMARY KEY,
                cryptocurrency TEXT,
                data TEXT,
                timestamp REAL
            )
        ''')

        conn.execute('''
            CREATE TABLE IF NOT EXISTS transaction_cache (
                tx_hash TEXT PRIMARY KEY,
                cryptocurrency TEXT,
                data TEXT,
                timestamp REAL
            )
        ''')

        conn.commit()
        conn.close()

    async def research_address(self, address: str, cryptocurrency: str = None) -> CryptoAnalysisResult:
        """Research a cryptocurrency address"""
        start_time = time.time()
        logger.info(f"Researching address: {address}")

        try:
            # Detect cryptocurrency if not provided
            if not cryptocurrency:
                crypto_type = await self._detect_cryptocurrency(address)
            else:
                crypto_type = CryptocurrencyType(cryptocurrency.lower())

            # Check cache first
            cached_data = self._get_cached_address(address, crypto_type)
            if cached_data and time.time() - cached_data['timestamp'] < 3600:  # 1 hour cache
                logger.info("Using cached address data")
                wallet_info = WalletInfo(**json.loads(cached_data['data']))
            else:
                # Analyze address
                wallet_info = await self.analyzer.analyze_address(address, crypto_type)

                # Cache result
                if wallet_info:
                    self._cache_address(address, crypto_type, wallet_info)

            if not wallet_info:
                raise ValueError(f"Failed to analyze address: {address}")

            # Get recent transactions
            transactions = await self._get_address_transactions(address, crypto_type, limit=50)

            # Assess overall risk
            risk_level = await self._assess_overall_risk(wallet_info, transactions)

            # Generate intelligence summary
            intelligence_summary = await self._generate_intelligence_summary(wallet_info, transactions)

            # Generate recommendations
            recommendations = await self._generate_recommendations(wallet_info, transactions, risk_level)

            processing_time = time.time() - start_time

            result = CryptoAnalysisResult(
                timestamp=datetime.now(),
                query_type="address",
                target=address,
                cryptocurrency=crypto_type,
                wallet_info=wallet_info,
                transactions=transactions,
                risk_assessment=risk_level,
                intelligence_summary=intelligence_summary,
                recommendations=recommendations,
                processing_time=processing_time
            )

            # Store results
            await self._store_analysis_result(result)

            logger.info(f"Address research completed in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Address research failed: {e}")
            raise

    async def research_transaction(self, tx_hash: str, cryptocurrency: str = None) -> CryptoAnalysisResult:
        """Research a cryptocurrency transaction"""
        start_time = time.time()
        logger.info(f"Researching transaction: {tx_hash}")

        try:
            # Detect cryptocurrency if not provided
            if not cryptocurrency:
                crypto_type = await self._detect_cryptocurrency_from_tx(tx_hash)
            else:
                crypto_type = CryptocurrencyType(cryptocurrency.lower())

            # Check cache
            cached_data = self._get_cached_transaction(tx_hash, crypto_type)
            if cached_data and time.time() - cached_data['timestamp'] < 3600:
                logger.info("Using cached transaction data")
                transaction_info = TransactionInfo(**json.loads(cached_data['data']))
            else:
                # Analyze transaction
                transaction_info = await self.analyzer.analyze_transaction(tx_hash, crypto_type)

                # Cache result
                if transaction_info:
                    self._cache_transaction(tx_hash, crypto_type, transaction_info)

            if not transaction_info:
                raise ValueError(f"Failed to analyze transaction: {tx_hash}")

            # Assess risk
            risk_level = await self._assess_transaction_risk(transaction_info)

            # Generate intelligence summary
            intelligence_summary = await self._generate_transaction_intelligence(transaction_info)

            # Generate recommendations
            recommendations = await self._generate_transaction_recommendations(transaction_info, risk_level)

            processing_time = time.time() - start_time

            result = CryptoAnalysisResult(
                timestamp=datetime.now(),
                query_type="transaction",
                target=tx_hash,
                cryptocurrency=crypto_type,
                wallet_info=None,
                transactions=[transaction_info],
                risk_assessment=risk_level,
                intelligence_summary=intelligence_summary,
                recommendations=recommendations,
                processing_time=processing_time
            )

            # Store results
            await self._store_analysis_result(result)

            logger.info(f"Transaction research completed in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Transaction research failed: {e}")
            raise

    async def _detect_cryptocurrency(self, address: str) -> CryptocurrencyType:
        """Detect cryptocurrency type from address format"""
        try:
            # Bitcoin address patterns
            if address.startswith(('1', '3', 'bc1')):
                return CryptocurrencyType.BITCOIN

            # Ethereum address pattern
            if address.startswith('0x') and len(address) == 42:
                return CryptocurrencyType.ETHEREUM

            # Litecoin patterns
            if address.startswith(('L', 'M', 'ltc1')):
                return CryptocurrencyType.LITECOIN

            # Bitcoin Cash patterns
            if address.startswith(('q', 'p', 'bitcoincash:')):
                return CryptocurrencyType.BITCOIN_CASH

            # Monero patterns
            if address.startswith('4') and len(address) == 95:
                return CryptocurrencyType.MONERO

        except Exception as e:
            logger.error(f"Cryptocurrency detection failed: {e}")

        return CryptocurrencyType.UNKNOWN

    async def _detect_cryptocurrency_from_tx(self, tx_hash: str) -> CryptocurrencyType:
        """Detect cryptocurrency from transaction hash format"""
        # Bitcoin and most cryptocurrencies use 64-character hex hashes
        if len(tx_hash) == 64 and all(c in '0123456789abcdefABCDEF' for c in tx_hash):
            # Would need additional logic to distinguish between different cryptos
            return CryptocurrencyType.BITCOIN

        return CryptocurrencyType.UNKNOWN

    def _get_cached_address(self, address: str, crypto_type: CryptocurrencyType) -> Optional[Dict]:
        """Get cached address data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT data, timestamp FROM address_cache WHERE address = ? AND cryptocurrency = ?",
                (address, crypto_type.value)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return {'data': row[0], 'timestamp': row[1]}

        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")

        return None

    def _cache_address(self, address: str, crypto_type: CryptocurrencyType, wallet_info: WalletInfo):
        """Cache address analysis result"""
        try:
            data = json.dumps(asdict(wallet_info), default=str)

            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO address_cache (address, cryptocurrency, data, timestamp) VALUES (?, ?, ?, ?)",
                (address, crypto_type.value, data, time.time())
            )
            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Cache storage failed: {e}")

    def _get_cached_transaction(self, tx_hash: str, crypto_type: CryptocurrencyType) -> Optional[Dict]:
        """Get cached transaction data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT data, timestamp FROM transaction_cache WHERE tx_hash = ? AND cryptocurrency = ?",
                (tx_hash, crypto_type.value)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return {'data': row[0], 'timestamp': row[1]}

        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")

        return None

    def _cache_transaction(self, tx_hash: str, crypto_type: CryptocurrencyType, transaction_info: TransactionInfo):
        """Cache transaction analysis result"""
        try:
            data = json.dumps(asdict(transaction_info), default=str)

            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT OR REPLACE INTO transaction_cache (tx_hash, cryptocurrency, data, timestamp) VALUES (?, ?, ?, ?)",
                (tx_hash, crypto_type.value, data, time.time())
            )
            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Cache storage failed: {e}")

    async def _get_address_transactions(self, address: str, crypto_type: CryptocurrencyType, limit: int = 50) -> List[TransactionInfo]:
        """Get recent transactions for an address"""
        # This would be implemented with proper API calls
        # For now, return empty list
        return []

    async def _assess_overall_risk(self, wallet_info: WalletInfo, transactions: List[TransactionInfo]) -> RiskLevel:
        """Assess overall risk level"""
        risk_score = wallet_info.risk_score

        # Adjust based on transaction patterns
        if transactions:
            high_risk_count = sum(1 for tx in transactions if tx.risk_indicators)
            risk_adjustment = high_risk_count / len(transactions) * 0.3
            risk_score += risk_adjustment

        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    async def _assess_transaction_risk(self, transaction_info: TransactionInfo) -> RiskLevel:
        """Assess transaction-specific risk"""
        risk_indicators = len(transaction_info.risk_indicators)
        privacy_features = len(transaction_info.privacy_features)

        if risk_indicators >= 3:
            return RiskLevel.CRITICAL
        elif risk_indicators >= 2 or privacy_features >= 3:
            return RiskLevel.HIGH
        elif risk_indicators >= 1 or privacy_features >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    async def _generate_intelligence_summary(self, wallet_info: WalletInfo, transactions: List[TransactionInfo]) -> Dict[str, Any]:
        """Generate intelligence summary"""
        return {
            'wallet_age_days': (datetime.now() - wallet_info.first_seen).days if wallet_info.first_seen else 0,
            'activity_level': 'High' if wallet_info.transaction_count > 100 else 'Low',
            'balance_category': 'Large' if wallet_info.balance > 10 else 'Small',
            'entity_type': wallet_info.labels[0] if wallet_info.labels else 'Unknown',
            'risk_category': 'High Risk' if wallet_info.risk_score > 0.6 else 'Low Risk',
            'cluster_size': 1,  # Would be calculated from actual clustering
            'transaction_patterns': list(set(tx.transaction_type.value for tx in transactions))
        }

    async def _generate_transaction_intelligence(self, transaction_info: TransactionInfo) -> Dict[str, Any]:
        """Generate transaction intelligence summary"""
        return {
            'transaction_age_days': (datetime.now() - transaction_info.timestamp).days if transaction_info.timestamp else 0,
            'amount_category': 'Large' if transaction_info.amount > 100 else 'Small',
            'complexity': 'Complex' if len(transaction_info.inputs) > 1 or len(transaction_info.outputs) > 2 else 'Simple',
            'privacy_score': len(transaction_info.privacy_features),
            'risk_score': len(transaction_info.risk_indicators),
            'transaction_type': transaction_info.transaction_type.value
        }

    async def _generate_recommendations(self, wallet_info: WalletInfo, transactions: List[TransactionInfo], risk_level: RiskLevel) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("CRITICAL: Address shows high-risk indicators - investigate immediately")
            recommendations.append("Consider blocking or flagging this address")

        elif risk_level == RiskLevel.HIGH:
            recommendations.append("HIGH RISK: Enhanced monitoring recommended")
            recommendations.append("Verify source of funds before processing")

        if 'Mixer' in wallet_info.labels:
            recommendations.append("Address associated with mixing services - enhanced due diligence required")

        if wallet_info.transaction_count > 1000:
            recommendations.append("High-activity address - likely commercial or exchange entity")

        if not recommendations:
            recommendations.append("Low risk profile - standard monitoring sufficient")

        return recommendations

    async def _generate_transaction_recommendations(self, transaction_info: TransactionInfo, risk_level: RiskLevel) -> List[str]:
        """Generate transaction-specific recommendations"""
        recommendations = []

        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("CRITICAL: Transaction shows multiple risk indicators")

        if transaction_info.transaction_type == TransactionType.MIXING:
            recommendations.append("Transaction shows mixing patterns - enhanced scrutiny required")

        if transaction_info.amount > 1000:
            recommendations.append("Large value transaction - verify legitimacy")

        if not recommendations:
            recommendations.append("Transaction appears normal - routine processing acceptable")

        return recommendations

    async def _store_analysis_result(self, result: CryptoAnalysisResult):
        """Store analysis result in databases"""
        try:
            # Store in Redis
            key = f"crypto:analysis:{int(time.time())}"
            data = asdict(result)
            data['timestamp'] = result.timestamp.isoformat()

            self.redis_client.hset(key, mapping={k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                                                for k, v in data.items()})
            self.redis_client.expire(key, 86400 * 7)  # 7 days

            # Store in InfluxDB
            point = Point("crypto_analysis") \
                .tag("cryptocurrency", result.cryptocurrency.value) \
                .tag("query_type", result.query_type) \
                .tag("risk_level", result.risk_assessment.value) \
                .field("processing_time", result.processing_time) \
                .field("transaction_count", len(result.transactions)) \
                .time(result.timestamp, WritePrecision.NS)

            write_api = self.influx_client.write_api()
            write_api.write(bucket="oracle-research", org="bev-research", record=point)

        except Exception as e:
            logger.error(f"Failed to store analysis result: {e}")

if __name__ == "__main__":
    async def main():
        worker = CryptocurrencyResearchWorker()

        # Example address analysis
        test_address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"  # Genesis block
        result = await worker.research_address(test_address, "bitcoin")

        print(f"Analysis completed: Risk level = {result.risk_assessment.value}")
        print(f"Wallet balance: {result.wallet_info.balance if result.wallet_info else 'N/A'}")

    asyncio.run(main())