"""Cryptocurrency Tracker Analyzer for IntelOwl
Tracks Bitcoin/Ethereum transactions and identifies wallet clusters
Monitors mixer usage and exchange flows
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib
from celery import shared_task
from api_app.analyzers_manager.observable_analyzers import ObservableAnalyzer
import logging
from neo4j import GraphDatabase
import networkx as nx
from decimal import Decimal

logger = logging.getLogger(__name__)


class CryptoTrackerAnalyzer(ObservableAnalyzer):
    """Track cryptocurrency transactions and wallet clusters"""
    
    KNOWN_EXCHANGES = {
        'binance': ['1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s', 'bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h'],
        'coinbase': ['1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa', '3Cbq7aT1tY8kMxWLbitaG7yT6bPbKChq64'],
        'kraken': ['3KHkGPNjvHPpxYmfJccuYCBGFYs7V3sSVp'],
        'bitfinex': ['3JZq6p8UW8M9XN3LAcCGbQPZhw8rRahqFe'],
        # Add more exchange addresses
    }
    
    MIXER_PATTERNS = {
        'chipmixer': ['1ChipGeeK8', '1ChipVNbW4'],
        'wasabi': ['bc1qs604c7jv6amk4cxqlnvuxv26hv3e48cds4m0ew'],
        'tornado_cash': ['0xA160cdAB225685dA1d56aa342Ad8841c3b53f291']  # ETH
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.etherscan_api = os.getenv('ETHERSCAN_API_KEY')
        self.blockchain_api = os.getenv('BLOCKCHAIN_INFO_API_KEY')
        self.chainalysis_api = os.getenv('CHAINALYSIS_API_KEY')
        self.neo4j_driver = None
        
    def set_params(self, params):
        """Set analyzer parameters"""
        self.depth = params.get('depth', 3)  # Transaction depth to analyze
        self.cluster_analysis = params.get('cluster_analysis', True)
        self.mixer_detection = params.get('mixer_detection', True)
        self.exchange_identification = params.get('exchange_identification', True)
        self.risk_scoring = params.get('risk_scoring', True)
        self.time_window = params.get('time_window', 30)  # Days to analyze
        
    def run(self):
        """Execute cryptocurrency analysis"""
        observable = self.observable_name
        observable_type = self.observable_classification
        
        results = {
            'address_info': {},
            'transactions': [],
            'wallet_cluster': [],
            'mixer_usage': [],
            'exchange_interactions': [],
            'money_flow': {},
            'risk_assessment': {},
            'criminal_associations': [],
            'total_volume': 0
        }
        
        try:
            # Detect cryptocurrency type
            crypto_type = self._detect_crypto_type(observable)
            results['address_info']['type'] = crypto_type
            
            if crypto_type == 'bitcoin':
                btc_analysis = self._analyze_bitcoin_address(observable)
                results.update(btc_analysis)
                
            elif crypto_type == 'ethereum':
                eth_analysis = self._analyze_ethereum_address(observable)
                results.update(eth_analysis)
                
            elif crypto_type == 'monero':
                xmr_analysis = self._analyze_monero_address(observable)
                results.update(xmr_analysis)
                
            # Perform wallet clustering
            if self.cluster_analysis:
                cluster = self._perform_wallet_clustering(observable, results['transactions'])
                results['wallet_cluster'] = cluster
                
            # Detect mixer usage
            if self.mixer_detection:
                mixer_txs = self._detect_mixer_usage(results['transactions'])
                results['mixer_usage'] = mixer_txs
                
            # Identify exchange interactions
            if self.exchange_identification:
                exchange_txs = self._identify_exchange_interactions(results['transactions'])
                results['exchange_interactions'] = exchange_txs
                
            # Analyze money flow
            results['money_flow'] = self._analyze_money_flow(results['transactions'])
            
            # Risk assessment
            if self.risk_scoring:
                results['risk_assessment'] = self._assess_risk(results)
                
            # Check criminal associations
            criminal_links = self._check_criminal_associations(observable, results)
            results['criminal_associations'] = criminal_links
            
            # Store in Neo4j for graph analysis
            self._store_in_neo4j(observable, results)
            
            # Cache results
            self._cache_results(observable, results)
            
        except Exception as e:
            logger.error(f"Crypto analysis failed: {str(e)}")
            return {'error': str(e)}
            
        return results
        
    def _detect_crypto_type(self, address: str) -> str:
        """Detect cryptocurrency type from address format"""
        if address.startswith(('1', '3')) and len(address) in range(26, 35):
            return 'bitcoin'
        elif address.startswith('bc1') and len(address) in range(42, 63):
            return 'bitcoin'
        elif address.startswith('0x') and len(address) == 42:
            return 'ethereum'
        elif address.startswith('4') and len(address) == 95:
            return 'monero'
        elif address.startswith(('X', 'T', 'M', 'L')) and len(address) == 34:
            return 'litecoin'
        else:
            return 'unknown'
            
    def _analyze_bitcoin_address(self, address: str) -> Dict:
        """Analyze Bitcoin address using blockchain APIs"""
        analysis = {
            'address_info': {},
            'transactions': [],
            'total_volume': 0
        }
        
        try:
            # Get address info from Blockchain.info
            response = requests.get(
                f'https://blockchain.info/rawaddr/{address}',
                params={'api_code': self.blockchain_api} if self.blockchain_api else {},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                analysis['address_info'] = {
                    'address': address,
                    'balance': data.get('final_balance', 0) / 100000000,  # Convert from satoshis
                    'total_received': data.get('total_received', 0) / 100000000,
                    'total_sent': data.get('total_sent', 0) / 100000000,
                    'n_transactions': data.get('n_tx', 0),
                    'first_seen': datetime.fromtimestamp(data.get('first_seen_time', 0)).isoformat() if data.get('first_seen_time') else None
                }
                
                # Process transactions
                for tx in data.get('txs', [])[:100]:  # Limit to recent 100
                    tx_info = self._process_bitcoin_transaction(tx, address)
                    analysis['transactions'].append(tx_info)
                    
                analysis['total_volume'] = analysis['address_info']['total_received'] + analysis['address_info']['total_sent']
                
            # Get additional data from other sources if available
            if self.chainalysis_api:
                chainalysis_data = self._get_chainalysis_data(address)
                analysis['address_info'].update(chainalysis_data)
                
        except Exception as e:
            logger.error(f"Bitcoin analysis failed: {str(e)}")
            
        return analysis
        
    def _analyze_ethereum_address(self, address: str) -> Dict:
        """Analyze Ethereum address using Etherscan API"""
        analysis = {
            'address_info': {},
            'transactions': [],
            'token_transfers': [],
            'smart_contracts': [],
            'total_volume': 0
        }
        
        try:
            if self.etherscan_api:
                # Get address balance
                balance_response = requests.get(
                    'https://api.etherscan.io/api',
                    params={
                        'module': 'account',
                        'action': 'balance',
                        'address': address,
                        'tag': 'latest',
                        'apikey': self.etherscan_api
                    },
                    timeout=30
                )
                
                if balance_response.status_code == 200:
                    balance_data = balance_response.json()
                    if balance_data['status'] == '1':
                        balance_wei = int(balance_data['result'])
                        balance_eth = balance_wei / 10**18
                        
                        analysis['address_info']['balance'] = balance_eth
                        
                # Get transaction list
                tx_response = requests.get(
                    'https://api.etherscan.io/api',
                    params={
                        'module': 'account',
                        'action': 'txlist',
                        'address': address,
                        'startblock': 0,
                        'endblock': 99999999,
                        'sort': 'desc',
                        'apikey': self.etherscan_api
                    },
                    timeout=30
                )
                
                if tx_response.status_code == 200:
                    tx_data = tx_response.json()
                    if tx_data['status'] == '1':
                        for tx in tx_data['result'][:100]:  # Recent 100
                            tx_info = self._process_ethereum_transaction(tx, address)
                            analysis['transactions'].append(tx_info)
                            
                # Get token transfers
                token_response = requests.get(
                    'https://api.etherscan.io/api',
                    params={
                        'module': 'account',
                        'action': 'tokentx',
                        'address': address,
                        'startblock': 0,
                        'endblock': 99999999,
                        'sort': 'desc',
                        'apikey': self.etherscan_api
                    },
                    timeout=30
                )
                
                if token_response.status_code == 200:
                    token_data = token_response.json()
                    if token_data['status'] == '1':
                        for transfer in token_data['result'][:50]:
                            token_info = {
                                'token': transfer.get('tokenName'),
                                'symbol': transfer.get('tokenSymbol'),
                                'from': transfer.get('from'),
                                'to': transfer.get('to'),
                                'value': float(transfer.get('value', 0)) / (10 ** int(transfer.get('tokenDecimal', 18))),
                                'timestamp': datetime.fromtimestamp(int(transfer.get('timeStamp', 0))).isoformat(),
                                'hash': transfer.get('hash')
                            }
                            analysis['token_transfers'].append(token_info)
                            
                # Calculate total volume
                total_in = sum(float(tx['value']) for tx in analysis['transactions'] if tx['to'] == address.lower())
                total_out = sum(float(tx['value']) for tx in analysis['transactions'] if tx['from'] == address.lower())
                analysis['total_volume'] = total_in + total_out
                
                # Check if address is a smart contract
                contract_response = requests.get(
                    'https://api.etherscan.io/api',
                    params={
                        'module': 'contract',
                        'action': 'getabi',
                        'address': address,
                        'apikey': self.etherscan_api
                    },
                    timeout=30
                )
                
                if contract_response.status_code == 200:
                    contract_data = contract_response.json()
                    if contract_data['status'] == '1':
                        analysis['address_info']['is_contract'] = True
                        analysis['address_info']['contract_abi'] = contract_data['result']
                        
        except Exception as e:
            logger.error(f"Ethereum analysis failed: {str(e)}")
            
        return analysis
        
    def _analyze_monero_address(self, address: str) -> Dict:
        """Analyze Monero address (limited due to privacy features)"""
        analysis = {
            'address_info': {
                'address': address,
                'type': 'monero',
                'privacy_coin': True,
                'note': 'Monero transactions are private by design. Limited analysis available.'
            },
            'transactions': [],
            'risk_assessment': {
                'privacy_coin_risk': 'HIGH',
                'traceability': 'LOW'
            }
        }
        
        # Monero analysis is limited due to privacy features
        # Check if address appears in known services or exchanges
        
        return analysis
        
    def _process_bitcoin_transaction(self, tx: Dict, address: str) -> Dict:
        """Process individual Bitcoin transaction"""
        tx_info = {
            'hash': tx.get('hash'),
            'time': datetime.fromtimestamp(tx.get('time', 0)).isoformat() if tx.get('time') else None,
            'block_height': tx.get('block_height'),
            'fee': tx.get('fee', 0) / 100000000,
            'inputs': [],
            'outputs': [],
            'direction': None,
            'amount': 0,
            'counterparty': []
        }
        
        # Process inputs
        for inp in tx.get('inputs', []):
            if inp.get('prev_out'):
                input_addr = inp['prev_out'].get('addr')
                input_value = inp['prev_out'].get('value', 0) / 100000000
                
                tx_info['inputs'].append({
                    'address': input_addr,
                    'value': input_value
                })
                
                if input_addr == address:
                    tx_info['direction'] = 'outgoing'
                    tx_info['amount'] += input_value
                    
        # Process outputs
        for out in tx.get('out', []):
            output_addr = out.get('addr')
            output_value = out.get('value', 0) / 100000000
            
            tx_info['outputs'].append({
                'address': output_addr,
                'value': output_value
            })
            
            if output_addr == address:
                if tx_info['direction'] != 'outgoing':
                    tx_info['direction'] = 'incoming'
                tx_info['amount'] += output_value
                
        # Identify counterparties
        if tx_info['direction'] == 'incoming':
            tx_info['counterparty'] = [inp['address'] for inp in tx_info['inputs'] if inp['address'] != address]
        elif tx_info['direction'] == 'outgoing':
            tx_info['counterparty'] = [out['address'] for out in tx_info['outputs'] if out['address'] != address]
            
        return tx_info
        
    def _process_ethereum_transaction(self, tx: Dict, address: str) -> Dict:
        """Process individual Ethereum transaction"""
        tx_info = {
            'hash': tx.get('hash'),
            'time': datetime.fromtimestamp(int(tx.get('timeStamp', 0))).isoformat() if tx.get('timeStamp') else None,
            'block_number': tx.get('blockNumber'),
            'from': tx.get('from', '').lower(),
            'to': tx.get('to', '').lower(),
            'value': float(tx.get('value', 0)) / 10**18,  # Convert from Wei to ETH
            'gas_used': int(tx.get('gasUsed', 0)),
            'gas_price': float(tx.get('gasPrice', 0)) / 10**9,  # Convert to Gwei
            'is_error': tx.get('isError') == '1',
            'direction': 'incoming' if tx.get('to', '').lower() == address.lower() else 'outgoing',
            'method': tx.get('functionName', '').split('(')[0] if tx.get('functionName') else None
        }
        
        return tx_info
        
    def _perform_wallet_clustering(self, address: str, transactions: List[Dict]) -> List[str]:
        """Cluster related wallet addresses using common inputs heuristic"""
        cluster = {address}
        
        # Common input heuristic for Bitcoin
        for tx in transactions:
            if tx.get('direction') == 'outgoing':
                # All inputs in same transaction likely controlled by same entity
                for inp in tx.get('inputs', []):
                    if inp.get('address'):
                        cluster.add(inp['address'])
                        
        # Change address heuristic
        for tx in transactions:
            if tx.get('direction') == 'outgoing' and len(tx.get('outputs', [])) == 2:
                outputs = tx['outputs']
                # One output likely change if it's a round number
                for out in outputs:
                    if out['value'] % 0.001 != 0:  # Not a round number, likely change
                        cluster.add(out['address'])
                        
        return list(cluster)
        
    def _detect_mixer_usage(self, transactions: List[Dict]) -> List[Dict]:
        """Detect transactions involving known mixers"""
        mixer_txs = []
        
        for tx in transactions:
            for pattern_name, patterns in self.MIXER_PATTERNS.items():
                # Check if any address matches mixer patterns
                addresses_to_check = []
                
                addresses_to_check.extend([inp.get('address', '') for inp in tx.get('inputs', [])])
                addresses_to_check.extend([out.get('address', '') for out in tx.get('outputs', [])])
                addresses_to_check.extend(tx.get('counterparty', []))
                
                for addr in addresses_to_check:
                    if addr and any(pattern in addr for pattern in patterns):
                        mixer_txs.append({
                            'transaction': tx['hash'],
                            'mixer': pattern_name,
                            'address': addr,
                            'timestamp': tx.get('time'),
                            'amount': tx.get('amount', 0),
                            'risk': 'HIGH'
                        })
                        break
                        
        # Detect CoinJoin transactions (multiple inputs, equal outputs)
        for tx in transactions:
            outputs = tx.get('outputs', [])
            if len(outputs) > 2:
                values = [out['value'] for out in outputs]
                # Check if many outputs have same value (CoinJoin indicator)
                from collections import Counter
                value_counts = Counter(values)
                max_count = max(value_counts.values()) if value_counts else 0
                
                if max_count > len(outputs) / 2:
                    mixer_txs.append({
                        'transaction': tx['hash'],
                        'mixer': 'possible_coinjoin',
                        'timestamp': tx.get('time'),
                        'equal_outputs': max_count,
                        'risk': 'MEDIUM'
                    })
                    
        return mixer_txs
        
    def _identify_exchange_interactions(self, transactions: List[Dict]) -> List[Dict]:
        """Identify interactions with known exchange addresses"""
        exchange_txs = []
        
        for tx in transactions:
            addresses_to_check = []
            
            # Collect all addresses from transaction
            addresses_to_check.extend([inp.get('address', '') for inp in tx.get('inputs', [])])
            addresses_to_check.extend([out.get('address', '') for out in tx.get('outputs', [])])
            addresses_to_check.extend(tx.get('counterparty', []))
            addresses_to_check.append(tx.get('from', ''))
            addresses_to_check.append(tx.get('to', ''))
            
            for exchange_name, exchange_addresses in self.KNOWN_EXCHANGES.items():
                for addr in addresses_to_check:
                    if addr in exchange_addresses:
                        exchange_txs.append({
                            'transaction': tx.get('hash'),
                            'exchange': exchange_name,
                            'exchange_address': addr,
                            'direction': 'deposit' if tx.get('direction') == 'outgoing' else 'withdrawal',
                            'amount': tx.get('amount', 0) or tx.get('value', 0),
                            'timestamp': tx.get('time'),
                            'risk': 'LOW'  # KYC exchanges are lower risk
                        })
                        break
                        
        return exchange_txs
        
    def _analyze_money_flow(self, transactions: List[Dict]) -> Dict:
        """Analyze money flow patterns"""
        flow = {
            'total_inflow': 0,
            'total_outflow': 0,
            'net_flow': 0,
            'largest_inflow': {'amount': 0, 'transaction': None},
            'largest_outflow': {'amount': 0, 'transaction': None},
            'flow_pattern': None,
            'velocity': 0,
            'time_analysis': {}
        }
        
        # Calculate flows
        for tx in transactions:
            amount = tx.get('amount', 0) or tx.get('value', 0)
            
            if tx.get('direction') == 'incoming':
                flow['total_inflow'] += amount
                if amount > flow['largest_inflow']['amount']:
                    flow['largest_inflow'] = {'amount': amount, 'transaction': tx.get('hash')}
                    
            elif tx.get('direction') == 'outgoing':
                flow['total_outflow'] += amount
                if amount > flow['largest_outflow']['amount']:
                    flow['largest_outflow'] = {'amount': amount, 'transaction': tx.get('hash')}
                    
        flow['net_flow'] = flow['total_inflow'] - flow['total_outflow']
        
        # Analyze patterns
        if flow['total_inflow'] > 0 and flow['total_outflow'] > 0:
            ratio = flow['total_outflow'] / flow['total_inflow']
            
            if ratio > 0.9 and ratio < 1.1:
                flow['flow_pattern'] = 'pass_through'  # Possible money laundering
            elif ratio < 0.1:
                flow['flow_pattern'] = 'accumulation'
            elif ratio > 10:
                flow['flow_pattern'] = 'distribution'
            else:
                flow['flow_pattern'] = 'mixed'
                
        # Calculate velocity (transactions per day)
        if transactions:
            timestamps = [tx.get('time') for tx in transactions if tx.get('time')]
            if len(timestamps) > 1:
                first_tx = min(timestamps)
                last_tx = max(timestamps)
                
                first_date = datetime.fromisoformat(first_tx)
                last_date = datetime.fromisoformat(last_tx)
                days_active = (last_date - first_date).days or 1
                
                flow['velocity'] = len(transactions) / days_active
                
                # Time-based analysis
                flow['time_analysis'] = {
                    'first_transaction': first_tx,
                    'last_transaction': last_tx,
                    'days_active': days_active,
                    'avg_transactions_per_day': flow['velocity']
                }
                
        return flow
        
    def _assess_risk(self, results: Dict) -> Dict:
        """Assess risk level of cryptocurrency address"""
        risk = {
            'risk_score': 0,
            'risk_level': 'LOW',
            'risk_factors': [],
            'recommendations': []
        }
        
        score = 0
        
        # Check mixer usage
        if results.get('mixer_usage'):
            score += 30
            risk['risk_factors'].append(f"Mixer usage detected: {len(results['mixer_usage'])} transactions")
            risk['recommendations'].append("Monitor for money laundering activity")
            
        # Check for high-risk patterns
        money_flow = results.get('money_flow', {})
        if money_flow.get('flow_pattern') == 'pass_through':
            score += 20
            risk['risk_factors'].append("Pass-through pattern detected (possible laundering)")
            
        # High velocity
        if money_flow.get('velocity', 0) > 10:
            score += 15
            risk['risk_factors'].append(f"High transaction velocity: {money_flow['velocity']:.2f} tx/day")
            
        # Criminal associations
        if results.get('criminal_associations'):
            score += 40
            risk['risk_factors'].append(f"Criminal associations found: {len(results['criminal_associations'])}")
            risk['recommendations'].append("Immediate investigation required")
            
        # Privacy coin
        if results.get('address_info', {}).get('privacy_coin'):
            score += 25
            risk['risk_factors'].append("Privacy coin (Monero) - limited traceability")
            
        # Large volume
        if results.get('total_volume', 0) > 1000:  # Over 1000 BTC/ETH
            score += 10
            risk['risk_factors'].append(f"High volume: {results['total_volume']:.2f}")
            
        # No KYC exchange interaction
        exchange_interactions = results.get('exchange_interactions', [])
        if not exchange_interactions and results.get('transactions'):
            score += 10
            risk['risk_factors'].append("No legitimate exchange interactions detected")
            
        # Determine risk level
        risk['risk_score'] = min(score, 100)
        
        if score >= 70:
            risk['risk_level'] = 'CRITICAL'
            risk['recommendations'].append("Freeze associated accounts")
            risk['recommendations'].append("File suspicious activity report")
        elif score >= 50:
            risk['risk_level'] = 'HIGH'
            risk['recommendations'].append("Enhanced due diligence required")
        elif score >= 30:
            risk['risk_level'] = 'MEDIUM'
            risk['recommendations'].append("Continue monitoring")
        else:
            risk['risk_level'] = 'LOW'
            risk['recommendations'].append("Standard monitoring")
            
        return risk
        
    def _check_criminal_associations(self, address: str, results: Dict) -> List[Dict]:
        """Check for associations with known criminal wallets"""
        criminal_links = []
        
        # Known criminal addresses (would be from threat intelligence feeds)
        CRIMINAL_ADDRESSES = {
            'silk_road': ['1F1tAaz5x1HUXrCNLbtMDqcw6o5GNn4xqX'],
            'alphabay': ['1Em4e3x8Zqfyqfq8DnMRfLFPPJRh3U1wzJ'],
            'wannacry': ['12t9YDPgwueZ9NyMgw519p7AA8isjr6SMw', '13AM4VW2dhxYgXeQepoHkHSQuy6NgaEb94'],
            'bitfinex_hack': ['1CGA1iszzkU5p7bVGU6qW8jVaUXVNXXUVW'],
            # Add more from threat intelligence
        }
        
        # Check direct interactions
        all_addresses = set()
        for tx in results.get('transactions', []):
            all_addresses.update(tx.get('counterparty', []))
            for inp in tx.get('inputs', []):
                if inp.get('address'):
                    all_addresses.add(inp['address'])
            for out in tx.get('outputs', []):
                if out.get('address'):
                    all_addresses.add(out['address'])
                    
        # Check for criminal addresses
        for crime_type, crime_addresses in CRIMINAL_ADDRESSES.items():
            for crime_addr in crime_addresses:
                if crime_addr in all_addresses:
                    criminal_links.append({
                        'type': crime_type,
                        'criminal_address': crime_addr,
                        'connection': 'direct_transaction',
                        'risk': 'CRITICAL'
                    })
                    
        # Check wallet cluster for criminal associations
        for cluster_addr in results.get('wallet_cluster', []):
            for crime_type, crime_addresses in CRIMINAL_ADDRESSES.items():
                if cluster_addr in crime_addresses:
                    criminal_links.append({
                        'type': crime_type,
                        'criminal_address': cluster_addr,
                        'connection': 'same_wallet_cluster',
                        'risk': 'HIGH'
                    })
                    
        return criminal_links
        
    def _get_chainalysis_data(self, address: str) -> Dict:
        """Get additional data from Chainalysis API"""
        # Placeholder for Chainalysis integration
        return {}
        
    def _cache_results(self, address: str, results: Dict):
        """Cache cryptocurrency analysis results"""
        # Implementation would cache to Redis or PostgreSQL
        pass
        
    def _store_in_neo4j(self, address: str, results: Dict):
        """Store crypto analysis in Neo4j for graph visualization"""
        try:
            if not self.neo4j_driver:
                self.neo4j_driver = GraphDatabase.driver(
                    os.getenv('NEO4J_URI'),
                    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
                )
                
            with self.neo4j_driver.session() as session:
                # Create address node
                session.run("""
                    MERGE (a:CryptoAddress {address: $address})
                    SET a.type = $type,
                        a.balance = $balance,
                        a.risk_score = $risk_score,
                        a.risk_level = $risk_level,
                        a.last_analyzed = datetime()
                """, address=address,
                     type=results['address_info'].get('type', 'unknown'),
                     balance=results['address_info'].get('balance', 0),
                     risk_score=results['risk_assessment'].get('risk_score', 0),
                     risk_level=results['risk_assessment'].get('risk_level', 'UNKNOWN'))
                
                # Create transaction nodes and relationships
                for tx in results.get('transactions', [])[:50]:  # Limit for performance
                    session.run("""
                        MERGE (t:Transaction {hash: $hash})
                        SET t.amount = $amount,
                            t.timestamp = $timestamp,
                            t.direction = $direction
                        MERGE (a:CryptoAddress {address: $address})
                        MERGE (a)-[r:HAS_TRANSACTION]->(t)
                    """, hash=tx['hash'],
                         amount=tx.get('amount', 0) or tx.get('value', 0),
                         timestamp=tx.get('time'),
                         direction=tx.get('direction'),
                         address=address)
                    
                    # Link counterparties
                    for counter in tx.get('counterparty', []):
                        session.run("""
                            MERGE (c:CryptoAddress {address: $counter})
                            MERGE (t:Transaction {hash: $hash})
                            MERGE (c)-[:INVOLVED_IN]->(t)
                        """, counter=counter, hash=tx['hash'])
                        
                # Store mixer usage
                for mixer_tx in results.get('mixer_usage', []):
                    session.run("""
                        MERGE (m:Mixer {name: $mixer})
                        MERGE (t:Transaction {hash: $hash})
                        MERGE (t)-[:USED_MIXER]->(m)
                        SET t.mixer_risk = $risk
                    """, mixer=mixer_tx['mixer'],
                         hash=mixer_tx['transaction'],
                         risk=mixer_tx['risk'])
                         
                # Store exchange interactions
                for ex_tx in results.get('exchange_interactions', []):
                    session.run("""
                        MERGE (e:Exchange {name: $exchange})
                        MERGE (t:Transaction {hash: $hash})
                        MERGE (t)-[:EXCHANGE_INTERACTION {direction: $direction}]->(e)
                    """, exchange=ex_tx['exchange'],
                         hash=ex_tx['transaction'],
                         direction=ex_tx['direction'])
                         
                # Store criminal associations
                for criminal in results.get('criminal_associations', []):
                    session.run("""
                        MERGE (c:CriminalAddress {address: $criminal_addr})
                        SET c.type = $crime_type
                        MERGE (a:CryptoAddress {address: $address})
                        MERGE (a)-[r:ASSOCIATED_WITH {connection: $connection, risk: $risk}]->(c)
                    """, criminal_addr=criminal['criminal_address'],
                         crime_type=criminal['type'],
                         address=address,
                         connection=criminal['connection'],
                         risk=criminal['risk'])
                         
        except Exception as e:
            logger.error(f"Neo4j storage failed: {str(e)}")
            
        finally:
            if self.neo4j_driver:
                self.neo4j_driver.close()
                
    @classmethod
    def _monkeypatch(cls):
        """Register analyzer with IntelOwl"""
        patches = [
            {
                'model': 'analyzers_manager.AnalyzerConfig',
                'name': 'CryptoTrackerAnalyzer',
                'description': 'Track cryptocurrency transactions and identify wallet clusters',
                'python_module': 'custom_analyzers.crypto_analyzer.CryptoTrackerAnalyzer',
                'disabled': False,
                'type': 'observable',
                'docker_based': False,
                'maximum_tlp': 'RED',
                'observable_supported': ['generic', 'bitcoin_address', 'ethereum_address'],
                'supported_filetypes': [],
                'run_hash': False,
                'run_hash_type': '',
                'not_supported_filetypes': [],
                'parameters': {
                    'depth': {
                        'type': 'int',
                        'description': 'Transaction depth to analyze',
                        'default': 3
                    },
                    'cluster_analysis': {
                        'type': 'bool',
                        'description': 'Perform wallet clustering',
                        'default': True
                    },
                    'mixer_detection': {
                        'type': 'bool',
                        'description': 'Detect mixer usage',
                        'default': True
                    },
                    'risk_scoring': {
                        'type': 'bool',
                        'description': 'Calculate risk scores',
                        'default': True
                    },
                    'time_window': {
                        'type': 'int',
                        'description': 'Days to analyze',
                        'default': 30
                    }
                }
            }
        ]
        return patches
