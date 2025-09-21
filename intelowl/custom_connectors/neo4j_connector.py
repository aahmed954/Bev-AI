"""Neo4j Connector for IntelOwl
Exports IntelOwl analysis results to Neo4j graph database
"""

import os
import json
from typing import Dict, List, Any
from datetime import datetime
from neo4j import GraphDatabase
from api_app.connectors_manager.connectors import Connector
import logging

logger = logging.getLogger(__name__)


class Neo4jConnector(Connector):
    """Export analysis results to Neo4j for graph visualization"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
        self.neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        self.driver = None
        
    def set_params(self, params):
        """Set connector parameters"""
        self.create_relationships = params.get('create_relationships', True)
        self.merge_duplicates = params.get('merge_duplicates', True)
        self.export_raw_data = params.get('export_raw_data', False)
        
    def run(self, analyzers_results: Dict, job_id: int) -> Dict:
        """Export analysis results to Neo4j"""
        results = {
            'exported_nodes': 0,
            'exported_relationships': 0,
            'errors': []
        }
        
        try:
            # Connect to Neo4j
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            
            with self.driver.session() as session:
                # Process each analyzer's results
                for analyzer_name, analyzer_data in analyzers_results.items():
                    try:
                        if 'BreachDatabase' in analyzer_name:
                            self._export_breach_data(session, analyzer_data)
                            
                        elif 'DarknetMarket' in analyzer_name:
                            self._export_darknet_data(session, analyzer_data)
                            
                        elif 'CryptoTracker' in analyzer_name:
                            self._export_crypto_data(session, analyzer_data)
                            
                        elif 'SocialMedia' in analyzer_name:
                            self._export_social_data(session, analyzer_data)
                            
                        else:
                            # Generic export for other analyzers
                            self._export_generic_data(session, analyzer_name, analyzer_data)
                            
                        results['exported_nodes'] += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to export {analyzer_name}: {str(e)}")
                        results['errors'].append(f"{analyzer_name}: {str(e)}")
                        
                # Create job node
                session.run("""
                    CREATE (j:IntelOwlJob {
                        job_id: $job_id,
                        timestamp: datetime(),
                        analyzers: $analyzers
                    })
                """, job_id=job_id, analyzers=list(analyzers_results.keys()))
                
        except Exception as e:
            logger.error(f"Neo4j export failed: {str(e)}")
            results['errors'].append(f"Connection error: {str(e)}")
            
        finally:
            if self.driver:
                self.driver.close()
                
        return results
        
    def _export_breach_data(self, session, data: Dict):
        """Export breach database results"""
        if data.get('breaches'):
            for breach in data['breaches']:
                session.run("""
                    MERGE (b:Breach {database: $database})
                    SET b.source = $source,
                        b.date = $date,
                        b.last_updated = datetime()
                    
                    MERGE (i:Identity {value: $identity})
                    SET i.type = $identity_type,
                        i.risk_score = $risk_score
                    
                    MERGE (i)-[r:COMPROMISED_IN]->(b)
                    SET r.password_leaked = $password_leaked,
                        r.data_exposed = $data_exposed
                """, 
                database=breach.get('database'),
                source=breach.get('source'),
                date=breach.get('date'),
                identity=breach.get('email') or breach.get('username'),
                identity_type='email' if breach.get('email') else 'username',
                risk_score=data.get('risk_score', 0),
                password_leaked=bool(breach.get('password')),
                data_exposed=json.dumps(breach))
                
    def _export_darknet_data(self, session, data: Dict):
        """Export darknet market results"""
        if data.get('vendor_profiles'):
            for vendor in data['vendor_profiles']:
                session.run("""
                    MERGE (v:DarknetVendor {name: $name})
                    SET v.market = $market,
                        v.reputation = $reputation,
                        v.products_count = $products_count,
                        v.last_seen = datetime()
                    
                    MERGE (m:DarknetMarket {name: $market})
                    MERGE (v)-[:OPERATES_ON]->(m)
                """,
                name=vendor['vendor_name'],
                market=vendor['market'],
                reputation=vendor.get('reputation'),
                products_count=vendor.get('products_count', 0))
                
        if data.get('crypto_addresses'):
            for addr in data['crypto_addresses']:
                session.run("""
                    MERGE (a:CryptoAddress {address: $address})
                    SET a.currency = $currency,
                        a.source = 'darknet'
                """,
                address=addr['address'],
                currency=addr['currency'])
                
    def _export_crypto_data(self, session, data: Dict):
        """Export cryptocurrency analysis results"""
        address_info = data.get('address_info', {})
        if address_info:
            session.run("""
                MERGE (a:CryptoAddress {address: $address})
                SET a.type = $type,
                    a.balance = $balance,
                    a.total_received = $total_received,
                    a.total_sent = $total_sent,
                    a.risk_score = $risk_score,
                    a.risk_level = $risk_level,
                    a.last_analyzed = datetime()
            """,
            address=address_info.get('address'),
            type=address_info.get('type'),
            balance=address_info.get('balance', 0),
            total_received=address_info.get('total_received', 0),
            total_sent=address_info.get('total_sent', 0),
            risk_score=data.get('risk_assessment', {}).get('risk_score', 0),
            risk_level=data.get('risk_assessment', {}).get('risk_level'))
            
        # Export transactions
        for tx in data.get('transactions', [])[:100]:
            session.run("""
                MERGE (t:Transaction {hash: $hash})
                SET t.amount = $amount,
                    t.timestamp = $timestamp,
                    t.block_height = $block_height
            """,
            hash=tx['hash'],
            amount=tx.get('amount', 0),
            timestamp=tx.get('time'),
            block_height=tx.get('block_height'))
            
    def _export_social_data(self, session, data: Dict):
        """Export social media analysis results"""
        for platform, profile in data.get('profiles', {}).items():
            if profile:
                session.run("""
                    MERGE (p:SocialProfile {username: $username, platform: $platform})
                    SET p.full_name = $full_name,
                        p.bio = $bio,
                        p.followers = $followers,
                        p.following = $following,
                        p.verified = $verified,
                        p.last_analyzed = datetime()
                """,
                username=profile.get('username'),
                platform=platform,
                full_name=profile.get('full_name'),
                bio=profile.get('bio'),
                followers=profile.get('followers', 0),
                following=profile.get('following', 0),
                verified=profile.get('verified', False))
                
    def _export_generic_data(self, session, analyzer_name: str, data: Dict):
        """Export generic analyzer data"""
        session.run("""
            CREATE (n:AnalysisResult {
                analyzer: $analyzer,
                timestamp: datetime(),
                data: $data
            })
        """,
        analyzer=analyzer_name,
        data=json.dumps(data)[:10000])  # Limit data size
        
    @classmethod
    def _monkeypatch(cls):
        """Register connector with IntelOwl"""
        patches = [
            {
                'model': 'connectors_manager.ConnectorConfig',
                'name': 'Neo4jConnector',
                'description': 'Export analysis results to Neo4j graph database',
                'python_module': 'custom_connectors.neo4j_connector.Neo4jConnector',
                'disabled': False,
                'maximum_tlp': 'RED',
                'parameters': {
                    'create_relationships': {
                        'type': 'bool',
                        'description': 'Create relationships between entities',
                        'default': True
                    },
                    'merge_duplicates': {
                        'type': 'bool',
                        'description': 'Merge duplicate entities',
                        'default': True
                    },
                    'export_raw_data': {
                        'type': 'bool',
                        'description': 'Export raw analysis data',
                        'default': False
                    }
                }
            }
        ]
        return patches
