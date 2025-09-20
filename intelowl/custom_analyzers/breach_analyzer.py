"""Breach Database Analyzer for IntelOwl
Connects to Dehashed, Snusbase, WeLeakInfo APIs
Searches through leaked credentials and PII
"""

import os
import json
import hashlib
import requests
from typing import Dict, List, Any
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from celery import shared_task
from api_app.analyzers_manager import classes
from api_app.analyzers_manager.observable_analyzers import ObservableAnalyzer
from api_app.models import Job
import logging

logger = logging.getLogger(__name__)


class BreachDatabaseAnalyzer(ObservableAnalyzer):
    """Search breach databases for compromised credentials"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dehashed_api_key = os.getenv('DEHASHED_API_KEY')
        self.dehashed_email = os.getenv('DEHASHED_EMAIL')
        self.snusbase_api_key = os.getenv('SNUSBASE_API_KEY')
        self.weleakinfo_api_key = os.getenv('WELEAKINFO_API_KEY')
        
        # PostgreSQL connection for caching results
        self.db_config = {
            'host': 'postgres',
            'database': 'breach_data',
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD')
        }
        
    def set_params(self, params):
        """Set analyzer parameters"""
        self.search_type = params.get('search_type', 'email')  # email, username, password, hash, ip, domain
        self.include_passwords = params.get('include_passwords', True)
        self.check_haveibeenpwned = params.get('check_haveibeenpwned', True)
        self.deep_search = params.get('deep_search', True)
        self.correlate_accounts = params.get('correlate_accounts', True)
        
    def run(self):
        """Execute breach database search"""
        observable = self.observable_name
        observable_type = self.observable_classification
        results = {
            'breaches': [],
            'leaked_passwords': [],
            'associated_accounts': [],
            'risk_score': 0,
            'compromised_count': 0,
            'first_breach_date': None,
            'latest_breach_date': None,
            'sensitive_data_exposed': []
        }
        
        try:
            # Check cache first
            cached = self._check_cache(observable)
            if cached and not self.deep_search:
                logger.info(f"Using cached breach data for {observable}")
                return cached
                
            # Search Dehashed
            if self.dehashed_api_key:
                dehashed_results = self._search_dehashed(observable, observable_type)
                results['breaches'].extend(dehashed_results.get('breaches', []))
                results['leaked_passwords'].extend(dehashed_results.get('passwords', []))
                
            # Search Snusbase  
            if self.snusbase_api_key:
                snusbase_results = self._search_snusbase(observable, observable_type)
                results['breaches'].extend(snusbase_results.get('breaches', []))
                results['associated_accounts'].extend(snusbase_results.get('accounts', []))
                
            # Search WeLeakInfo
            if self.weleakinfo_api_key:
                weleakinfo_results = self._search_weleakinfo(observable, observable_type)
                results['breaches'].extend(weleakinfo_results.get('breaches', []))
                
            # Check HaveIBeenPwned
            if self.check_haveibeenpwned and observable_type == 'email':
                hibp_results = self._check_haveibeenpwned(observable)
                results['breaches'].extend(hibp_results)
                
            # Deep correlation search
            if self.deep_search and self.correlate_accounts:
                correlated = self._correlate_accounts(observable, results)
                results['associated_accounts'].extend(correlated)
                
            # Calculate risk score
            results['risk_score'] = self._calculate_risk_score(results)
            
            # Extract breach timeline
            if results['breaches']:
                dates = [b.get('date') for b in results['breaches'] if b.get('date')]
                if dates:
                    results['first_breach_date'] = min(dates)
                    results['latest_breach_date'] = max(dates)
                    
            results['compromised_count'] = len(results['breaches'])
            
            # Identify sensitive data types exposed
            results['sensitive_data_exposed'] = self._identify_sensitive_data(results)
            
            # Cache results
            self._cache_results(observable, results)
            
            # Store in graph database for correlation
            self._store_in_neo4j(observable, results)
            
        except Exception as e:
            logger.error(f"Breach database search failed: {str(e)}")
            return {'error': str(e)}
            
        return results
        
    def _search_dehashed(self, query: str, query_type: str) -> Dict:
        """Search Dehashed API for breaches"""
        results = {'breaches': [], 'passwords': []}
        
        try:
            headers = {
                'Accept': 'application/json',
            }
            
            auth = (self.dehashed_email, self.dehashed_api_key)
            
            # Build query based on type
            search_params = {
                'email': f'email:{query}',
                'username': f'username:{query}',
                'ip': f'ip_address:{query}',
                'domain': f'email:*@{query}',
                'hash': f'hashed_password:{query}',
                'password': f'password:{query}'
            }
            
            search_query = search_params.get(query_type, f'email:{query}')
            
            response = requests.get(
                'https://api.dehashed.com/search',
                params={'query': search_query, 'size': 10000},
                headers=headers,
                auth=auth,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for entry in data.get('entries', []):
                    breach = {
                        'source': 'Dehashed',
                        'database': entry.get('database_name', 'Unknown'),
                        'email': entry.get('email'),
                        'username': entry.get('username'),
                        'password': entry.get('password') if self.include_passwords else None,
                        'hashed_password': entry.get('hashed_password'),
                        'name': entry.get('name'),
                        'ip_address': entry.get('ip_address'),
                        'phone': entry.get('phone'),
                        'address': entry.get('address'),
                        'date': entry.get('obtained_from')
                    }
                    
                    results['breaches'].append(breach)
                    
                    if entry.get('password') and self.include_passwords:
                        results['passwords'].append({
                            'plaintext': entry.get('password'),
                            'database': entry.get('database_name'),
                            'hash': hashlib.sha256(entry.get('password', '').encode()).hexdigest()
                        })
                        
        except Exception as e:
            logger.error(f"Dehashed search failed: {str(e)}")
            
        return results
        
    def _search_snusbase(self, query: str, query_type: str) -> Dict:
        """Search Snusbase API for breaches"""
        results = {'breaches': [], 'accounts': []}
        
        try:
            headers = {
                'Auth': self.snusbase_api_key,
                'Content-Type': 'application/json'
            }
            
            # Snusbase search types
            search_types = {
                'email': 'email',
                'username': 'username',
                'ip': 'lastip',
                'hash': 'hash',
                'password': 'password',
                'name': 'name'
            }
            
            search_type = search_types.get(query_type, 'email')
            
            payload = {
                'type': search_type,
                'term': query
            }
            
            response = requests.post(
                'https://api.snusbase.com/search',
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for db_name, entries in data.get('result', {}).items():
                    for entry in entries:
                        breach = {
                            'source': 'Snusbase',
                            'database': db_name,
                            'email': entry.get('email'),
                            'username': entry.get('username'),
                            'password': entry.get('password') if self.include_passwords else None,
                            'hash': entry.get('hash'),
                            'salt': entry.get('salt'),
                            'name': entry.get('name'),
                            'lastip': entry.get('lastip'),
                            'created': entry.get('created')
                        }
                        
                        results['breaches'].append(breach)
                        
                        # Find associated accounts
                        if entry.get('username') and entry.get('username') != query:
                            results['accounts'].append({
                                'username': entry.get('username'),
                                'email': entry.get('email'),
                                'database': db_name
                            })
                            
        except Exception as e:
            logger.error(f"Snusbase search failed: {str(e)}")
            
        return results
        
    def _search_weleakinfo(self, query: str, query_type: str) -> Dict:
        """Search WeLeakInfo API"""
        results = {'breaches': []}
        
        try:
            headers = {
                'Authorization': f'Bearer {self.weleakinfo_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'query': query,
                'type': query_type,
                'wildcard': False
            }
            
            response = requests.post(
                'https://api.weleakinfo.to/api/search',
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for result in data.get('results', []):
                    breach = {
                        'source': 'WeLeakInfo',
                        'database': result.get('source'),
                        'email': result.get('email'),
                        'username': result.get('username'),
                        'password': result.get('password') if self.include_passwords else None,
                        'hash': result.get('password_hash'),
                        'phone': result.get('phone'),
                        'name': result.get('full_name')
                    }
                    
                    results['breaches'].append(breach)
                    
        except Exception as e:
            logger.error(f"WeLeakInfo search failed: {str(e)}")
            
        return results
        
    def _check_haveibeenpwned(self, email: str) -> List[Dict]:
        """Check HaveIBeenPwned for breaches"""
        breaches = []
        
        try:
            # Note: HIBP requires API key for full results
            response = requests.get(
                f'https://haveibeenpwned.com/api/v3/breachedaccount/{email}',
                headers={'User-Agent': 'Bev-OSINT-Framework'},
                timeout=10
            )
            
            if response.status_code == 200:
                for breach in response.json():
                    breaches.append({
                        'source': 'HaveIBeenPwned',
                        'database': breach.get('Name'),
                        'date': breach.get('BreachDate'),
                        'description': breach.get('Description'),
                        'data_classes': breach.get('DataClasses', []),
                        'verified': breach.get('IsVerified'),
                        'compromised_accounts': breach.get('PwnCount')
                    })
                    
        except Exception as e:
            logger.error(f"HIBP check failed: {str(e)}")
            
        return breaches
        
    def _correlate_accounts(self, initial_query: str, current_results: Dict) -> List[Dict]:
        """Deep correlation to find related accounts"""
        correlated = []
        searched = {initial_query}
        to_search = set()
        
        # Extract usernames and emails from current results
        for breach in current_results.get('breaches', []):
            if breach.get('username') and breach['username'] not in searched:
                to_search.add(breach['username'])
            if breach.get('email') and breach['email'] not in searched:
                to_search.add(breach['email'])
                
        # Search for correlated accounts (limit depth to avoid infinite loops)
        max_depth = 3
        current_depth = 0
        
        while to_search and current_depth < max_depth:
            current_depth += 1
            next_search = set()
            
            for item in to_search:
                if item in searched:
                    continue
                    
                searched.add(item)
                
                # Determine search type
                search_type = 'email' if '@' in item else 'username'
                
                # Search all sources
                if self.dehashed_api_key:
                    dh_results = self._search_dehashed(item, search_type)
                    for breach in dh_results.get('breaches', []):
                        if breach.get('username') and breach['username'] not in searched:
                            next_search.add(breach['username'])
                            correlated.append({
                                'type': 'username',
                                'value': breach['username'],
                                'source': item,
                                'database': breach.get('database'),
                                'depth': current_depth
                            })
                            
                        if breach.get('email') and breach['email'] not in searched:
                            next_search.add(breach['email'])
                            correlated.append({
                                'type': 'email',
                                'value': breach['email'],
                                'source': item,
                                'database': breach.get('database'),
                                'depth': current_depth
                            })
                            
            to_search = next_search
            
        return correlated
        
    def _calculate_risk_score(self, results: Dict) -> int:
        """Calculate risk score based on breach data"""
        score = 0
        
        # Base score on number of breaches
        breach_count = len(results.get('breaches', []))
        score += min(breach_count * 10, 50)
        
        # Add score for exposed passwords
        if results.get('leaked_passwords'):
            score += 20
            
        # Add score for sensitive data types
        sensitive_types = results.get('sensitive_data_exposed', [])
        if 'SSN' in sensitive_types or 'credit_card' in sensitive_types:
            score += 30
        elif 'password' in sensitive_types:
            score += 15
            
        # Recent breaches are higher risk
        latest_breach = results.get('latest_breach_date')
        if latest_breach:
            try:
                breach_date = datetime.fromisoformat(latest_breach)
                days_ago = (datetime.now() - breach_date).days
                if days_ago < 90:
                    score += 20
                elif days_ago < 365:
                    score += 10
            except:
                pass
                
        return min(score, 100)
        
    def _identify_sensitive_data(self, results: Dict) -> List[str]:
        """Identify types of sensitive data exposed"""
        sensitive_types = set()
        
        for breach in results.get('breaches', []):
            if breach.get('password') or breach.get('hashed_password'):
                sensitive_types.add('password')
            if breach.get('phone'):
                sensitive_types.add('phone')
            if breach.get('address'):
                sensitive_types.add('address')
            if breach.get('ip_address'):
                sensitive_types.add('IP_address')
                
            # Check data classes from HIBP
            data_classes = breach.get('data_classes', [])
            for dc in data_classes:
                if 'Credit' in dc or 'Payment' in dc:
                    sensitive_types.add('credit_card')
                if 'SSN' in dc or 'Social' in dc:
                    sensitive_types.add('SSN')
                if 'Medical' in dc or 'Health' in dc:
                    sensitive_types.add('medical')
                    
        return list(sensitive_types)
        
    def _check_cache(self, query: str) -> Dict:
        """Check PostgreSQL cache for previous results"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT results FROM breach_cache 
                WHERE query = %s 
                AND created_at > NOW() - INTERVAL '24 hours'
                ORDER BY created_at DESC
                LIMIT 1
            """, (query,))
            
            row = cur.fetchone()
            if row:
                return json.loads(row['results'])
                
        except Exception as e:
            logger.error(f"Cache check failed: {str(e)}")
            
        return None
        
    def _cache_results(self, query: str, results: Dict):
        """Cache results in PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO breach_cache (query, results, created_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (query) 
                DO UPDATE SET results = EXCLUDED.results, created_at = NOW()
            """, (query, json.dumps(results)))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Cache storage failed: {str(e)}")
            
    def _store_in_neo4j(self, query: str, results: Dict):
        """Store breach relationships in Neo4j for graph analysis"""
        try:
            from neo4j import GraphDatabase
            
            driver = GraphDatabase.driver(
                os.getenv('NEO4J_URI'),
                auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
            )
            
            with driver.session() as session:
                # Create identity node
                session.run("""
                    MERGE (i:Identity {value: $query})
                    SET i.type = $type,
                        i.risk_score = $risk_score,
                        i.last_updated = datetime()
                """, query=query, 
                     type='email' if '@' in query else 'username',
                     risk_score=results.get('risk_score', 0))
                
                # Create breach nodes and relationships
                for breach in results.get('breaches', []):
                    session.run("""
                        MERGE (b:Breach {database: $database})
                        SET b.source = $source,
                            b.date = $date
                        MERGE (i:Identity {value: $query})
                        MERGE (i)-[r:COMPROMISED_IN]->(b)
                        SET r.password_leaked = $password_leaked,
                            r.data_exposed = $data_exposed
                    """, database=breach.get('database'),
                         source=breach.get('source'),
                         date=breach.get('date'),
                         query=query,
                         password_leaked=bool(breach.get('password')),
                         data_exposed=json.dumps(breach))
                    
                # Create correlated account relationships
                for account in results.get('associated_accounts', []):
                    session.run("""
                        MERGE (i1:Identity {value: $query})
                        MERGE (i2:Identity {value: $related})
                        SET i2.type = $type
                        MERGE (i1)-[r:RELATED_TO]->(i2)
                        SET r.database = $database,
                            r.correlation_depth = $depth
                    """, query=query,
                         related=account.get('value'),
                         type=account.get('type'),
                         database=account.get('database'),
                         depth=account.get('depth', 1))
                         
            driver.close()
            
        except Exception as e:
            logger.error(f"Neo4j storage failed: {str(e)}")
            
    @classmethod
    def _monkeypatch(cls):
        """Register analyzer with IntelOwl"""
        patches = [
            {
                'model': 'analyzers_manager.AnalyzerConfig',
                'name': 'BreachDatabaseAnalyzer',
                'description': 'Search breach databases for compromised credentials',
                'python_module': 'custom_analyzers.breach_analyzer.BreachDatabaseAnalyzer',
                'disabled': False,
                'type': 'observable',
                'docker_based': False,
                'maximum_tlp': 'RED',
                'observable_supported': ['generic', 'email', 'username', 'hash', 'ip', 'domain'],
                'supported_filetypes': [],
                'run_hash': False,
                'run_hash_type': '',
                'not_supported_filetypes': [],
                'parameters': {
                    'search_type': {
                        'type': 'str',
                        'description': 'Type of search: email, username, password, hash, ip, domain',
                        'default': 'email'
                    },
                    'include_passwords': {
                        'type': 'bool', 
                        'description': 'Include plaintext passwords in results',
                        'default': True
                    },
                    'deep_search': {
                        'type': 'bool',
                        'description': 'Perform deep correlation search',
                        'default': True
                    }
                }
            }
        ]
        return patches
