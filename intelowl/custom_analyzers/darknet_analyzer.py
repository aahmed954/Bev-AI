"""Darknet Market Analyzer for IntelOwl
Scrapes darknet markets via Tor for threat intelligence
Monitors vendor activity, product listings, and cryptocurrency flows
"""

import os
import json
import time
import hashlib
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import socks
import socket
from urllib.parse import urljoin, urlparse
from celery import shared_task
from api_app.analyzers_manager.observable_analyzers import ObservableAnalyzer
import logging
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class DarknetMarketAnalyzer(ObservableAnalyzer):
    """Analyze darknet markets and vendor networks via Tor"""
    
    # Known darknet markets (onion addresses)
    DARKNET_MARKETS = {
        'alphabay': {
            'url': 'http://alphabay522szl32u4ci5e3iokdsyth56ei7rwngr2wm7i5jo54j2eid.onion',
            'type': 'marketplace',
            'login_required': True
        },
        'whitehouse': {
            'url': 'http://whitehousemarketvwcjsvwig2czq4bvq5ckvfk2ey7e3crtsn3p7oid.onion',
            'type': 'marketplace',
            'login_required': True
        },
        'torrez': {
            'url': 'http://torrezmarket2kyl7kqonqyivlpu5ghat7kor5r6ud3xvbcgvneuakoad.onion',
            'type': 'marketplace',
            'login_required': True
        },
        'darkfox': {
            'url': 'http://darkfoxmarket24l5z3t5zz35flbbergqfhnaon2hpral5ajkn3tjiead.onion',
            'type': 'marketplace',
            'login_required': True
        },
        'archetyp': {
            'url': 'http://archetyp2nw4myzqdo7flqrkxgfmu2rkdsal2mxg2x7dg4hzrbhz4id.onion',
            'type': 'marketplace',
            'login_required': False
        },
        'dread': {
            'url': 'http://dreadytofatroptsdj6io7l3xptbet6onoyno2yv7jicoxknyazubrad.onion',
            'type': 'forum',
            'login_required': False
        },
        'darknet_stats': {
            'url': 'http://darkfailenbsdla5mal2mxn2uz66od5vtzd5qozslagrfzachha3f3id.onion',
            'type': 'directory',
            'login_required': False
        }
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tor_proxy = os.getenv('TOR_PROXY', 'socks5://tor:9050')
        self.tor_control_password = os.getenv('TOR_CONTROL_PASSWORD')
        self.neo4j_driver = None
        
    def set_params(self, params):
        """Set analyzer parameters"""
        self.search_type = params.get('search_type', 'vendor')  # vendor, product, cryptocurrency, forum_post
        self.deep_crawl = params.get('deep_crawl', False)
        self.extract_crypto = params.get('extract_crypto', True)
        self.monitor_prices = params.get('monitor_prices', True)
        self.track_vendors = params.get('track_vendors', True)
        self.max_pages = params.get('max_pages', 10)
        self.use_selenium = params.get('use_selenium', False)
        
    def run(self):
        """Execute darknet market analysis"""
        observable = self.observable_name
        observable_type = self.observable_classification
        
        results = {
            'darknet_mentions': [],
            'vendor_profiles': [],
            'product_listings': [],
            'crypto_addresses': [],
            'market_stats': {},
            'threat_indicators': [],
            'network_graph': {},
            'risk_assessment': {}
        }
        
        try:
            # Configure Tor proxy
            self._configure_tor_proxy()
            
            # Rotate Tor circuit for fresh identity
            self._rotate_tor_circuit()
            
            # Search based on observable type
            if observable_type == 'hash':
                # Search for file hashes (malware, documents)
                results['darknet_mentions'] = self._search_hash(observable)
                
            elif observable_type == 'bitcoin_address' or 'bitcoin' in observable.lower():
                # Track Bitcoin address across markets
                results['crypto_addresses'] = self._track_crypto_address(observable)
                
            elif observable_type == 'username' or self.search_type == 'vendor':
                # Search for vendor profiles
                results['vendor_profiles'] = self._search_vendor(observable)
                
            elif observable_type == 'generic' or self.search_type == 'product':
                # Search for products/services
                results['product_listings'] = self._search_products(observable)
                
            # Deep crawl if requested
            if self.deep_crawl:
                crawl_results = self._deep_crawl_markets(observable)
                results['darknet_mentions'].extend(crawl_results.get('mentions', []))
                results['network_graph'] = crawl_results.get('graph', {})
                
            # Extract cryptocurrency addresses
            if self.extract_crypto:
                crypto_addrs = self._extract_crypto_addresses(results)
                results['crypto_addresses'].extend(crypto_addrs)
                
            # Build vendor network graph
            if results['vendor_profiles']:
                results['network_graph'] = self._build_vendor_network(results['vendor_profiles'])
                
            # Calculate market statistics
            results['market_stats'] = self._calculate_market_stats(results)
            
            # Threat assessment
            results['risk_assessment'] = self._assess_threats(results)
            
            # Store in Neo4j for correlation
            self._store_in_neo4j(observable, results)
            
            # Cache results
            self._cache_results(observable, results)
            
        except Exception as e:
            logger.error(f"Darknet analysis failed: {str(e)}")
            return {'error': str(e)}
            
        return results
        
    def _configure_tor_proxy(self):
        """Configure requests to use Tor SOCKS5 proxy"""
        proxy_parts = urlparse(self.tor_proxy)
        
        # Set SOCKS proxy for requests
        socks.set_default_proxy(
            socks.SOCKS5,
            proxy_parts.hostname or 'tor',
            proxy_parts.port or 9050
        )
        socket.socket = socks.socksocket
        
    def _rotate_tor_circuit(self):
        """Request new Tor circuit for fresh IP"""
        try:
            from stem import Signal
            from stem.control import Controller
            
            with Controller.from_port(port=9051) as controller:
                controller.authenticate(password=self.tor_control_password)
                controller.signal(Signal.NEWNYM)
                time.sleep(3)  # Wait for new circuit
                
            logger.info("Tor circuit rotated successfully")
            
        except Exception as e:
            logger.warning(f"Could not rotate Tor circuit: {str(e)}")
            
    def _create_tor_session(self) -> requests.Session:
        """Create requests session with Tor proxy"""
        session = requests.Session()
        session.proxies = {
            'http': self.tor_proxy,
            'https': self.tor_proxy
        }
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/109.0'
        })
        return session
        
    def _create_selenium_driver(self):
        """Create Selenium driver with Tor proxy for JavaScript-heavy sites"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument(f'--proxy-server={self.tor_proxy}')
        
        driver = webdriver.Chrome(options=options)
        return driver
        
    def _search_vendor(self, vendor_name: str) -> List[Dict]:
        """Search for vendor profiles across darknet markets"""
        vendors = []
        session = self._create_tor_session()
        
        for market_name, market_info in self.DARKNET_MARKETS.items():
            if market_info['type'] != 'marketplace':
                continue
                
            try:
                # Basic crawl for vendor info (would need credentials for full access)
                response = session.get(
                    market_info['url'],
                    timeout=30,
                    verify=False
                )
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for vendor mentions (patterns vary by market)
                    vendor_patterns = [
                        rf'vendor[/\s:]*{re.escape(vendor_name)}',
                        rf'seller[/\s:]*{re.escape(vendor_name)}',
                        rf'shop[/\s:]*{re.escape(vendor_name)}'
                    ]
                    
                    for pattern in vendor_patterns:
                        matches = soup.find_all(text=re.compile(pattern, re.I))
                        
                        for match in matches:
                            parent = match.parent
                            
                            vendor_info = {
                                'market': market_name,
                                'vendor_name': vendor_name,
                                'profile_url': market_info['url'],
                                'reputation': self._extract_reputation(parent),
                                'products_count': self._extract_product_count(parent),
                                'pgp_key': self._extract_pgp_key(parent),
                                'bitcoin_addresses': self._extract_bitcoin_addresses(str(parent)),
                                'last_seen': datetime.now().isoformat(),
                                'raw_html': str(parent)[:1000]  # Store snippet
                            }
                            
                            vendors.append(vendor_info)
                            
            except Exception as e:
                logger.error(f"Error searching {market_name}: {str(e)}")
                continue
                
        return vendors
        
    def _search_products(self, search_term: str) -> List[Dict]:
        """Search for products/services across markets"""
        products = []
        session = self._create_tor_session()
        
        for market_name, market_info in self.DARKNET_MARKETS.items():
            if market_info['type'] != 'marketplace':
                continue
                
            try:
                # Search for products (simplified - real implementation would handle auth)
                search_url = f"{market_info['url']}/search?q={search_term}"
                response = session.get(search_url, timeout=30, verify=False)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract product listings (structure varies by market)
                    listings = soup.find_all(['div', 'li'], class_=re.compile('listing|product|item'))
                    
                    for listing in listings[:50]:  # Limit results
                        product = {
                            'market': market_name,
                            'title': self._extract_text(listing, ['title', 'name', 'h2', 'h3']),
                            'vendor': self._extract_text(listing, ['vendor', 'seller', 'by']),
                            'price_btc': self._extract_price_btc(listing),
                            'price_usd': self._extract_price_usd(listing),
                            'category': self._extract_text(listing, ['category', 'cat']),
                            'ships_from': self._extract_text(listing, ['ships', 'from', 'origin']),
                            'ships_to': self._extract_text(listing, ['ships-to', 'destination']),
                            'escrow': 'escrow' in str(listing).lower(),
                            'reviews': self._extract_reviews(listing),
                            'url': market_info['url'],
                            'scraped_at': datetime.now().isoformat()
                        }
                        
                        if product['title']:  # Only add if we found a title
                            products.append(product)
                            
            except Exception as e:
                logger.error(f"Error searching products on {market_name}: {str(e)}")
                continue
                
        return products
        
    def _track_crypto_address(self, address: str) -> List[Dict]:
        """Track cryptocurrency address across darknet markets"""
        crypto_mentions = []
        session = self._create_tor_session()
        
        for market_name, market_info in self.DARKNET_MARKETS.items():
            try:
                response = session.get(market_info['url'], timeout=30, verify=False)
                
                if response.status_code == 200:
                    # Search for the address in page content
                    if address in response.text:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Find context around the address
                        for element in soup.find_all(text=re.compile(re.escape(address))):
                            parent = element.parent
                            
                            mention = {
                                'market': market_name,
                                'address': address,
                                'context': str(parent)[:500],
                                'vendor': self._extract_nearby_vendor(parent),
                                'product': self._extract_nearby_product(parent),
                                'transaction_amount': self._extract_amount(parent),
                                'currency': 'BTC' if address.startswith(('1', '3', 'bc1')) else 'Unknown',
                                'found_at': datetime.now().isoformat(),
                                'url': market_info['url']
                            }
                            
                            crypto_mentions.append(mention)
                            
            except Exception as e:
                logger.error(f"Error tracking crypto on {market_name}: {str(e)}")
                continue
                
        return crypto_mentions
        
    def _deep_crawl_markets(self, search_term: str) -> Dict:
        """Deep crawl darknet markets for comprehensive data"""
        crawl_results = {
            'mentions': [],
            'graph': {'nodes': [], 'edges': []}
        }
        
        session = self._create_tor_session()
        visited_urls = set()
        to_visit = []
        
        # Initialize with market homepages
        for market_name, market_info in self.DARKNET_MARKETS.items():
            to_visit.append((market_info['url'], 0))  # (url, depth)
            
        pages_crawled = 0
        
        while to_visit and pages_crawled < self.max_pages:
            current_url, depth = to_visit.pop(0)
            
            if current_url in visited_urls or depth > 3:
                continue
                
            visited_urls.add(current_url)
            pages_crawled += 1
            
            try:
                response = session.get(current_url, timeout=30, verify=False)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Search for mentions of our search term
                    if search_term.lower() in response.text.lower():
                        mention = {
                            'url': current_url,
                            'search_term': search_term,
                            'occurrences': response.text.lower().count(search_term.lower()),
                            'page_title': soup.title.string if soup.title else 'Unknown',
                            'depth': depth,
                            'timestamp': datetime.now().isoformat()
                        }
                        crawl_results['mentions'].append(mention)
                        
                    # Extract links for further crawling
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        
                        # Only follow onion links
                        if '.onion' in href:
                            if href.startswith('/'):
                                href = urljoin(current_url, href)
                                
                            if href not in visited_urls:
                                to_visit.append((href, depth + 1))
                                
                    # Build network graph
                    node = {
                        'id': hashlib.md5(current_url.encode()).hexdigest(),
                        'url': current_url,
                        'type': 'page',
                        'depth': depth
                    }
                    crawl_results['graph']['nodes'].append(node)
                    
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error crawling {current_url}: {str(e)}")
                continue
                
        return crawl_results
        
    def _extract_crypto_addresses(self, results: Dict) -> List[Dict]:
        """Extract cryptocurrency addresses from all results"""
        crypto_addresses = []
        
        # Bitcoin address patterns
        btc_patterns = [
            r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',  # Legacy
            r'\bbc1[a-z0-9]{39,59}\b',  # Bech32
            r'\b3[a-km-zA-HJ-NP-Z1-9]{25,34}\b'  # P2SH
        ]
        
        # Monero address pattern
        xmr_pattern = r'\b4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}\b'
        
        # Ethereum address pattern
        eth_pattern = r'\b0x[a-fA-F0-9]{40}\b'
        
        # Search through all text content
        text_to_search = []
        
        for vendor in results.get('vendor_profiles', []):
            text_to_search.append(vendor.get('raw_html', ''))
            
        for product in results.get('product_listings', []):
            text_to_search.append(str(product))
            
        for mention in results.get('darknet_mentions', []):
            text_to_search.append(mention.get('context', ''))
            
        combined_text = ' '.join(text_to_search)
        
        # Extract Bitcoin addresses
        for pattern in btc_patterns:
            matches = re.findall(pattern, combined_text)
            for match in matches:
                crypto_addresses.append({
                    'address': match,
                    'currency': 'BTC',
                    'source': 'darknet_extraction'
                })
                
        # Extract Monero addresses
        xmr_matches = re.findall(xmr_pattern, combined_text)
        for match in xmr_matches:
            crypto_addresses.append({
                'address': match,
                'currency': 'XMR',
                'source': 'darknet_extraction'
            })
            
        # Extract Ethereum addresses
        eth_matches = re.findall(eth_pattern, combined_text)
        for match in eth_matches:
            crypto_addresses.append({
                'address': match,
                'currency': 'ETH',
                'source': 'darknet_extraction'
            })
            
        # Deduplicate
        unique_addresses = []
        seen = set()
        for addr in crypto_addresses:
            if addr['address'] not in seen:
                seen.add(addr['address'])
                unique_addresses.append(addr)
                
        return unique_addresses
        
    def _build_vendor_network(self, vendors: List[Dict]) -> Dict:
        """Build network graph of vendor relationships"""
        graph = {
            'nodes': [],
            'edges': []
        }
        
        # Create vendor nodes
        for vendor in vendors:
            node = {
                'id': hashlib.md5(vendor['vendor_name'].encode()).hexdigest(),
                'label': vendor['vendor_name'],
                'type': 'vendor',
                'market': vendor['market'],
                'reputation': vendor.get('reputation', 'Unknown'),
                'products_count': vendor.get('products_count', 0)
            }
            graph['nodes'].append(node)
            
            # Add market node
            market_node = {
                'id': hashlib.md5(vendor['market'].encode()).hexdigest(),
                'label': vendor['market'],
                'type': 'market'
            }
            
            if market_node not in graph['nodes']:
                graph['nodes'].append(market_node)
                
            # Create edge between vendor and market
            edge = {
                'source': node['id'],
                'target': market_node['id'],
                'type': 'operates_on'
            }
            graph['edges'].append(edge)
            
        return graph
        
    def _calculate_market_stats(self, results: Dict) -> Dict:
        """Calculate darknet market statistics"""
        stats = {
            'total_vendors': len(results.get('vendor_profiles', [])),
            'total_products': len(results.get('product_listings', [])),
            'markets_searched': len(self.DARKNET_MARKETS),
            'crypto_addresses_found': len(results.get('crypto_addresses', [])),
            'average_price_btc': 0,
            'average_price_usd': 0,
            'most_common_category': None,
            'most_active_vendor': None
        }
        
        # Calculate average prices
        btc_prices = [p['price_btc'] for p in results.get('product_listings', []) 
                      if p.get('price_btc')]
        if btc_prices:
            stats['average_price_btc'] = sum(btc_prices) / len(btc_prices)
            
        usd_prices = [p['price_usd'] for p in results.get('product_listings', []) 
                      if p.get('price_usd')]
        if usd_prices:
            stats['average_price_usd'] = sum(usd_prices) / len(usd_prices)
            
        # Find most common category
        categories = [p['category'] for p in results.get('product_listings', []) 
                      if p.get('category')]
        if categories:
            from collections import Counter
            stats['most_common_category'] = Counter(categories).most_common(1)[0][0]
            
        # Find most active vendor
        vendors = [p['vendor'] for p in results.get('product_listings', []) 
                   if p.get('vendor')]
        if vendors:
            from collections import Counter
            stats['most_active_vendor'] = Counter(vendors).most_common(1)[0][0]
            
        return stats
        
    def _assess_threats(self, results: Dict) -> Dict:
        """Assess threat level based on darknet findings"""
        assessment = {
            'threat_level': 'LOW',
            'risk_score': 0,
            'indicators': [],
            'recommendations': []
        }
        
        risk_score = 0
        
        # Check for direct mentions
        if results.get('darknet_mentions'):
            risk_score += 20
            assessment['indicators'].append('Direct mentions found on darknet')
            
        # Check for vendor profiles
        if results.get('vendor_profiles'):
            risk_score += 30
            assessment['indicators'].append(f"Vendor profiles found: {len(results['vendor_profiles'])}")
            
        # Check for high-risk products
        high_risk_categories = ['drugs', 'weapons', 'exploits', 'malware', 'stolen']
        for product in results.get('product_listings', []):
            if any(risk in str(product).lower() for risk in high_risk_categories):
                risk_score += 10
                assessment['indicators'].append(f"High-risk product: {product.get('title', 'Unknown')}")
                
        # Check for cryptocurrency activity
        if results.get('crypto_addresses'):
            risk_score += 15
            assessment['indicators'].append(f"Cryptocurrency addresses found: {len(results['crypto_addresses'])}")
            
        # Determine threat level
        if risk_score >= 60:
            assessment['threat_level'] = 'HIGH'
            assessment['recommendations'].append('Immediate investigation recommended')
            assessment['recommendations'].append('Monitor cryptocurrency transactions')
            
        elif risk_score >= 30:
            assessment['threat_level'] = 'MEDIUM'
            assessment['recommendations'].append('Further monitoring recommended')
            assessment['recommendations'].append('Cross-reference with other intelligence sources')
            
        else:
            assessment['threat_level'] = 'LOW'
            assessment['recommendations'].append('Continue routine monitoring')
            
        assessment['risk_score'] = min(risk_score, 100)
        
        return assessment
        
    def _extract_text(self, element, selectors: List[str]) -> str:
        """Extract text from element using multiple selectors"""
        for selector in selectors:
            found = element.find(class_=re.compile(selector, re.I))
            if not found:
                found = element.find(selector)
                
            if found:
                return found.get_text(strip=True)
                
        return ''
        
    def _extract_price_btc(self, element) -> float:
        """Extract Bitcoin price from element"""
        price_patterns = [
            r'(\d+\.?\d*)\s*BTC',
            r'₿\s*(\d+\.?\d*)',
            r'Bitcoin:\s*(\d+\.?\d*)'
        ]
        
        text = str(element)
        for pattern in price_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
                    
        return 0
        
    def _extract_price_usd(self, element) -> float:
        """Extract USD price from element"""
        price_patterns = [
            r'\$\s*(\d+\.?\d*)',
            r'USD\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*USD'
        ]
        
        text = str(element)
        for pattern in price_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
                    
        return 0
        
    def _extract_bitcoin_addresses(self, text: str) -> List[str]:
        """Extract Bitcoin addresses from text"""
        patterns = [
            r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',
            r'\bbc1[a-z0-9]{39,59}\b'
        ]
        
        addresses = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            addresses.extend(matches)
            
        return addresses
        
    def _extract_reputation(self, element) -> str:
        """Extract vendor reputation/rating"""
        rep_patterns = [
            r'(\d+\.?\d*)\s*stars?',
            r'rating:\s*(\d+\.?\d*)',
            r'reputation:\s*(\d+)',
            r'(\d+)%\s*positive'
        ]
        
        text = str(element)
        for pattern in rep_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(1)
                
        return 'Unknown'
        
    def _extract_product_count(self, element) -> int:
        """Extract number of products from vendor"""
        count_patterns = [
            r'(\d+)\s*products?',
            r'(\d+)\s*listings?',
            r'items:\s*(\d+)'
        ]
        
        text = str(element)
        for pattern in count_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                try:
                    return int(match.group(1))
                except:
                    pass
                    
        return 0
        
    def _extract_pgp_key(self, element) -> Optional[str]:
        """Extract PGP key if present"""
        pgp_pattern = r'-----BEGIN PGP PUBLIC KEY BLOCK-----(.*?)-----END PGP PUBLIC KEY BLOCK-----'
        match = re.search(pgp_pattern, str(element), re.DOTALL)
        
        if match:
            return match.group(0)
            
        return None
        
    def _extract_reviews(self, element) -> int:
        """Extract review count"""
        review_patterns = [
            r'(\d+)\s*reviews?',
            r'(\d+)\s*feedbacks?',
            r'reviewed\s*(\d+)\s*times?'
        ]
        
        text = str(element)
        for pattern in review_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                try:
                    return int(match.group(1))
                except:
                    pass
                    
        return 0
        
    def _extract_nearby_vendor(self, element) -> Optional[str]:
        """Extract vendor name near an element"""
        # Look for vendor indicators in parent elements
        parent = element
        for _ in range(5):  # Check up to 5 parent levels
            if parent:
                text = str(parent)
                vendor_match = re.search(r'vendor[:\s]+([a-zA-Z0-9_-]+)', text, re.I)
                if vendor_match:
                    return vendor_match.group(1)
                    
                parent = parent.parent
                
        return None
        
    def _extract_nearby_product(self, element) -> Optional[str]:
        """Extract product name near an element"""
        parent = element
        for _ in range(5):
            if parent:
                # Look for product title
                title = parent.find(['h1', 'h2', 'h3', 'h4'])
                if title:
                    return title.get_text(strip=True)
                    
                parent = parent.parent
                
        return None
        
    def _extract_amount(self, element) -> Optional[float]:
        """Extract transaction amount near crypto address"""
        amount_patterns = [
            r'(\d+\.?\d*)\s*BTC',
            r'(\d+\.?\d*)\s*XMR',
            r'amount:\s*(\d+\.?\d*)'
        ]
        
        text = str(element)
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
                    
        return None
        
    def _cache_results(self, query: str, results: Dict):
        """Cache darknet search results"""
        # Implementation would cache to Redis or PostgreSQL
        pass
        
    def _store_in_neo4j(self, query: str, results: Dict):
        """Store darknet intelligence in Neo4j graph"""
        try:
            if not self.neo4j_driver:
                self.neo4j_driver = GraphDatabase.driver(
                    os.getenv('NEO4J_URI'),
                    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
                )
                
            with self.neo4j_driver.session() as session:
                # Create search node
                session.run("""
                    MERGE (s:DarknetSearch {query: $query})
                    SET s.timestamp = datetime(),
                        s.threat_level = $threat_level,
                        s.risk_score = $risk_score
                """, query=query,
                     threat_level=results['risk_assessment'].get('threat_level', 'UNKNOWN'),
                     risk_score=results['risk_assessment'].get('risk_score', 0))
                
                # Create vendor nodes
                for vendor in results.get('vendor_profiles', []):
                    session.run("""
                        MERGE (v:DarknetVendor {name: $name})
                        SET v.market = $market,
                            v.reputation = $reputation,
                            v.products_count = $products_count,
                            v.last_seen = datetime()
                        MERGE (s:DarknetSearch {query: $query})
                        MERGE (s)-[:FOUND_VENDOR]->(v)
                    """, name=vendor['vendor_name'],
                         market=vendor['market'],
                         reputation=vendor.get('reputation', 'Unknown'),
                         products_count=vendor.get('products_count', 0),
                         query=query)
                    
                    # Link Bitcoin addresses to vendors
                    for btc_addr in vendor.get('bitcoin_addresses', []):
                        session.run("""
                            MERGE (a:BitcoinAddress {address: $address})
                            MERGE (v:DarknetVendor {name: $vendor})
                            MERGE (v)-[:USES_ADDRESS]->(a)
                        """, address=btc_addr,
                             vendor=vendor['vendor_name'])
                             
                # Create product nodes
                for product in results.get('product_listings', []):
                    if product.get('title') and product.get('vendor'):
                        session.run("""
                            MERGE (p:DarknetProduct {title: $title, market: $market})
                            SET p.price_btc = $price_btc,
                                p.price_usd = $price_usd,
                                p.category = $category,
                                p.scraped_at = datetime()
                            MERGE (v:DarknetVendor {name: $vendor})
                            MERGE (v)-[:SELLS]->(p)
                            MERGE (s:DarknetSearch {query: $query})
                            MERGE (s)-[:FOUND_PRODUCT]->(p)
                        """, title=product['title'],
                             market=product['market'],
                             price_btc=product.get('price_btc', 0),
                             price_usd=product.get('price_usd', 0),
                             category=product.get('category', 'Unknown'),
                             vendor=product['vendor'],
                             query=query)
                             
                # Create crypto address nodes
                for crypto in results.get('crypto_addresses', []):
                    session.run("""
                        MERGE (a:CryptoAddress {address: $address})
                        SET a.currency = $currency,
                            a.source = $source
                        MERGE (s:DarknetSearch {query: $query})
                        MERGE (s)-[:FOUND_ADDRESS]->(a)
                    """, address=crypto['address'],
                         currency=crypto['currency'],
                         source=crypto['source'],
                         query=query)
                         
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
                'name': 'DarknetMarketAnalyzer',
                'description': 'Analyze darknet markets and vendor networks via Tor',
                'python_module': 'custom_analyzers.darknet_analyzer.DarknetMarketAnalyzer',
                'disabled': False,
                'type': 'observable',
                'docker_based': False,
                'maximum_tlp': 'RED',
                'observable_supported': ['generic', 'hash', 'bitcoin_address', 'username'],
                'supported_filetypes': [],
                'run_hash': False,
                'run_hash_type': '',
                'not_supported_filetypes': [],
                'parameters': {
                    'search_type': {
                        'type': 'str',
                        'description': 'Type of search: vendor, product, cryptocurrency, forum_post',
                        'default': 'vendor'
                    },
                    'deep_crawl': {
                        'type': 'bool',
                        'description': 'Perform deep crawl of darknet markets',
                        'default': False
                    },
                    'extract_crypto': {
                        'type': 'bool',
                        'description': 'Extract cryptocurrency addresses',
                        'default': True
                    },
                    'max_pages': {
                        'type': 'int',
                        'description': 'Maximum pages to crawl',
                        'default': 10
                    }
                }
            }
        ]
        return patches
