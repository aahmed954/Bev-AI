import os
#!/usr/bin/env python3
"""
Decentralized Market Crawler - Phase 7 Alternative Market Intelligence Platform
Place in: /home/starlord/Projects/Bev/src/alternative_market/dm_crawler.py

Tor-enabled marketplace discovery and monitoring with multi-marketplace protocol support.
"""

import asyncio
import aiohttp
import json
import time
import hashlib
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import asyncpg
import aioredis
from aiokafka import AIOKafkaProducer
import socks
import aiosocks
from bs4 import BeautifulSoup
import numpy as np
from collections import defaultdict
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketplaceConfig:
    """Configuration for a marketplace"""
    name: str
    base_url: str
    protocol: str  # 'tor', 'i2p', 'clearnet'
    auth_required: bool
    rate_limit_per_second: float
    endpoints: Dict[str, str]
    selectors: Dict[str, str]
    headers: Dict[str, str]

@dataclass
class VendorListing:
    """Vendor information structure"""
    vendor_id: str
    name: str
    marketplace: str
    rating: float
    total_orders: int
    registration_date: Optional[datetime]
    last_seen: datetime
    verification_status: str
    categories: List[str]
    geographical_location: Optional[str]
    pgp_key: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class ProductListing:
    """Product catalog structure"""
    product_id: str
    vendor_id: str
    marketplace: str
    title: str
    description: str
    category: str
    subcategory: Optional[str]
    price: float
    currency: str
    quantity_available: int
    shipping_from: Optional[str]
    shipping_to: List[str]
    posted_date: datetime
    last_updated: datetime
    images: List[str]
    metadata: Dict[str, Any]

@dataclass
class PriceTrend:
    """Price trend analysis structure"""
    product_id: str
    marketplace: str
    category: str
    timestamp: datetime
    price: float
    currency: str
    volume_indicator: int
    trend_direction: str  # 'up', 'down', 'stable'
    volatility_score: float


class TorProxyManager:
    """Manages Tor proxy connections for anonymized crawling"""

    def __init__(self, proxy_host: str = "172.30.0.17", proxy_port: int = 9050):
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.session_pool = []
        self.max_sessions = 10

    async def create_session(self) -> aiohttp.ClientSession:
        """Create Tor-enabled HTTP session"""

        # Create SOCKS5 connector for Tor
        connector = aiohttp.TCPConnector(
            use_dns_cache=False,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )

        # Configure proxy
        proxy_url = f"socks5://{self.proxy_host}:{self.proxy_port}"

        session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0'
            }
        )

        return session

    async def get_session(self) -> aiohttp.ClientSession:
        """Get available session from pool"""

        if len(self.session_pool) < self.max_sessions:
            session = await self.create_session()
            self.session_pool.append(session)
            return session

        # Return random session from pool
        import random
        return random.choice(self.session_pool)

    async def rotate_identity(self):
        """Request new Tor circuit"""

        try:
            # Connect to Tor control port (if available)
            # This would require stem library in production
            logger.info("Tor identity rotation requested")
        except Exception as e:
            logger.warning(f"Could not rotate Tor identity: {e}")


class RateLimiter:
    """Advanced rate limiting with per-domain tracking"""

    def __init__(self):
        self.domain_limits: Dict[str, float] = {}
        self.request_history: Dict[str, List[float]] = defaultdict(list)
        self.burst_allowance: Dict[str, int] = defaultdict(lambda: 5)

    async def wait_if_needed(self, domain: str, rate_limit: float):
        """Wait if rate limit would be exceeded"""

        now = time.time()
        history = self.request_history[domain]

        # Clean old requests (older than 1 second)
        cutoff = now - 1.0
        history[:] = [t for t in history if t > cutoff]

        # Check if we need to wait
        if len(history) >= rate_limit:
            sleep_time = 1.0 - (now - history[0])
            if sleep_time > 0:
                logger.debug(f"Rate limiting {domain}: sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

        # Record this request
        history.append(now)

    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Get rate limiting statistics for domain"""

        now = time.time()
        history = self.request_history[domain]
        recent_requests = [t for t in history if t > now - 60]  # Last minute

        return {
            'domain': domain,
            'requests_last_minute': len(recent_requests),
            'burst_allowance_remaining': self.burst_allowance[domain],
            'current_rate': len([t for t in history if t > now - 1.0])  # Per second
        }


class MarketplaceCrawler:
    """Main marketplace crawler with protocol support"""

    def __init__(self, db_config: Dict[str, Any], redis_config: Dict[str, Any],
                 kafka_config: Dict[str, Any]):

        self.db_config = db_config
        self.redis_config = redis_config
        self.kafka_config = kafka_config

        # Core components
        self.tor_manager = TorProxyManager()
        self.rate_limiter = RateLimiter()

        # Data storage
        self.db_pool = None
        self.redis_client = None
        self.kafka_producer = None

        # Crawling state
        self.active_crawls: Set[str] = set()
        self.crawl_stats = defaultdict(dict)
        self.last_crawl_times: Dict[str, datetime] = {}

        # Marketplace configurations
        self.marketplaces: Dict[str, MarketplaceConfig] = {}
        self._load_marketplace_configs()

    def _load_marketplace_configs(self):
        """Load marketplace configurations"""

        # Example marketplace configurations
        self.marketplaces = {
            'example_market_1': MarketplaceConfig(
                name="Example Market 1",
                base_url="http://example1.onion",
                protocol="tor",
                auth_required=True,
                rate_limit_per_second=2.0,
                endpoints={
                    'vendors': '/vendors',
                    'products': '/products',
                    'categories': '/categories'
                },
                selectors={
                    'vendor_name': '.vendor-name',
                    'vendor_rating': '.rating',
                    'product_title': '.product-title',
                    'product_price': '.price'
                },
                headers={
                    'Accept': 'text/html,application/xhtml+xml',
                    'Accept-Language': 'en-US,en;q=0.5'
                }
            ),
            'example_market_2': MarketplaceConfig(
                name="Example Market 2",
                base_url="http://example2.onion",
                protocol="tor",
                auth_required=False,
                rate_limit_per_second=1.5,
                endpoints={
                    'vendors': '/api/vendors',
                    'products': '/api/products'
                },
                selectors={
                    'vendor_info': '.vendor-card',
                    'product_info': '.product-item'
                },
                headers={
                    'Accept': 'application/json'
                }
            )
        }

    async def initialize(self):
        """Initialize all connections and resources"""

        logger.info("Initializing Decentralized Market Crawler...")

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
            f"redis://{self.redis_config['host']}:{self.redis_config['port']}/0"
        )

        # Kafka producer
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_config['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )
        await self.kafka_producer.start()

        # Initialize database tables
        await self._initialize_database_tables()

        logger.info("Decentralized Market Crawler initialized successfully")

    async def _initialize_database_tables(self):
        """Create database tables if they don't exist"""

        tables = [
            '''
            CREATE TABLE IF NOT EXISTS marketplace_vendors (
                vendor_id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(500),
                marketplace VARCHAR(100),
                rating FLOAT,
                total_orders INTEGER,
                registration_date TIMESTAMP,
                last_seen TIMESTAMP,
                verification_status VARCHAR(50),
                categories TEXT[],
                geographical_location VARCHAR(100),
                pgp_key TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            ''',
            '''
            CREATE TABLE IF NOT EXISTS marketplace_products (
                product_id VARCHAR(255) PRIMARY KEY,
                vendor_id VARCHAR(255),
                marketplace VARCHAR(100),
                title TEXT,
                description TEXT,
                category VARCHAR(100),
                subcategory VARCHAR(100),
                price DECIMAL(15,2),
                currency VARCHAR(10),
                quantity_available INTEGER,
                shipping_from VARCHAR(100),
                shipping_to TEXT[],
                posted_date TIMESTAMP,
                last_updated TIMESTAMP,
                images TEXT[],
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            ''',
            '''
            CREATE TABLE IF NOT EXISTS marketplace_price_trends (
                id SERIAL PRIMARY KEY,
                product_id VARCHAR(255),
                marketplace VARCHAR(100),
                category VARCHAR(100),
                timestamp TIMESTAMP,
                price DECIMAL(15,2),
                currency VARCHAR(10),
                volume_indicator INTEGER,
                trend_direction VARCHAR(20),
                volatility_score FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            );
            ''',
            '''
            CREATE INDEX IF NOT EXISTS idx_vendors_marketplace ON marketplace_vendors(marketplace);
            CREATE INDEX IF NOT EXISTS idx_vendors_last_seen ON marketplace_vendors(last_seen);
            CREATE INDEX IF NOT EXISTS idx_products_marketplace ON marketplace_products(marketplace);
            CREATE INDEX IF NOT EXISTS idx_products_category ON marketplace_products(category);
            CREATE INDEX IF NOT EXISTS idx_price_trends_timestamp ON marketplace_price_trends(timestamp);
            '''
        ]

        async with self.db_pool.acquire() as conn:
            for table_sql in tables:
                await conn.execute(table_sql)

    async def crawl_marketplace(self, marketplace_name: str,
                              crawl_type: str = 'full') -> Dict[str, Any]:
        """Crawl a specific marketplace"""

        if marketplace_name not in self.marketplaces:
            raise ValueError(f"Unknown marketplace: {marketplace_name}")

        if marketplace_name in self.active_crawls:
            logger.warning(f"Crawl already active for {marketplace_name}")
            return {'status': 'already_active'}

        config = self.marketplaces[marketplace_name]
        self.active_crawls.add(marketplace_name)

        try:
            logger.info(f"Starting {crawl_type} crawl of {marketplace_name}")

            # Rate limiting check
            await self.rate_limiter.wait_if_needed(
                marketplace_name,
                config.rate_limit_per_second
            )

            # Get Tor session
            session = await self.tor_manager.get_session()

            results = {
                'marketplace': marketplace_name,
                'crawl_type': crawl_type,
                'start_time': datetime.now(),
                'vendors_found': 0,
                'products_found': 0,
                'errors': []
            }

            if crawl_type in ['full', 'vendors']:
                vendor_results = await self._crawl_vendors(config, session)
                results['vendors_found'] = vendor_results['count']
                results['errors'].extend(vendor_results['errors'])

            if crawl_type in ['full', 'products']:
                product_results = await self._crawl_products(config, session)
                results['products_found'] = product_results['count']
                results['errors'].extend(product_results['errors'])

            results['end_time'] = datetime.now()
            results['duration'] = (results['end_time'] - results['start_time']).total_seconds()

            # Update last crawl time
            self.last_crawl_times[marketplace_name] = datetime.now()

            # Store crawl statistics
            await self._store_crawl_stats(results)

            # Publish to Kafka
            await self.kafka_producer.send(
                'marketplace_crawl_results',
                key=marketplace_name,
                value=results
            )

            logger.info(f"Completed crawl of {marketplace_name}: "
                       f"{results['vendors_found']} vendors, "
                       f"{results['products_found']} products")

            return results

        except Exception as e:
            logger.error(f"Error crawling {marketplace_name}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'marketplace': marketplace_name
            }
        finally:
            self.active_crawls.discard(marketplace_name)

    async def _crawl_vendors(self, config: MarketplaceConfig,
                           session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Crawl vendor listings from marketplace"""

        vendors_found = 0
        errors = []

        try:
            vendor_url = urljoin(config.base_url, config.endpoints['vendors'])

            async with session.get(
                vendor_url,
                headers=config.headers,
                proxy=f"socks5://{self.tor_manager.proxy_host}:{self.tor_manager.proxy_port}"
            ) as response:

                if response.status != 200:
                    errors.append(f"HTTP {response.status} for vendors endpoint")
                    return {'count': 0, 'errors': errors}

                content = await response.text()
                vendors = await self._parse_vendors(content, config)

                # Store vendors in database
                for vendor in vendors:
                    await self._store_vendor(vendor)
                    vendors_found += 1

        except Exception as e:
            errors.append(f"Vendor crawling error: {e}")

        return {'count': vendors_found, 'errors': errors}

    async def _crawl_products(self, config: MarketplaceConfig,
                            session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Crawl product listings from marketplace"""

        products_found = 0
        errors = []

        try:
            product_url = urljoin(config.base_url, config.endpoints['products'])

            async with session.get(
                product_url,
                headers=config.headers,
                proxy=f"socks5://{self.tor_manager.proxy_host}:{self.tor_manager.proxy_port}"
            ) as response:

                if response.status != 200:
                    errors.append(f"HTTP {response.status} for products endpoint")
                    return {'count': 0, 'errors': errors}

                content = await response.text()
                products = await self._parse_products(content, config)

                # Store products and analyze price trends
                for product in products:
                    await self._store_product(product)
                    await self._analyze_price_trend(product)
                    products_found += 1

        except Exception as e:
            errors.append(f"Product crawling error: {e}")

        return {'count': products_found, 'errors': errors}

    async def _parse_vendors(self, content: str,
                           config: MarketplaceConfig) -> List[VendorListing]:
        """Parse vendor information from HTML/JSON content"""

        vendors = []

        try:
            soup = BeautifulSoup(content, 'html.parser')
            vendor_elements = soup.select(config.selectors.get('vendor_info', '.vendor'))

            for element in vendor_elements:
                try:
                    # Extract vendor data using selectors
                    name = element.select_one(config.selectors.get('vendor_name', '.name'))
                    rating = element.select_one(config.selectors.get('vendor_rating', '.rating'))

                    vendor = VendorListing(
                        vendor_id=hashlib.md5(f"{config.name}_{name.text if name else 'unknown'}".encode()).hexdigest(),
                        name=name.text.strip() if name else 'Unknown',
                        marketplace=config.name,
                        rating=float(rating.text.strip()) if rating and rating.text.strip() else 0.0,
                        total_orders=0,  # Would extract from page
                        registration_date=None,  # Would parse from page
                        last_seen=datetime.now(),
                        verification_status='unverified',
                        categories=[],
                        geographical_location=None,
                        pgp_key=None,
                        metadata={
                            'raw_html': str(element)[:1000],  # Truncated for storage
                            'crawl_timestamp': datetime.now().isoformat()
                        }
                    )

                    vendors.append(vendor)

                except Exception as e:
                    logger.warning(f"Error parsing vendor element: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing vendors: {e}")

        return vendors

    async def _parse_products(self, content: str,
                            config: MarketplaceConfig) -> List[ProductListing]:
        """Parse product information from HTML/JSON content"""

        products = []

        try:
            soup = BeautifulSoup(content, 'html.parser')
            product_elements = soup.select(config.selectors.get('product_info', '.product'))

            for element in product_elements:
                try:
                    title = element.select_one(config.selectors.get('product_title', '.title'))
                    price = element.select_one(config.selectors.get('product_price', '.price'))

                    # Extract price value and currency
                    price_text = price.text.strip() if price else '0'
                    price_match = re.search(r'(\d+\.?\d*)', price_text)
                    price_value = float(price_match.group(1)) if price_match else 0.0

                    # Extract currency
                    currency_match = re.search(r'([A-Z]{3}|BTC|ETH|XMR)', price_text)
                    currency = currency_match.group(1) if currency_match else 'USD'

                    product = ProductListing(
                        product_id=hashlib.md5(f"{config.name}_{title.text if title else uuid.uuid4()}".encode()).hexdigest(),
                        vendor_id='unknown',  # Would extract from page
                        marketplace=config.name,
                        title=title.text.strip() if title else 'Unknown Product',
                        description='',  # Would extract from detailed page
                        category='uncategorized',
                        subcategory=None,
                        price=price_value,
                        currency=currency,
                        quantity_available=0,
                        shipping_from=None,
                        shipping_to=[],
                        posted_date=datetime.now(),
                        last_updated=datetime.now(),
                        images=[],
                        metadata={
                            'raw_html': str(element)[:1000],
                            'crawl_timestamp': datetime.now().isoformat()
                        }
                    )

                    products.append(product)

                except Exception as e:
                    logger.warning(f"Error parsing product element: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing products: {e}")

        return products

    async def _store_vendor(self, vendor: VendorListing):
        """Store vendor information in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO marketplace_vendors
                (vendor_id, name, marketplace, rating, total_orders, registration_date,
                 last_seen, verification_status, categories, geographical_location,
                 pgp_key, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (vendor_id) DO UPDATE SET
                    last_seen = EXCLUDED.last_seen,
                    rating = EXCLUDED.rating,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                ''',
                vendor.vendor_id, vendor.name, vendor.marketplace, vendor.rating,
                vendor.total_orders, vendor.registration_date, vendor.last_seen,
                vendor.verification_status, vendor.categories,
                vendor.geographical_location, vendor.pgp_key, json.dumps(vendor.metadata)
            )

    async def _store_product(self, product: ProductListing):
        """Store product information in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO marketplace_products
                (product_id, vendor_id, marketplace, title, description, category,
                 subcategory, price, currency, quantity_available, shipping_from,
                 shipping_to, posted_date, last_updated, images, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (product_id) DO UPDATE SET
                    price = EXCLUDED.price,
                    quantity_available = EXCLUDED.quantity_available,
                    last_updated = EXCLUDED.last_updated,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                ''',
                product.product_id, product.vendor_id, product.marketplace,
                product.title, product.description, product.category,
                product.subcategory, product.price, product.currency,
                product.quantity_available, product.shipping_from,
                product.shipping_to, product.posted_date, product.last_updated,
                product.images, json.dumps(product.metadata)
            )

    async def _analyze_price_trend(self, product: ProductListing):
        """Analyze and store price trend data"""

        # Get historical prices for this product
        async with self.db_pool.acquire() as conn:
            historical_prices = await conn.fetch(
                '''
                SELECT price, timestamp FROM marketplace_price_trends
                WHERE product_id = $1 AND marketplace = $2
                ORDER BY timestamp DESC LIMIT 10
                ''',
                product.product_id, product.marketplace
            )

        # Calculate trend direction and volatility
        prices = [float(record['price']) for record in historical_prices]
        prices.append(product.price)

        if len(prices) >= 2:
            # Simple trend calculation
            if product.price > prices[-2]:
                trend_direction = 'up'
            elif product.price < prices[-2]:
                trend_direction = 'down'
            else:
                trend_direction = 'stable'

            # Calculate volatility (coefficient of variation)
            if len(prices) >= 3:
                volatility_score = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
            else:
                volatility_score = 0.0
        else:
            trend_direction = 'stable'
            volatility_score = 0.0

        # Store price trend
        price_trend = PriceTrend(
            product_id=product.product_id,
            marketplace=product.marketplace,
            category=product.category,
            timestamp=datetime.now(),
            price=product.price,
            currency=product.currency,
            volume_indicator=1,  # Simple indicator
            trend_direction=trend_direction,
            volatility_score=volatility_score
        )

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                '''
                INSERT INTO marketplace_price_trends
                (product_id, marketplace, category, timestamp, price, currency,
                 volume_indicator, trend_direction, volatility_score)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ''',
                price_trend.product_id, price_trend.marketplace,
                price_trend.category, price_trend.timestamp, price_trend.price,
                price_trend.currency, price_trend.volume_indicator,
                price_trend.trend_direction, price_trend.volatility_score
            )

    async def _store_crawl_stats(self, results: Dict[str, Any]):
        """Store crawling statistics in Redis"""

        stats_key = f"crawl_stats:{results['marketplace']}:{datetime.now().strftime('%Y%m%d')}"

        await self.redis_client.hset(
            stats_key,
            mapping={
                'vendors_found': results['vendors_found'],
                'products_found': results['products_found'],
                'duration': results['duration'],
                'errors': len(results['errors']),
                'timestamp': results['start_time'].isoformat()
            }
        )

        # Set expiration (30 days)
        await self.redis_client.expire(stats_key, 30 * 24 * 3600)

    async def schedule_crawls(self, schedule_config: Dict[str, Any]):
        """Schedule automatic crawls for all configured marketplaces"""

        logger.info("Starting scheduled crawling...")

        while True:
            try:
                for marketplace_name, config in self.marketplaces.items():
                    # Check if enough time has passed since last crawl
                    last_crawl = self.last_crawl_times.get(marketplace_name)
                    interval = schedule_config.get('interval_hours', 24)

                    if (not last_crawl or
                        (datetime.now() - last_crawl).total_seconds() > interval * 3600):

                        # Start crawl in background
                        asyncio.create_task(
                            self.crawl_marketplace(marketplace_name, 'full')
                        )

                # Wait before next schedule check
                await asyncio.sleep(schedule_config.get('check_interval_minutes', 60) * 60)

            except Exception as e:
                logger.error(f"Error in scheduled crawling: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def get_crawl_statistics(self) -> Dict[str, Any]:
        """Get comprehensive crawling statistics"""

        stats = {
            'active_crawls': list(self.active_crawls),
            'marketplaces_configured': len(self.marketplaces),
            'last_crawl_times': {
                name: time.isoformat() if time else None
                for name, time in self.last_crawl_times.items()
            },
            'rate_limiter_stats': {}
        }

        # Get rate limiter stats for each marketplace
        for marketplace_name in self.marketplaces:
            stats['rate_limiter_stats'][marketplace_name] = \
                self.rate_limiter.get_domain_stats(marketplace_name)

        # Get database statistics
        async with self.db_pool.acquire() as conn:
            vendor_count = await conn.fetchval(
                'SELECT COUNT(*) FROM marketplace_vendors'
            )
            product_count = await conn.fetchval(
                'SELECT COUNT(*) FROM marketplace_products'
            )

            stats['database_stats'] = {
                'total_vendors': vendor_count,
                'total_products': product_count
            }

        return stats

    async def cleanup(self):
        """Cleanup resources"""

        logger.info("Cleaning up Decentralized Market Crawler...")

        if self.kafka_producer:
            await self.kafka_producer.stop()

        if self.redis_client:
            await self.redis_client.close()

        if self.db_pool:
            await self.db_pool.close()

        # Close Tor sessions
        for session in self.tor_manager.session_pool:
            await session.close()


# Example usage and configuration
async def main():
    """Example usage of the Decentralized Market Crawler"""

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

    schedule_config = {
        'interval_hours': 24,
        'check_interval_minutes': 60
    }

    # Initialize crawler
    crawler = MarketplaceCrawler(db_config, redis_config, kafka_config)
    await crawler.initialize()

    try:
        # Start scheduled crawling
        crawl_task = asyncio.create_task(
            crawler.schedule_crawls(schedule_config)
        )

        # Run manual crawl as example
        results = await crawler.crawl_marketplace('example_market_1', 'full')
        print(f"Crawl results: {results}")

        # Get statistics
        stats = await crawler.get_crawl_statistics()
        print(f"Crawl statistics: {stats}")

        # Keep running scheduled crawls
        await crawl_task

    except KeyboardInterrupt:
        logger.info("Shutting down crawler...")
    finally:
        await crawler.cleanup()


if __name__ == "__main__":
    asyncio.run(main())