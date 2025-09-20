#!/usr/bin/env python3
"""
Geographic Distribution Manager for BEV OSINT Framework
Manages geographic routing and region-specific proxy allocation

Features:
- Geographic proxy routing (US-East, US-West, EU-Central, Asia-Pacific)
- Intelligent region selection based on target analysis
- Latency-based geographic optimization
- Regional failover and load distribution
- GeoIP analysis and target locality detection
- Regional compliance and data sovereignty handling

Place in: /home/starlord/Projects/Bev/src/infrastructure/geo_router.py
"""

import asyncio
import aiohttp
import json
import logging
import time
import ipaddress
import geoip2.database
import geoip2.errors
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import asyncpg
import aioredis
from urllib.parse import urlparse
import statistics
import pycountry
import math
import requests
from pathlib import Path

# Import proxy manager components
from .proxy_manager import ProxyEndpoint, ProxyRegion, ProxyType, ProxyStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TargetRegion(Enum):
    """Target geographic regions for optimization"""
    NORTH_AMERICA = "north_america"
    SOUTH_AMERICA = "south_america"
    EUROPE = "europe"
    ASIA = "asia"
    OCEANIA = "oceania"
    AFRICA = "africa"
    MIDDLE_EAST = "middle_east"
    UNKNOWN = "unknown"

class ComplianceRegion(Enum):
    """Compliance and data sovereignty regions"""
    GDPR_EU = "gdpr_eu"              # European Union - GDPR compliance
    CCPA_US = "ccpa_us"              # California - CCPA compliance
    PIPEDA_CA = "pipeda_ca"          # Canada - PIPEDA compliance
    LGPD_BR = "lgpd_br"              # Brazil - LGPD compliance
    PDPA_SG = "pdpa_sg"              # Singapore - PDPA compliance
    DPA_UK = "dpa_uk"                # United Kingdom - DPA compliance
    STANDARD = "standard"             # Standard regions

@dataclass
class GeoLocation:
    """Geographic location information"""
    latitude: float
    longitude: float
    country: str
    country_code: str
    region: str
    city: str
    timezone: Optional[str] = None
    accuracy_radius: Optional[int] = None
    asn: Optional[int] = None
    isp: Optional[str] = None

    def distance_to(self, other: 'GeoLocation') -> float:
        """Calculate distance in kilometers using Haversine formula"""
        R = 6371  # Earth's radius in kilometers

        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (math.sin(dlat/2)**2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))

        return R * c

@dataclass
class TargetAnalysis:
    """Analysis result for target geographic optimization"""
    target_host: str
    target_ip: Optional[str] = None
    geo_location: Optional[GeoLocation] = None
    target_region: TargetRegion = TargetRegion.UNKNOWN
    compliance_regions: List[ComplianceRegion] = None
    optimal_proxy_regions: List[ProxyRegion] = None
    cdn_detected: bool = False
    load_balancer_detected: bool = False
    geo_blocking_detected: bool = False
    analysis_timestamp: datetime = None

    def __post_init__(self):
        if self.compliance_regions is None:
            self.compliance_regions = []
        if self.optimal_proxy_regions is None:
            self.optimal_proxy_regions = []
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now()

@dataclass
class RegionLatency:
    """Latency measurements for a region"""
    region: ProxyRegion
    average_latency: float
    min_latency: float
    max_latency: float
    sample_count: int
    last_updated: datetime
    reliability_score: float = 1.0

class GeoIPResolver:
    """GeoIP resolution and analysis"""

    def __init__(self, geoip_db_path: Optional[str] = None):
        self.geoip_db_path = geoip_db_path or self._download_geoip_db()
        self.geoip_reader = None
        self.dns_resolver_cache = {}
        self.cache_ttl = 3600  # 1 hour

    def _download_geoip_db(self) -> str:
        """Download GeoLite2 database if not present"""
        # Placeholder - in production you'd download from MaxMind
        # For now, return a default path
        db_path = "/tmp/GeoLite2-City.mmdb"

        # Create placeholder database info
        logger.warning(
            "GeoIP database not configured. "
            "For production use, download GeoLite2-City.mmdb from MaxMind"
        )

        return db_path

    async def initialize(self):
        """Initialize GeoIP resolver"""
        try:
            if Path(self.geoip_db_path).exists():
                self.geoip_reader = geoip2.database.Reader(self.geoip_db_path)
                logger.info(f"GeoIP database loaded: {self.geoip_db_path}")
            else:
                logger.warning("GeoIP database not found - using fallback methods")
        except Exception as e:
            logger.error(f"Failed to initialize GeoIP reader: {e}")

    async def resolve_target_location(self, target: str) -> Optional[GeoLocation]:
        """Resolve geographic location of target"""
        try:
            # First, resolve hostname to IP if needed
            target_ip = await self._resolve_hostname(target)
            if not target_ip:
                return None

            # Use GeoIP database if available
            if self.geoip_reader:
                return await self._geoip_lookup(target_ip)
            else:
                # Fallback to online services
                return await self._fallback_geo_lookup(target_ip)

        except Exception as e:
            logger.error(f"Error resolving location for {target}: {e}")
            return None

    async def _resolve_hostname(self, hostname: str) -> Optional[str]:
        """Resolve hostname to IP address"""
        try:
            # Check if it's already an IP
            ipaddress.ip_address(hostname)
            return hostname
        except ValueError:
            pass

        # Check cache first
        if hostname in self.dns_resolver_cache:
            cache_entry = self.dns_resolver_cache[hostname]
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                return cache_entry['ip']

        try:
            # Resolve DNS
            loop = asyncio.get_event_loop()
            addrinfo = await loop.getaddrinfo(hostname, None)

            if addrinfo:
                ip = addrinfo[0][4][0]

                # Cache result
                self.dns_resolver_cache[hostname] = {
                    'ip': ip,
                    'timestamp': time.time()
                }

                return ip

        except Exception as e:
            logger.error(f"DNS resolution failed for {hostname}: {e}")

        return None

    async def _geoip_lookup(self, ip_address: str) -> Optional[GeoLocation]:
        """Lookup location using GeoIP database"""
        try:
            response = self.geoip_reader.city(ip_address)

            return GeoLocation(
                latitude=float(response.location.latitude) if response.location.latitude else 0.0,
                longitude=float(response.location.longitude) if response.location.longitude else 0.0,
                country=response.country.name or "Unknown",
                country_code=response.country.iso_code or "XX",
                region=response.subdivisions.most_specific.name or "Unknown",
                city=response.city.name or "Unknown",
                timezone=response.location.time_zone,
                accuracy_radius=response.location.accuracy_radius
            )

        except geoip2.errors.AddressNotFoundError:
            logger.warning(f"IP address {ip_address} not found in GeoIP database")
            return None
        except Exception as e:
            logger.error(f"GeoIP lookup failed for {ip_address}: {e}")
            return None

    async def _fallback_geo_lookup(self, ip_address: str) -> Optional[GeoLocation]:
        """Fallback geographic lookup using online services"""
        try:
            # Use ip-api.com as fallback (free service)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://ip-api.com/json/{ip_address}",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        if data.get('status') == 'success':
                            return GeoLocation(
                                latitude=float(data.get('lat', 0)),
                                longitude=float(data.get('lon', 0)),
                                country=data.get('country', 'Unknown'),
                                country_code=data.get('countryCode', 'XX'),
                                region=data.get('regionName', 'Unknown'),
                                city=data.get('city', 'Unknown'),
                                timezone=data.get('timezone'),
                                isp=data.get('isp')
                            )

        except Exception as e:
            logger.error(f"Fallback geo lookup failed for {ip_address}: {e}")

        return None

    def close(self):
        """Close GeoIP database reader"""
        if self.geoip_reader:
            self.geoip_reader.close()

class RegionAnalyzer:
    """Analyzes targets and determines optimal regions"""

    def __init__(self, geoip_resolver: GeoIPResolver):
        self.geoip_resolver = geoip_resolver

        # Region mapping configurations
        self.country_to_target_region = {
            # North America
            'US': TargetRegion.NORTH_AMERICA,
            'CA': TargetRegion.NORTH_AMERICA,
            'MX': TargetRegion.NORTH_AMERICA,

            # South America
            'BR': TargetRegion.SOUTH_AMERICA,
            'AR': TargetRegion.SOUTH_AMERICA,
            'CL': TargetRegion.SOUTH_AMERICA,
            'CO': TargetRegion.SOUTH_AMERICA,
            'PE': TargetRegion.SOUTH_AMERICA,

            # Europe
            'DE': TargetRegion.EUROPE,
            'FR': TargetRegion.EUROPE,
            'GB': TargetRegion.EUROPE,
            'IT': TargetRegion.EUROPE,
            'ES': TargetRegion.EUROPE,
            'NL': TargetRegion.EUROPE,
            'PL': TargetRegion.EUROPE,
            'RU': TargetRegion.EUROPE,

            # Asia
            'CN': TargetRegion.ASIA,
            'JP': TargetRegion.ASIA,
            'KR': TargetRegion.ASIA,
            'IN': TargetRegion.ASIA,
            'SG': TargetRegion.ASIA,
            'HK': TargetRegion.ASIA,
            'TH': TargetRegion.ASIA,
            'VN': TargetRegion.ASIA,

            # Oceania
            'AU': TargetRegion.OCEANIA,
            'NZ': TargetRegion.OCEANIA,

            # Africa
            'ZA': TargetRegion.AFRICA,
            'NG': TargetRegion.AFRICA,
            'EG': TargetRegion.AFRICA,
            'KE': TargetRegion.AFRICA,

            # Middle East
            'AE': TargetRegion.MIDDLE_EAST,
            'SA': TargetRegion.MIDDLE_EAST,
            'IL': TargetRegion.MIDDLE_EAST,
            'TR': TargetRegion.MIDDLE_EAST,
        }

        self.target_region_to_proxy_regions = {
            TargetRegion.NORTH_AMERICA: [
                ProxyRegion.US_EAST,
                ProxyRegion.US_WEST,
                ProxyRegion.GLOBAL
            ],
            TargetRegion.SOUTH_AMERICA: [
                ProxyRegion.US_EAST,
                ProxyRegion.GLOBAL
            ],
            TargetRegion.EUROPE: [
                ProxyRegion.EU_CENTRAL,
                ProxyRegion.EU_WEST,
                ProxyRegion.GLOBAL
            ],
            TargetRegion.ASIA: [
                ProxyRegion.ASIA_PACIFIC,
                ProxyRegion.GLOBAL
            ],
            TargetRegion.OCEANIA: [
                ProxyRegion.ASIA_PACIFIC,
                ProxyRegion.GLOBAL
            ],
            TargetRegion.AFRICA: [
                ProxyRegion.EU_CENTRAL,
                ProxyRegion.GLOBAL
            ],
            TargetRegion.MIDDLE_EAST: [
                ProxyRegion.EU_CENTRAL,
                ProxyRegion.ASIA_PACIFIC,
                ProxyRegion.GLOBAL
            ],
            TargetRegion.UNKNOWN: [
                ProxyRegion.GLOBAL
            ]
        }

        self.compliance_mapping = {
            # GDPR countries
            'AT': [ComplianceRegion.GDPR_EU],
            'BE': [ComplianceRegion.GDPR_EU],
            'BG': [ComplianceRegion.GDPR_EU],
            'HR': [ComplianceRegion.GDPR_EU],
            'CY': [ComplianceRegion.GDPR_EU],
            'CZ': [ComplianceRegion.GDPR_EU],
            'DK': [ComplianceRegion.GDPR_EU],
            'EE': [ComplianceRegion.GDPR_EU],
            'FI': [ComplianceRegion.GDPR_EU],
            'FR': [ComplianceRegion.GDPR_EU],
            'DE': [ComplianceRegion.GDPR_EU],
            'GR': [ComplianceRegion.GDPR_EU],
            'HU': [ComplianceRegion.GDPR_EU],
            'IE': [ComplianceRegion.GDPR_EU],
            'IT': [ComplianceRegion.GDPR_EU],
            'LV': [ComplianceRegion.GDPR_EU],
            'LT': [ComplianceRegion.GDPR_EU],
            'LU': [ComplianceRegion.GDPR_EU],
            'MT': [ComplianceRegion.GDPR_EU],
            'NL': [ComplianceRegion.GDPR_EU],
            'PL': [ComplianceRegion.GDPR_EU],
            'PT': [ComplianceRegion.GDPR_EU],
            'RO': [ComplianceRegion.GDPR_EU],
            'SK': [ComplianceRegion.GDPR_EU],
            'SI': [ComplianceRegion.GDPR_EU],
            'ES': [ComplianceRegion.GDPR_EU],
            'SE': [ComplianceRegion.GDPR_EU],

            # Other compliance regions
            'US': [ComplianceRegion.CCPA_US],
            'CA': [ComplianceRegion.PIPEDA_CA],
            'BR': [ComplianceRegion.LGPD_BR],
            'SG': [ComplianceRegion.PDPA_SG],
            'GB': [ComplianceRegion.DPA_UK],
        }

    async def analyze_target(self, target: str) -> TargetAnalysis:
        """Analyze target and determine optimal routing"""
        analysis = TargetAnalysis(target_host=target)

        try:
            # Resolve geographic location
            geo_location = await self.geoip_resolver.resolve_target_location(target)
            if geo_location:
                analysis.geo_location = geo_location
                analysis.target_ip = await self.geoip_resolver._resolve_hostname(target)

                # Determine target region
                analysis.target_region = self.country_to_target_region.get(
                    geo_location.country_code, TargetRegion.UNKNOWN
                )

                # Determine compliance requirements
                analysis.compliance_regions = self.compliance_mapping.get(
                    geo_location.country_code, [ComplianceRegion.STANDARD]
                )

                # Determine optimal proxy regions
                analysis.optimal_proxy_regions = self.target_region_to_proxy_regions.get(
                    analysis.target_region, [ProxyRegion.GLOBAL]
                )

            # Detect CDN and load balancing
            await self._detect_infrastructure(analysis)

            logger.info(
                f"Target analysis completed for {target}: "
                f"region={analysis.target_region.value}, "
                f"optimal_regions={[r.value for r in analysis.optimal_proxy_regions]}"
            )

        except Exception as e:
            logger.error(f"Error analyzing target {target}: {e}")

        return analysis

    async def _detect_infrastructure(self, analysis: TargetAnalysis):
        """Detect CDN, load balancers, and geo-blocking"""
        try:
            # Multiple DNS queries to detect load balancing
            ips = set()
            for _ in range(5):
                ip = await self.geoip_resolver._resolve_hostname(analysis.target_host)
                if ip:
                    ips.add(ip)
                await asyncio.sleep(0.1)

            if len(ips) > 1:
                analysis.load_balancer_detected = True
                logger.info(f"Load balancer detected for {analysis.target_host}: {len(ips)} IPs")

            # Check for common CDN patterns
            cdn_indicators = [
                'cloudflare', 'cloudfront', 'fastly', 'akamai', 'maxcdn',
                'keycdn', 'jsdelivr', 'cdnjs', 'bootstrapcdn'
            ]

            if any(indicator in analysis.target_host.lower() for indicator in cdn_indicators):
                analysis.cdn_detected = True
                logger.info(f"CDN detected for {analysis.target_host}")

            # Basic geo-blocking detection (would need actual testing)
            # For now, mark high-risk regions
            if analysis.geo_location and analysis.geo_location.country_code in ['CN', 'RU', 'IR', 'KP']:
                analysis.geo_blocking_detected = True

        except Exception as e:
            logger.error(f"Error detecting infrastructure for {analysis.target_host}: {e}")

class LatencyMonitor:
    """Monitors and tracks latency between regions"""

    def __init__(self, redis_pool):
        self.redis_pool = redis_pool
        self.latency_cache = {}
        self.measurement_interval = 300  # 5 minutes
        self.test_endpoints = {
            ProxyRegion.US_EAST: [
                "ping.us-east-1.amazonaws.com",
                "ping.nyc1.digitalocean.com"
            ],
            ProxyRegion.US_WEST: [
                "ping.us-west-1.amazonaws.com",
                "ping.sfo2.digitalocean.com"
            ],
            ProxyRegion.EU_CENTRAL: [
                "ping.eu-central-1.amazonaws.com",
                "ping.fra1.digitalocean.com"
            ],
            ProxyRegion.EU_WEST: [
                "ping.eu-west-1.amazonaws.com",
                "ping.lon1.digitalocean.com"
            ],
            ProxyRegion.ASIA_PACIFIC: [
                "ping.ap-southeast-1.amazonaws.com",
                "ping.sgp1.digitalocean.com"
            ]
        }

    async def start_monitoring(self):
        """Start continuous latency monitoring"""
        logger.info("Starting latency monitoring")

        while True:
            try:
                await self._measure_all_regions()
                await asyncio.sleep(self.measurement_interval)
            except Exception as e:
                logger.error(f"Latency monitoring error: {e}")
                await asyncio.sleep(60)

    async def _measure_all_regions(self):
        """Measure latency to all regions"""
        measurement_tasks = []

        for region, endpoints in self.test_endpoints.items():
            task = asyncio.create_task(self._measure_region_latency(region, endpoints))
            measurement_tasks.append(task)

        results = await asyncio.gather(*measurement_tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, RegionLatency):
                region = list(self.test_endpoints.keys())[i]
                self.latency_cache[region] = result

                # Store in Redis
                await self.redis_pool.hset(
                    f"latency:{region.value}",
                    mapping={
                        'average_latency': str(result.average_latency),
                        'min_latency': str(result.min_latency),
                        'max_latency': str(result.max_latency),
                        'sample_count': str(result.sample_count),
                        'last_updated': result.last_updated.isoformat(),
                        'reliability_score': str(result.reliability_score)
                    }
                )

    async def _measure_region_latency(self, region: ProxyRegion, endpoints: List[str]) -> RegionLatency:
        """Measure latency to a specific region"""
        latencies = []

        for endpoint in endpoints:
            try:
                start_time = time.time()

                # Simple HTTP ping
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(
                            f"http://{endpoint}",
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            latency = (time.time() - start_time) * 1000  # ms
                            latencies.append(latency)
                    except:
                        # Try ICMP ping via system call
                        latency = await self._system_ping(endpoint)
                        if latency:
                            latencies.append(latency)

            except Exception as e:
                logger.debug(f"Latency measurement failed for {endpoint}: {e}")

        # Calculate statistics
        if latencies:
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            reliability_score = len(latencies) / len(endpoints)
        else:
            avg_latency = float('inf')
            min_latency = float('inf')
            max_latency = float('inf')
            reliability_score = 0.0

        return RegionLatency(
            region=region,
            average_latency=avg_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            sample_count=len(latencies),
            last_updated=datetime.now(),
            reliability_score=reliability_score
        )

    async def _system_ping(self, host: str) -> Optional[float]:
        """Perform system ping and extract latency"""
        try:
            import subprocess

            # Run ping command
            result = subprocess.run(
                ['ping', '-c', '1', '-W', '2000', host],
                capture_output=True,
                text=True,
                timeout=3
            )

            if result.returncode == 0:
                # Extract latency from ping output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'time=' in line:
                        time_part = line.split('time=')[1].split()[0]
                        return float(time_part)

        except Exception as e:
            logger.debug(f"System ping failed for {host}: {e}")

        return None

    async def get_region_latency(self, region: ProxyRegion) -> Optional[RegionLatency]:
        """Get cached latency for a region"""
        # Try cache first
        if region in self.latency_cache:
            return self.latency_cache[region]

        # Try Redis
        try:
            data = await self.redis_pool.hgetall(f"latency:{region.value}")
            if data:
                return RegionLatency(
                    region=region,
                    average_latency=float(data['average_latency']),
                    min_latency=float(data['min_latency']),
                    max_latency=float(data['max_latency']),
                    sample_count=int(data['sample_count']),
                    last_updated=datetime.fromisoformat(data['last_updated']),
                    reliability_score=float(data['reliability_score'])
                )
        except Exception as e:
            logger.error(f"Error retrieving latency for {region}: {e}")

        return None

class GeoRouter:
    """Main geographic distribution and routing manager"""

    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 postgres_url: str = "postgresql://localhost:5432/osint",
                 geoip_db_path: Optional[str] = None):

        self.redis_url = redis_url
        self.postgres_url = postgres_url

        # Core components
        self.geoip_resolver = GeoIPResolver(geoip_db_path)
        self.region_analyzer = RegionAnalyzer(self.geoip_resolver)
        self.latency_monitor = None  # Initialized after Redis connection

        # Storage
        self.redis_pool = None
        self.postgres_pool = None

        # Caches
        self.target_analysis_cache = {}
        self.cache_ttl = 3600  # 1 hour

        # Configuration
        self.region_preferences = {
            # Default regional preferences for different operation types
            'osint': [ProxyRegion.GLOBAL, ProxyRegion.US_EAST, ProxyRegion.EU_CENTRAL],
            'social_media': [ProxyRegion.US_EAST, ProxyRegion.US_WEST, ProxyRegion.GLOBAL],
            'business_intel': [ProxyRegion.EU_CENTRAL, ProxyRegion.US_EAST, ProxyRegion.ASIA_PACIFIC],
            'academic': [ProxyRegion.GLOBAL, ProxyRegion.EU_CENTRAL, ProxyRegion.US_EAST],
        }

        logger.info("GeoRouter initialized")

    async def initialize(self):
        """Initialize geo router and all components"""
        logger.info("Initializing GeoRouter...")

        try:
            # Initialize storage
            await self._initialize_storage()

            # Initialize GeoIP resolver
            await self.geoip_resolver.initialize()

            # Initialize latency monitor
            self.latency_monitor = LatencyMonitor(self.redis_pool)

            # Setup database schema
            await self._setup_database_schema()

            # Start background tasks
            asyncio.create_task(self.latency_monitor.start_monitoring())

            logger.info("GeoRouter initialization completed")

        except Exception as e:
            logger.error(f"GeoRouter initialization failed: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down GeoRouter...")

        # Close GeoIP resolver
        self.geoip_resolver.close()

        # Close storage connections
        if self.redis_pool:
            await self.redis_pool.close()

        if self.postgres_pool:
            await self.postgres_pool.close()

        logger.info("GeoRouter shutdown completed")

    async def _initialize_storage(self):
        """Initialize storage connections"""
        # Redis
        self.redis_pool = aioredis.from_url(
            self.redis_url,
            max_connections=20,
            retry_on_timeout=True
        )
        await self.redis_pool.ping()
        logger.info("Redis connection established")

        # PostgreSQL
        self.postgres_pool = await asyncpg.create_pool(
            self.postgres_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        logger.info("PostgreSQL connection pool established")

    async def _setup_database_schema(self):
        """Setup database schema"""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS target_analysis (
            id SERIAL PRIMARY KEY,
            target_host VARCHAR(255) NOT NULL,
            target_ip INET,
            country VARCHAR(100),
            country_code VARCHAR(10),
            region VARCHAR(100),
            city VARCHAR(100),
            latitude REAL,
            longitude REAL,
            target_region VARCHAR(50),
            optimal_proxy_regions TEXT[],
            compliance_regions TEXT[],
            cdn_detected BOOLEAN DEFAULT FALSE,
            load_balancer_detected BOOLEAN DEFAULT FALSE,
            geo_blocking_detected BOOLEAN DEFAULT FALSE,
            analysis_timestamp TIMESTAMP DEFAULT NOW(),
            cache_expires TIMESTAMP,
            UNIQUE(target_host)
        );

        CREATE TABLE IF NOT EXISTS latency_measurements (
            id SERIAL PRIMARY KEY,
            region VARCHAR(50) NOT NULL,
            average_latency REAL NOT NULL,
            min_latency REAL NOT NULL,
            max_latency REAL NOT NULL,
            sample_count INTEGER NOT NULL,
            reliability_score REAL NOT NULL,
            measurement_timestamp TIMESTAMP DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_target_analysis_host ON target_analysis(target_host);
        CREATE INDEX IF NOT EXISTS idx_target_analysis_expires ON target_analysis(cache_expires);
        CREATE INDEX IF NOT EXISTS idx_latency_measurements_region ON latency_measurements(region);
        CREATE INDEX IF NOT EXISTS idx_latency_measurements_timestamp ON latency_measurements(measurement_timestamp);
        """

        async with self.postgres_pool.acquire() as conn:
            await conn.execute(schema_sql)

        logger.info("Database schema setup completed")

    async def get_optimal_regions(self,
                                 target: str,
                                 operation_type: str = 'osint',
                                 force_analysis: bool = False) -> List[ProxyRegion]:
        """Get optimal proxy regions for a target"""

        try:
            # Check cache first unless forced
            if not force_analysis:
                cached_analysis = await self._get_cached_analysis(target)
                if cached_analysis and cached_analysis.optimal_proxy_regions:
                    return cached_analysis.optimal_proxy_regions

            # Perform fresh analysis
            analysis = await self.region_analyzer.analyze_target(target)

            # Store analysis
            await self._store_analysis(analysis)

            # Enhance with latency information
            optimal_regions = await self._optimize_with_latency(
                analysis.optimal_proxy_regions
            )

            # Apply operation-specific preferences
            final_regions = self._apply_operation_preferences(
                optimal_regions, operation_type
            )

            logger.info(
                f"Optimal regions for {target} ({operation_type}): "
                f"{[r.value for r in final_regions]}"
            )

            return final_regions

        except Exception as e:
            logger.error(f"Error getting optimal regions for {target}: {e}")
            # Return default regions
            return self.region_preferences.get(operation_type, [ProxyRegion.GLOBAL])

    async def _get_cached_analysis(self, target: str) -> Optional[TargetAnalysis]:
        """Get cached target analysis"""
        try:
            # Check in-memory cache
            if target in self.target_analysis_cache:
                analysis = self.target_analysis_cache[target]
                if (datetime.now() - analysis.analysis_timestamp).total_seconds() < self.cache_ttl:
                    return analysis

            # Check database cache
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM target_analysis
                    WHERE target_host = $1 AND cache_expires > NOW()
                """, target)

                if row:
                    analysis = TargetAnalysis(
                        target_host=row['target_host'],
                        target_ip=str(row['target_ip']) if row['target_ip'] else None,
                        target_region=TargetRegion(row['target_region']) if row['target_region'] else TargetRegion.UNKNOWN,
                        optimal_proxy_regions=[ProxyRegion(r) for r in row['optimal_proxy_regions']] if row['optimal_proxy_regions'] else [],
                        compliance_regions=[ComplianceRegion(r) for r in row['compliance_regions']] if row['compliance_regions'] else [],
                        cdn_detected=row['cdn_detected'],
                        load_balancer_detected=row['load_balancer_detected'],
                        geo_blocking_detected=row['geo_blocking_detected'],
                        analysis_timestamp=row['analysis_timestamp']
                    )

                    if row['latitude'] and row['longitude']:
                        analysis.geo_location = GeoLocation(
                            latitude=row['latitude'],
                            longitude=row['longitude'],
                            country=row['country'] or "Unknown",
                            country_code=row['country_code'] or "XX",
                            region=row['region'] or "Unknown",
                            city=row['city'] or "Unknown"
                        )

                    # Update in-memory cache
                    self.target_analysis_cache[target] = analysis
                    return analysis

        except Exception as e:
            logger.error(f"Error retrieving cached analysis for {target}: {e}")

        return None

    async def _store_analysis(self, analysis: TargetAnalysis):
        """Store target analysis in cache and database"""
        try:
            # Update in-memory cache
            self.target_analysis_cache[analysis.target_host] = analysis

            # Store in database
            cache_expires = datetime.now() + timedelta(seconds=self.cache_ttl)

            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO target_analysis
                    (target_host, target_ip, country, country_code, region, city,
                     latitude, longitude, target_region, optimal_proxy_regions,
                     compliance_regions, cdn_detected, load_balancer_detected,
                     geo_blocking_detected, analysis_timestamp, cache_expires)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (target_host) DO UPDATE SET
                        target_ip = EXCLUDED.target_ip,
                        country = EXCLUDED.country,
                        country_code = EXCLUDED.country_code,
                        region = EXCLUDED.region,
                        city = EXCLUDED.city,
                        latitude = EXCLUDED.latitude,
                        longitude = EXCLUDED.longitude,
                        target_region = EXCLUDED.target_region,
                        optimal_proxy_regions = EXCLUDED.optimal_proxy_regions,
                        compliance_regions = EXCLUDED.compliance_regions,
                        cdn_detected = EXCLUDED.cdn_detected,
                        load_balancer_detected = EXCLUDED.load_balancer_detected,
                        geo_blocking_detected = EXCLUDED.geo_blocking_detected,
                        analysis_timestamp = EXCLUDED.analysis_timestamp,
                        cache_expires = EXCLUDED.cache_expires
                """,
                    analysis.target_host,
                    analysis.target_ip,
                    analysis.geo_location.country if analysis.geo_location else None,
                    analysis.geo_location.country_code if analysis.geo_location else None,
                    analysis.geo_location.region if analysis.geo_location else None,
                    analysis.geo_location.city if analysis.geo_location else None,
                    analysis.geo_location.latitude if analysis.geo_location else None,
                    analysis.geo_location.longitude if analysis.geo_location else None,
                    analysis.target_region.value,
                    [r.value for r in analysis.optimal_proxy_regions],
                    [r.value for r in analysis.compliance_regions],
                    analysis.cdn_detected,
                    analysis.load_balancer_detected,
                    analysis.geo_blocking_detected,
                    analysis.analysis_timestamp,
                    cache_expires
                )

        except Exception as e:
            logger.error(f"Error storing analysis for {analysis.target_host}: {e}")

    async def _optimize_with_latency(self, regions: List[ProxyRegion]) -> List[ProxyRegion]:
        """Optimize region selection with latency information"""
        if not self.latency_monitor or not regions:
            return regions

        try:
            # Get latency information for each region
            region_latencies = []
            for region in regions:
                latency_info = await self.latency_monitor.get_region_latency(region)
                if latency_info:
                    region_latencies.append((region, latency_info))

            # Sort by latency (lower is better) and reliability
            region_latencies.sort(
                key=lambda x: (x[1].average_latency, -x[1].reliability_score)
            )

            # Return sorted regions
            optimized_regions = [region for region, _ in region_latencies]

            # Add any regions that didn't have latency data at the end
            for region in regions:
                if region not in optimized_regions:
                    optimized_regions.append(region)

            return optimized_regions

        except Exception as e:
            logger.error(f"Error optimizing with latency: {e}")
            return regions

    def _apply_operation_preferences(self,
                                   regions: List[ProxyRegion],
                                   operation_type: str) -> List[ProxyRegion]:
        """Apply operation-specific region preferences"""

        # Get preferences for operation type
        preferred_regions = self.region_preferences.get(operation_type, [])

        # Merge with analyzed regions, maintaining preference order
        final_regions = []

        # First add preferred regions that are in the analyzed regions
        for pref_region in preferred_regions:
            if pref_region in regions and pref_region not in final_regions:
                final_regions.append(pref_region)

        # Then add remaining analyzed regions
        for region in regions:
            if region not in final_regions:
                final_regions.append(region)

        # Finally add remaining preferred regions as fallbacks
        for pref_region in preferred_regions:
            if pref_region not in final_regions:
                final_regions.append(pref_region)

        return final_regions

    async def get_compliance_requirements(self, target: str) -> List[ComplianceRegion]:
        """Get compliance requirements for a target"""
        try:
            # Get or create analysis
            analysis = await self._get_cached_analysis(target)
            if not analysis:
                analysis = await self.region_analyzer.analyze_target(target)
                await self._store_analysis(analysis)

            return analysis.compliance_regions

        except Exception as e:
            logger.error(f"Error getting compliance requirements for {target}: {e}")
            return [ComplianceRegion.STANDARD]

    async def get_geo_statistics(self) -> Dict[str, Any]:
        """Get geographic routing statistics"""
        try:
            stats = {
                'cache_size': len(self.target_analysis_cache),
                'region_distribution': defaultdict(int),
                'compliance_distribution': defaultdict(int),
                'latency_stats': {},
                'detection_stats': {
                    'cdn_detected': 0,
                    'load_balancer_detected': 0,
                    'geo_blocking_detected': 0
                }
            }

            # Analyze cached data
            for analysis in self.target_analysis_cache.values():
                # Region distribution
                for region in analysis.optimal_proxy_regions:
                    stats['region_distribution'][region.value] += 1

                # Compliance distribution
                for compliance in analysis.compliance_regions:
                    stats['compliance_distribution'][compliance.value] += 1

                # Detection stats
                if analysis.cdn_detected:
                    stats['detection_stats']['cdn_detected'] += 1
                if analysis.load_balancer_detected:
                    stats['detection_stats']['load_balancer_detected'] += 1
                if analysis.geo_blocking_detected:
                    stats['detection_stats']['geo_blocking_detected'] += 1

            # Get latency statistics
            if self.latency_monitor:
                for region in ProxyRegion:
                    latency_info = await self.latency_monitor.get_region_latency(region)
                    if latency_info:
                        stats['latency_stats'][region.value] = {
                            'average_latency': latency_info.average_latency,
                            'reliability_score': latency_info.reliability_score,
                            'last_updated': latency_info.last_updated.isoformat()
                        }

            return stats

        except Exception as e:
            logger.error(f"Error getting geo statistics: {e}")
            return {}

# Factory function
async def create_geo_router(
    redis_url: str = "redis://localhost:6379",
    postgres_url: str = "postgresql://localhost:5432/osint",
    geoip_db_path: Optional[str] = None
) -> GeoRouter:
    """Create and initialize a GeoRouter instance"""
    router = GeoRouter(redis_url, postgres_url, geoip_db_path)
    await router.initialize()
    return router

if __name__ == "__main__":
    # Example usage
    async def main():
        router = await create_geo_router()

        try:
            # Test target analysis
            targets = [
                "google.com",
                "facebook.com",
                "baidu.com",
                "yandex.ru"
            ]

            for target in targets:
                optimal_regions = await router.get_optimal_regions(target, "osint")
                compliance = await router.get_compliance_requirements(target)

                print(f"\nTarget: {target}")
                print(f"Optimal regions: {[r.value for r in optimal_regions]}")
                print(f"Compliance: {[c.value for c in compliance]}")

            # Get statistics
            stats = await router.get_geo_statistics()
            print(f"\nGeo statistics: {json.dumps(stats, indent=2, default=str)}")

        finally:
            await router.shutdown()

    asyncio.run(main())