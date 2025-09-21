#!/usr/bin/env python3
"""
Alternative Market Intelligence Platform - Phase 7 BEV OSINT Framework
Package initialization and main orchestrator for all alternative market components.
"""

__version__ = "1.0.0"
__author__ = "BEV OSINT Framework"
__description__ = "Alternative Market Intelligence Platform for decentralized market analysis"

from .dm_crawler import (
    MarketplaceCrawler,
    TorProxyManager,
    RateLimiter,
    MarketplaceConfig,
    VendorListing,
    ProductListing,
    PriceTrend
)

from .crypto_analyzer import (
    CryptocurrencyIntelligenceEngine,
    BlockchainAPI,
    WalletClusteringEngine,
    MixingServiceDetector,
    ExchangeFlowAnalyzer,
    Transaction,
    WalletCluster,
    MixingService,
    ExchangeFlow
)

from .reputation_analyzer import (
    VendorReputationFramework,
    ReputationSourceManager,
    EscrowMonitor,
    SentimentAnalyzer,
    FraudDetector,
    VendorReputation,
    EscrowTransaction,
    DisputeRecord,
    FeedbackAnalysis,
    FraudPattern
)

from .economics_processor import (
    MarketEconomicsProcessor,
    SupplyDemandAnalyzer,
    VolatilityAnalyzer,
    ManipulationDetector,
    EconomicForecaster,
    ArbitrageDetector,
    MarketDataPoint,
    SupplyDemandModel,
    PriceVolatilityAnalysis,
    MarketManipulationAlert,
    EconomicForecast,
    ArbitrageOpportunity
)

__all__ = [
    # DM Crawler
    'MarketplaceCrawler',
    'TorProxyManager',
    'RateLimiter',
    'MarketplaceConfig',
    'VendorListing',
    'ProductListing',
    'PriceTrend',

    # Crypto Analyzer
    'CryptocurrencyIntelligenceEngine',
    'BlockchainAPI',
    'WalletClusteringEngine',
    'MixingServiceDetector',
    'ExchangeFlowAnalyzer',
    'Transaction',
    'WalletCluster',
    'MixingService',
    'ExchangeFlow',

    # Reputation Analyzer
    'VendorReputationFramework',
    'ReputationSourceManager',
    'EscrowMonitor',
    'SentimentAnalyzer',
    'FraudDetector',
    'VendorReputation',
    'EscrowTransaction',
    'DisputeRecord',
    'FeedbackAnalysis',
    'FraudPattern',

    # Economics Processor
    'MarketEconomicsProcessor',
    'SupplyDemandAnalyzer',
    'VolatilityAnalyzer',
    'ManipulationDetector',
    'EconomicForecaster',
    'ArbitrageDetector',
    'MarketDataPoint',
    'SupplyDemandModel',
    'PriceVolatilityAnalysis',
    'MarketManipulationAlert',
    'EconomicForecast',
    'ArbitrageOpportunity'
]


class AlternativeMarketIntelligencePlatform:
    """
    Main orchestrator for the Alternative Market Intelligence Platform.
    Coordinates all four components for comprehensive market analysis.
    """

    def __init__(self, config: dict):
        """
        Initialize the platform with configuration.

        Args:
            config: Dictionary containing database, Redis, Kafka, and component configurations
        """
        self.config = config
        self.components = {}

    async def initialize(self):
        """Initialize all platform components"""

        # Initialize DM Crawler
        self.components['dm_crawler'] = MarketplaceCrawler(
            self.config['database'],
            self.config['redis'],
            self.config['kafka']
        )
        await self.components['dm_crawler'].initialize()

        # Initialize Crypto Analyzer
        self.components['crypto_analyzer'] = CryptocurrencyIntelligenceEngine(
            self.config['database'],
            self.config['redis'],
            self.config['kafka']
        )
        await self.components['crypto_analyzer'].initialize()

        # Initialize Reputation Analyzer
        self.components['reputation_analyzer'] = VendorReputationFramework(
            self.config['database'],
            self.config['redis'],
            self.config['kafka']
        )
        await self.components['reputation_analyzer'].initialize()

        # Initialize Economics Processor
        self.components['economics_processor'] = MarketEconomicsProcessor(
            self.config['database'],
            self.config['redis'],
            self.config['kafka']
        )
        await self.components['economics_processor'].initialize()

    async def start_full_analysis(self, marketplaces: list):
        """
        Start comprehensive analysis across all marketplaces.

        Args:
            marketplaces: List of marketplace names to analyze
        """

        for marketplace in marketplaces:
            # Start crawling
            await self.components['dm_crawler'].crawl_marketplace(marketplace, 'full')

            # Monitor crypto transactions related to marketplace
            # (This would require identifying marketplace-related addresses)

            # Analyze vendor reputations
            # (This would analyze vendors discovered during crawling)

            # Process market economics
            # (This would use data gathered from crawling)

    async def cleanup(self):
        """Cleanup all platform components"""

        for component in self.components.values():
            await component.cleanup()


# Example configuration template
EXAMPLE_CONFIG = {
    'database': {
        'host': 'localhost',
        'port': 5432,
        'database': 'bev_osint',
        'user': 'bev_user',
        'password': 'secure_password'
    },
    'redis': {
        'host': 'localhost',
        'port': 6379
    },
    'kafka': {
        'bootstrap_servers': ['localhost:9092']
    },
    'tor_proxy': {
        'host': '172.30.0.17',
        'port': 9050
    },
    'marketplace_configs': {
        # Marketplace-specific configurations would go here
    }
}


async def main():
    """Example usage of the Alternative Market Intelligence Platform"""

    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize platform
    platform = AlternativeMarketIntelligencePlatform(EXAMPLE_CONFIG)

    try:
        await platform.initialize()
        logger.info("Alternative Market Intelligence Platform initialized successfully")

        # Start analysis for example marketplaces
        marketplaces = ['example_market_1', 'example_market_2']
        await platform.start_full_analysis(marketplaces)

    except KeyboardInterrupt:
        logger.info("Shutting down platform...")
    except Exception as e:
        logger.error(f"Platform error: {e}")
    finally:
        await platform.cleanup()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())