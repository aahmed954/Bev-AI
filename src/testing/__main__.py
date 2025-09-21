"""
Main entry point for the BEV Chaos Engineering System.

This module starts the comprehensive chaos engineering system including:
- Chaos engineering orchestration
- Fault injection management
- Resilience testing framework
- Scenario library management
- REST API server for integration

Usage:
    python -m src.testing.chaos_engineer
    python -m src.testing [options]

Author: BEV Infrastructure Team
Version: 1.0.0
"""

import asyncio
import logging
import argparse
import signal
import sys
from typing import Optional

from .chaos_engineer import ChaosEngineer
from .fault_injector import FaultInjectionManager
from .resilience_tester import ResilienceTester
from .scenario_library import ScenarioLibrary
from .chaos_api import ChaosEngineeringAPI


class ChaosEngineeringService:
    """
    Main service that orchestrates all chaos engineering components.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "/app/config/chaos_engineer.yaml"

        # Components
        self.chaos_engineer: Optional[ChaosEngineer] = None
        self.fault_manager: Optional[FaultInjectionManager] = None
        self.resilience_tester: Optional[ResilienceTester] = None
        self.scenario_library: Optional[ScenarioLibrary] = None
        self.api_server: Optional[ChaosEngineeringAPI] = None

        # Control
        self.shutdown_event = asyncio.Event()

        # Logging
        self.logger = logging.getLogger("chaos_service")

    async def initialize(self):
        """Initialize all components."""
        self.logger.info("Initializing BEV Chaos Engineering System...")

        try:
            # Initialize chaos engineer
            self.chaos_engineer = ChaosEngineer(config_path=self.config_path)
            await self.chaos_engineer.initialize()
            self.logger.info("Chaos Engineer initialized")

            # Initialize fault injection manager
            self.fault_manager = FaultInjectionManager(
                docker_client=self.chaos_engineer.docker_client,
                redis_client=self.chaos_engineer.redis_client
            )
            self.logger.info("Fault Injection Manager initialized")

            # Initialize resilience tester
            self.resilience_tester = ResilienceTester()
            await self.resilience_tester.initialize()
            self.logger.info("Resilience Tester initialized")

            # Initialize scenario library
            self.scenario_library = ScenarioLibrary()
            self.logger.info("Scenario Library initialized")

            # Initialize API server
            self.api_server = ChaosEngineeringAPI(
                chaos_engineer=self.chaos_engineer,
                resilience_tester=self.resilience_tester,
                scenario_library=self.scenario_library,
                fault_manager=self.fault_manager
            )
            self.logger.info("API Server initialized")

            self.logger.info("BEV Chaos Engineering System initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise

    async def start(self):
        """Start all services."""
        self.logger.info("Starting BEV Chaos Engineering System...")

        try:
            # Start background monitoring tasks
            background_tasks = [
                asyncio.create_task(self._monitor_system_health()),
                asyncio.create_task(self._periodic_safety_checks()),
                asyncio.create_task(self.api_server.start_server())
            ]

            self.logger.info("BEV Chaos Engineering System started successfully")
            self.logger.info("API Server available at: http://0.0.0.0:8080")
            self.logger.info("WebSocket monitoring available at: ws://0.0.0.0:8080/ws")

            # Wait for shutdown signal
            await self.shutdown_event.wait()

            # Cancel background tasks
            for task in background_tasks:
                task.cancel()

            # Wait for tasks to complete
            await asyncio.gather(*background_tasks, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"Error running system: {e}")
            raise

    async def shutdown(self):
        """Shutdown all services gracefully."""
        self.logger.info("Shutting down BEV Chaos Engineering System...")

        try:
            # Signal shutdown
            self.shutdown_event.set()

            # Emergency stop all chaos activities
            if self.chaos_engineer:
                for experiment_name in list(self.chaos_engineer.active_experiments.keys()):
                    await self.chaos_engineer.stop_experiment(experiment_name, emergency=True)

            if self.fault_manager:
                await self.fault_manager.remove_all_faults()

            if self.resilience_tester:
                for test_name in list(self.resilience_tester.active_tests.keys()):
                    await self.resilience_tester.stop_test(test_name)

            # Shutdown components
            if self.chaos_engineer:
                await self.chaos_engineer.shutdown()

            if self.resilience_tester:
                await self.resilience_tester.shutdown()

            self.logger.info("BEV Chaos Engineering System shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def _monitor_system_health(self):
        """Monitor overall system health."""
        while not self.shutdown_event.is_set():
            try:
                # Check component health
                if self.chaos_engineer:
                    chaos_status = self.chaos_engineer.get_system_status()
                    if not chaos_status.get('safety_monitor_active', False):
                        self.logger.warning("Chaos engineer safety monitor is inactive")

                if self.resilience_tester:
                    resilience_status = self.resilience_tester.get_system_status()
                    # Log status if needed

                if self.fault_manager:
                    fault_status = await self.fault_manager.get_all_faults_status()
                    active_faults = fault_status.get('active_faults', 0)
                    if active_faults > 10:  # Threshold for too many faults
                        self.logger.warning(f"High number of active faults: {active_faults}")

                # Sleep between health checks
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)

    async def _periodic_safety_checks(self):
        """Perform periodic safety checks."""
        while not self.shutdown_event.is_set():
            try:
                # Validate all active faults are working correctly
                if self.fault_manager:
                    validation_results = await self.fault_manager.validate_all_active_faults()
                    invalid_faults = [
                        fault_id for fault_id, valid in validation_results.items()
                        if not valid
                    ]

                    if invalid_faults:
                        self.logger.warning(f"Invalid faults detected: {invalid_faults}")
                        # Optionally remove invalid faults
                        for fault_id in invalid_faults:
                            await self.fault_manager.remove_fault(fault_id)

                # Check system resource usage (if available)
                # This could integrate with monitoring systems

                # Sleep between safety checks
                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Safety check error: {e}")
                await asyncio.sleep(300)


def setup_signal_handlers(service: ChaosEngineeringService):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(service.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/app/logs/chaos_engineering.log', mode='a')
        ]
    )

    # Set specific logger levels
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('docker').setLevel(logging.WARNING)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='BEV Chaos Engineering System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.testing                    # Start with default configuration
  python -m src.testing --config custom.yaml  # Start with custom config
  python -m src.testing --log-level DEBUG     # Enable debug logging
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='/app/config/chaos_engineer.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    parser.add_argument(
        '--api-port',
        type=int,
        default=8080,
        help='API server port'
    )

    parser.add_argument(
        '--safety-mode',
        action='store_true',
        help='Enable enhanced safety mode'
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)

    logger = logging.getLogger("main")
    logger.info("Starting BEV Chaos Engineering System")

    try:
        # Create and initialize service
        service = ChaosEngineeringService(config_path=args.config)

        # Setup signal handlers
        setup_signal_handlers(service)

        # Initialize and start
        await service.initialize()
        await service.start()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("BEV Chaos Engineering System stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass