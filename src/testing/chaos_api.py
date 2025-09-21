"""
Chaos Engineering REST API for BEV OSINT Framework
=================================================

RESTful API interface for the chaos engineering system providing
endpoints for experiment management, fault injection, and system integration.

Features:
- RESTful API for chaos experiment management
- Real-time experiment monitoring and control
- Integration with auto-recovery and health monitoring systems
- Safety controls and emergency stop functionality
- Comprehensive logging and audit trails

Author: BEV Infrastructure Team
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import aiohttp
from aiohttp import web, WSMsgType
import aiohttp_cors
from aiohttp.web import Response, Request, WebSocketResponse
import aioredis
import weakref

from .chaos_engineer import ChaosEngineer, ExperimentConfig, FaultInjectionConfig, FaultType, SafetyLevel
from .resilience_tester import ResilienceTester, ResilienceTestConfig, ResilienceMetric
from .scenario_library import ScenarioLibrary, ChaosScenario, ScenarioCategory, ScenarioComplexity
from .fault_injector import FaultInjectionManager


class ChaosEngineeringAPI:
    """
    REST API server for chaos engineering system.
    """

    def __init__(self,
                 chaos_engineer: ChaosEngineer,
                 resilience_tester: ResilienceTester,
                 scenario_library: ScenarioLibrary,
                 fault_manager: FaultInjectionManager,
                 host: str = "0.0.0.0",
                 port: int = 8080):
        """
        Initialize the chaos engineering API server.

        Args:
            chaos_engineer: Main chaos engineering system
            resilience_tester: Resilience testing framework
            scenario_library: Scenario library for predefined tests
            fault_manager: Fault injection manager
            host: Server host address
            port: Server port
        """
        self.chaos_engineer = chaos_engineer
        self.resilience_tester = resilience_tester
        self.scenario_library = scenario_library
        self.fault_manager = fault_manager
        self.host = host
        self.port = port

        # Web application and WebSocket connections
        self.app: Optional[web.Application] = None
        self.websocket_connections: weakref.WeakSet = weakref.WeakSet()

        # Logging
        self.logger = logging.getLogger("chaos_api")

    async def create_app(self) -> web.Application:
        """Create and configure the web application."""
        self.app = web.Application()

        # Configure CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })

        # Health and status endpoints
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/status', self.system_status)
        self.app.router.add_get('/metrics', self.system_metrics)

        # Experiment management
        self.app.router.add_post('/experiments', self.create_experiment)
        self.app.router.add_get('/experiments', self.list_experiments)
        self.app.router.add_get('/experiments/{experiment_id}', self.get_experiment)
        self.app.router.add_post('/experiments/{experiment_id}/start', self.start_experiment)
        self.app.router.add_post('/experiments/{experiment_id}/stop', self.stop_experiment)
        self.app.router.add_delete('/experiments/{experiment_id}', self.delete_experiment)

        # Resilience testing
        self.app.router.add_post('/resilience-tests', self.create_resilience_test)
        self.app.router.add_get('/resilience-tests', self.list_resilience_tests)
        self.app.router.add_get('/resilience-tests/{test_id}', self.get_resilience_test)
        self.app.router.add_post('/resilience-tests/{test_id}/start', self.start_resilience_test)

        # Fault injection
        self.app.router.add_post('/faults', self.inject_fault)
        self.app.router.add_get('/faults', self.list_active_faults)
        self.app.router.add_get('/faults/{fault_id}', self.get_fault_status)
        self.app.router.add_delete('/faults/{fault_id}', self.remove_fault)
        self.app.router.add_delete('/faults', self.remove_all_faults)

        # Scenario management
        self.app.router.add_get('/scenarios', self.list_scenarios)
        self.app.router.add_get('/scenarios/{scenario_name}', self.get_scenario)
        self.app.router.add_post('/scenarios', self.create_scenario)
        self.app.router.add_get('/scenario-suites', self.list_scenario_suites)
        self.app.router.add_get('/scenario-suites/{suite_name}', self.get_scenario_suite)

        # Safety and emergency controls
        self.app.router.add_post('/emergency-stop', self.emergency_stop)
        self.app.router.add_post('/safety-check', self.safety_check)

        # WebSocket for real-time monitoring
        self.app.router.add_get('/ws', self.websocket_handler)

        # Integration endpoints
        self.app.router.add_post('/integration/auto-recovery/validate', self.validate_auto_recovery)
        self.app.router.add_post('/integration/health-monitor/metrics', self.collect_health_metrics)

        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)

        # Add middleware for request logging
        self.app.middlewares.append(self.logging_middleware)
        self.app.middlewares.append(self.error_middleware)

        return self.app

    async def logging_middleware(self, request: Request, handler):
        """Middleware for request logging."""
        start_time = time.time()

        try:
            response = await handler(request)
            duration = time.time() - start_time

            self.logger.info(
                f"{request.method} {request.path} - {response.status} - {duration:.3f}s"
            )

            return response
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"{request.method} {request.path} - ERROR: {str(e)} - {duration:.3f}s"
            )
            raise

    async def error_middleware(self, request: Request, handler):
        """Middleware for error handling."""
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Unhandled error in {request.path}: {str(e)}")
            return web.json_response(
                {'error': 'Internal server error', 'message': str(e)},
                status=500
            )

    async def health_check(self, request: Request) -> Response:
        """Health check endpoint."""
        try:
            # Check system components
            chaos_status = self.chaos_engineer.get_system_status()
            resilience_status = self.resilience_tester.get_system_status()
            fault_status = await self.fault_manager.get_all_faults_status()

            health_data = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'components': {
                    'chaos_engineer': {
                        'status': 'healthy' if chaos_status['safety_monitor_active'] else 'degraded',
                        'active_experiments': chaos_status['active_experiments']
                    },
                    'resilience_tester': {
                        'status': 'healthy',
                        'active_tests': resilience_status['active_tests']
                    },
                    'fault_manager': {
                        'status': 'healthy',
                        'active_faults': fault_status['active_faults']
                    },
                    'scenario_library': {
                        'status': 'healthy',
                        'total_scenarios': len(self.scenario_library.scenarios)
                    }
                }
            }

            return web.json_response(health_data)

        except Exception as e:
            return web.json_response(
                {'status': 'unhealthy', 'error': str(e)},
                status=503
            )

    async def system_status(self, request: Request) -> Response:
        """Get comprehensive system status."""
        try:
            status_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'chaos_engineer': self.chaos_engineer.get_system_status(),
                'resilience_tester': self.resilience_tester.get_system_status(),
                'fault_manager': await self.fault_manager.get_all_faults_status(),
                'scenario_library': self.scenario_library.get_scenario_statistics()
            }

            return web.json_response(status_data)

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def system_metrics(self, request: Request) -> Response:
        """Get system metrics in Prometheus format."""
        try:
            chaos_status = self.chaos_engineer.get_system_status()
            resilience_status = self.resilience_tester.get_system_status()
            fault_status = await self.fault_manager.get_all_faults_status()

            metrics = []

            # Chaos engineering metrics
            metrics.append(f"chaos_active_experiments {chaos_status['active_experiments']}")
            metrics.append(f"chaos_experiment_history_count {chaos_status['experiment_history_count']}")

            # Resilience testing metrics
            metrics.append(f"resilience_active_tests {resilience_status['active_tests']}")
            metrics.append(f"resilience_test_history_count {resilience_status['test_history_count']}")

            # Fault injection metrics
            metrics.append(f"fault_injection_active_faults {fault_status['active_faults']}")
            metrics.append(f"fault_injection_fault_history_count {fault_status['fault_history_count']}")

            return web.Response(text='\n'.join(metrics), content_type='text/plain')

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def create_experiment(self, request: Request) -> Response:
        """Create a new chaos experiment."""
        try:
            data = await request.json()

            # Convert data to ExperimentConfig
            config = ExperimentConfig(**data)

            # Validate experiment
            if not config.name:
                return web.json_response({'error': 'Experiment name is required'}, status=400)

            # Store experiment configuration
            experiment_id = f"exp_{int(time.time())}_{hash(config.name) % 10000}"

            # Here you would typically store the experiment config
            # For now, return the experiment ID

            response_data = {
                'experiment_id': experiment_id,
                'name': config.name,
                'status': 'created',
                'created_at': datetime.utcnow().isoformat()
            }

            return web.json_response(response_data, status=201)

        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)

    async def list_experiments(self, request: Request) -> Response:
        """List all experiments."""
        try:
            experiments = []

            # Get active experiments from chaos engineer
            for name, result in self.chaos_engineer.active_experiments.items():
                experiments.append({
                    'experiment_id': name,
                    'name': name,
                    'status': 'active',
                    'start_time': result.start_time.isoformat(),
                    'phase': result.phase.value if hasattr(result, 'phase') else 'unknown'
                })

            # Get experiment history
            for result in self.chaos_engineer.experiment_history[-10:]:  # Last 10
                experiments.append({
                    'experiment_id': result.experiment_name,
                    'name': result.experiment_name,
                    'status': 'completed' if result.success else 'failed',
                    'start_time': result.start_time.isoformat() if hasattr(result, 'start_time') else None,
                    'end_time': result.end_time.isoformat() if hasattr(result, 'end_time') else None
                })

            return web.json_response({'experiments': experiments})

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def inject_fault(self, request: Request) -> Response:
        """Inject a fault into the system."""
        try:
            data = await request.json()

            injector_name = data.get('injector_name')
            target_service = data.get('target_service')
            profile_name = data.get('profile_name')
            parameters = data.get('parameters', {})

            if not all([injector_name, target_service, profile_name]):
                return web.json_response({
                    'error': 'injector_name, target_service, and profile_name are required'
                }, status=400)

            # Inject fault
            fault_id = await self.fault_manager.inject_fault(
                injector_name, target_service, profile_name, parameters
            )

            # Broadcast to WebSocket clients
            await self.broadcast_websocket({
                'type': 'fault_injected',
                'fault_id': fault_id,
                'injector': injector_name,
                'target': target_service,
                'timestamp': datetime.utcnow().isoformat()
            })

            response_data = {
                'fault_id': fault_id,
                'status': 'injected',
                'injector_name': injector_name,
                'target_service': target_service,
                'profile_name': profile_name,
                'timestamp': datetime.utcnow().isoformat()
            }

            return web.json_response(response_data, status=201)

        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)

    async def list_active_faults(self, request: Request) -> Response:
        """List all active faults."""
        try:
            fault_status = await self.fault_manager.get_all_faults_status()
            return web.json_response(fault_status)

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def get_fault_status(self, request: Request) -> Response:
        """Get status of a specific fault."""
        try:
            fault_id = request.match_info['fault_id']

            fault_status = await self.fault_manager.get_fault_status(fault_id)
            if fault_status is None:
                return web.json_response({'error': 'Fault not found'}, status=404)

            return web.json_response(fault_status)

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def remove_fault(self, request: Request) -> Response:
        """Remove a specific fault."""
        try:
            fault_id = request.match_info['fault_id']

            success = await self.fault_manager.remove_fault(fault_id)
            if not success:
                return web.json_response({'error': 'Failed to remove fault'}, status=400)

            # Broadcast to WebSocket clients
            await self.broadcast_websocket({
                'type': 'fault_removed',
                'fault_id': fault_id,
                'timestamp': datetime.utcnow().isoformat()
            })

            return web.json_response({
                'fault_id': fault_id,
                'status': 'removed',
                'timestamp': datetime.utcnow().isoformat()
            })

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def remove_all_faults(self, request: Request) -> Response:
        """Remove all active faults."""
        try:
            results = await self.fault_manager.remove_all_faults()

            # Broadcast to WebSocket clients
            await self.broadcast_websocket({
                'type': 'all_faults_removed',
                'removed_count': sum(results.values()),
                'timestamp': datetime.utcnow().isoformat()
            })

            return web.json_response({
                'status': 'completed',
                'results': results,
                'timestamp': datetime.utcnow().isoformat()
            })

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def list_scenarios(self, request: Request) -> Response:
        """List all available scenarios."""
        try:
            scenarios = self.scenario_library.list_available_scenarios()
            return web.json_response({'scenarios': scenarios})

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def get_scenario(self, request: Request) -> Response:
        """Get a specific scenario."""
        try:
            scenario_name = request.match_info['scenario_name']

            scenario = self.scenario_library.get_scenario(scenario_name)
            if scenario is None:
                return web.json_response({'error': 'Scenario not found'}, status=404)

            return web.json_response(asdict(scenario))

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def create_scenario(self, request: Request) -> Response:
        """Create a new custom scenario."""
        try:
            data = await request.json()

            scenario = ChaosScenario(**data)
            success = self.scenario_library.create_custom_scenario(scenario)

            if not success:
                return web.json_response({'error': 'Failed to create scenario'}, status=400)

            return web.json_response({
                'scenario_name': scenario.name,
                'status': 'created',
                'timestamp': datetime.utcnow().isoformat()
            }, status=201)

        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)

    async def emergency_stop(self, request: Request) -> Response:
        """Emergency stop all chaos activities."""
        try:
            # Stop all active experiments
            stopped_experiments = []
            for experiment_name in list(self.chaos_engineer.active_experiments.keys()):
                await self.chaos_engineer.stop_experiment(experiment_name, emergency=True)
                stopped_experiments.append(experiment_name)

            # Remove all active faults
            fault_results = await self.fault_manager.remove_all_faults()

            # Stop all resilience tests
            stopped_tests = []
            for test_name in list(self.resilience_tester.active_tests.keys()):
                await self.resilience_tester.stop_test(test_name)
                stopped_tests.append(test_name)

            # Broadcast emergency stop
            await self.broadcast_websocket({
                'type': 'emergency_stop',
                'stopped_experiments': stopped_experiments,
                'removed_faults': fault_results,
                'stopped_tests': stopped_tests,
                'timestamp': datetime.utcnow().isoformat()
            })

            response_data = {
                'status': 'emergency_stop_completed',
                'stopped_experiments': stopped_experiments,
                'removed_faults': fault_results,
                'stopped_tests': stopped_tests,
                'timestamp': datetime.utcnow().isoformat()
            }

            return web.json_response(response_data)

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def safety_check(self, request: Request) -> Response:
        """Perform safety check of the system."""
        try:
            # Validate all active faults
            fault_validation = await self.fault_manager.validate_all_active_faults()

            # Check system health
            health_response = await self.health_check(request)
            health_data = await health_response.json()

            safety_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_safe': health_data['status'] == 'healthy',
                'active_faults_valid': all(fault_validation.values()),
                'fault_validation_results': fault_validation,
                'system_health': health_data,
                'recommendations': []
            }

            # Add recommendations based on findings
            if not safety_data['overall_safe']:
                safety_data['recommendations'].append('System health issues detected - consider emergency stop')

            if not safety_data['active_faults_valid']:
                safety_data['recommendations'].append('Some faults are not working as expected - validate fault configurations')

            return web.json_response(safety_data)

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def websocket_handler(self, request: Request) -> WebSocketResponse:
        """WebSocket handler for real-time monitoring."""
        ws = WebSocketResponse()
        await ws.prepare(request)

        self.websocket_connections.add(ws)
        self.logger.info("WebSocket client connected")

        try:
            # Send initial status
            initial_status = {
                'type': 'initial_status',
                'timestamp': datetime.utcnow().isoformat(),
                'chaos_engineer': self.chaos_engineer.get_system_status(),
                'fault_manager': await self.fault_manager.get_all_faults_status()
            }
            await ws.send_str(json.dumps(initial_status))

            # Handle incoming messages
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self.handle_websocket_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({
                            'type': 'error',
                            'message': 'Invalid JSON format'
                        }))
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f'WebSocket error: {ws.exception()}')

        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            self.logger.info("WebSocket client disconnected")

        return ws

    async def handle_websocket_message(self, ws: WebSocketResponse, data: Dict[str, Any]):
        """Handle WebSocket messages from clients."""
        message_type = data.get('type')

        if message_type == 'subscribe':
            # Client wants to subscribe to specific events
            await ws.send_str(json.dumps({
                'type': 'subscription_confirmed',
                'subscribed_to': data.get('events', [])
            }))

        elif message_type == 'get_status':
            # Client requests current status
            status = {
                'type': 'status_update',
                'timestamp': datetime.utcnow().isoformat(),
                'chaos_engineer': self.chaos_engineer.get_system_status(),
                'fault_manager': await self.fault_manager.get_all_faults_status()
            }
            await ws.send_str(json.dumps(status))

    async def broadcast_websocket(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSocket clients."""
        if not self.websocket_connections:
            return

        message_str = json.dumps(message)
        disconnected = []

        for ws in self.websocket_connections:
            try:
                await ws.send_str(message_str)
            except Exception:
                disconnected.append(ws)

        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_connections.discard(ws)

    async def validate_auto_recovery(self, request: Request) -> Response:
        """Validate integration with auto-recovery system."""
        try:
            data = await request.json()
            service_names = data.get('services', [])

            # This would integrate with the auto-recovery validation
            # For now, return a mock response
            validation_results = {}
            for service in service_names:
                validation_results[service] = {
                    'auto_recovery_available': True,
                    'last_recovery_time': '2024-01-01T00:00:00Z',
                    'recovery_success_rate': 0.95
                }

            return web.json_response({
                'validation_results': validation_results,
                'overall_integration_status': 'healthy',
                'timestamp': datetime.utcnow().isoformat()
            })

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def collect_health_metrics(self, request: Request) -> Response:
        """Collect metrics from health monitoring system."""
        try:
            data = await request.json()
            service_names = data.get('services', [])

            # This would integrate with the health monitoring system
            # For now, return mock metrics
            metrics = {}
            for service in service_names:
                metrics[service] = {
                    'availability': 99.5,
                    'response_time_ms': 150,
                    'error_rate': 0.1,
                    'cpu_usage': 45.2,
                    'memory_usage': 67.8,
                    'timestamp': datetime.utcnow().isoformat()
                }

            return web.json_response({
                'metrics': metrics,
                'collection_timestamp': datetime.utcnow().isoformat()
            })

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def start_server(self):
        """Start the API server."""
        try:
            app = await self.create_app()
            runner = web.AppRunner(app)
            await runner.setup()

            site = web.TCPSite(runner, self.host, self.port)
            await site.start()

            self.logger.info(f"Chaos Engineering API server started on {self.host}:{self.port}")

            # Keep the server running
            try:
                while True:
                    await asyncio.sleep(3600)  # Sleep for 1 hour intervals
            except KeyboardInterrupt:
                pass
            finally:
                await runner.cleanup()

        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
            raise


# Main entry point for the API server
async def main():
    """Main entry point for the chaos engineering API server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Initialize components
        chaos_engineer = ChaosEngineer()
        await chaos_engineer.initialize()

        resilience_tester = ResilienceTester()
        await resilience_tester.initialize()

        scenario_library = ScenarioLibrary()

        fault_manager = FaultInjectionManager(
            docker_client=chaos_engineer.docker_client,
            redis_client=chaos_engineer.redis_client
        )

        # Create and start API server
        api = ChaosEngineeringAPI(
            chaos_engineer=chaos_engineer,
            resilience_tester=resilience_tester,
            scenario_library=scenario_library,
            fault_manager=fault_manager
        )

        await api.start_server()

    except Exception as e:
        logging.error(f"Failed to start chaos engineering API: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())