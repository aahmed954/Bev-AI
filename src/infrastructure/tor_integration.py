import os
#!/usr/bin/env python3
"""
Tor Network Integration for BEV Proxy Management System
Integrates with existing Tor infrastructure for enhanced anonymity

Features:
- Tor circuit management and rotation
- Control port integration for circuit selection
- Tor stream isolation and identity management
- Hidden service discovery and routing
- Tor network health monitoring
- Integration with proxy pool management

Place in: /home/starlord/Projects/Bev/src/infrastructure/tor_integration.py
"""

import asyncio
import socket
import hashlib
import hmac
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import struct
import base64

# Import proxy manager components
from .proxy_manager import ProxyEndpoint, ProxyType, ProxyRegion, ProxyStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TorCircuitPurpose(Enum):
    """Tor circuit purposes"""
    GENERAL = "general"
    HIDDEN_SERVICE = "hidden_service"
    BRIDGE = "bridge"
    CONTROLLER = "controller"
    MEASURE = "measure"

class TorStreamIsolation(Enum):
    """Tor stream isolation levels"""
    NONE = "none"
    DESTINATION = "destination"
    ISOLATION_FLAG = "isolation_flag"
    SESSION = "session"
    NYM = "nym"

@dataclass
class TorCircuit:
    """Tor circuit information"""
    circuit_id: str
    status: str
    path: List[str]
    build_flags: List[str]
    purpose: TorCircuitPurpose
    time_created: datetime
    time_last_used: Optional[datetime] = None
    is_internal: bool = False
    bytes_read: int = 0
    bytes_written: int = 0

@dataclass
class TorRelay:
    """Tor relay information"""
    fingerprint: str
    nickname: str
    address: str
    or_port: int
    dir_port: int
    flags: List[str]
    bandwidth: int
    country_code: Optional[str] = None
    exit_policy: Optional[str] = None

@dataclass
class TorStream:
    """Tor stream information"""
    stream_id: str
    status: str
    circuit_id: str
    target: str
    source: str
    purpose: str
    isolation_fields: Dict[str, str] = None

    def __post_init__(self):
        if self.isolation_fields is None:
            self.isolation_fields = {}

class TorController:
    """Tor control port interface"""

    def __init__(self,
                 control_host: str = "127.0.0.1",
                 control_port: int = 9051,
                 control_password: Optional[str] = None,
                 socket_timeout: float = 10.0):

        self.control_host = control_host
        self.control_port = control_port
        self.control_password = control_password
        self.socket_timeout = socket_timeout

        self.socket = None
        self.authenticated = False
        self.protocol_info = {}

        # State tracking
        self.circuits = {}
        self.streams = {}
        self.relays = {}

        logger.info(f"TorController initialized for {control_host}:{control_port}")

    async def connect(self) -> bool:
        """Connect to Tor control port"""
        try:
            # Create socket connection
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.socket_timeout)

            await asyncio.get_event_loop().run_in_executor(
                None, self.socket.connect, (self.control_host, self.control_port)
            )

            logger.info("Connected to Tor control port")

            # Get protocol info
            await self._get_protocol_info()

            # Authenticate
            if await self._authenticate():
                self.authenticated = True
                logger.info("Authenticated with Tor control port")

                # Enable events
                await self._enable_events()

                return True
            else:
                logger.error("Failed to authenticate with Tor")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Tor control port: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Tor control port"""
        try:
            if self.socket:
                await self._send_command("QUIT")
                self.socket.close()
                self.socket = None
                self.authenticated = False
                logger.info("Disconnected from Tor control port")
        except Exception as e:
            logger.error(f"Error disconnecting from Tor: {e}")

    async def _get_protocol_info(self):
        """Get Tor protocol information"""
        try:
            response = await self._send_command("PROTOCOLINFO")

            for line in response:
                if line.startswith("250-"):
                    line = line[4:]  # Remove "250-"

                    if "AUTH" in line:
                        # Parse authentication methods
                        auth_part = line.split("AUTH")[1].strip()
                        if "METHODS=" in auth_part:
                            methods = auth_part.split("METHODS=")[1].split()[0]
                            self.protocol_info['auth_methods'] = methods.split(",")

                    elif "VERSION" in line:
                        version = line.split("VERSION=")[1].strip().strip('"')
                        self.protocol_info['version'] = version

        except Exception as e:
            logger.error(f"Error getting protocol info: {e}")

    async def _authenticate(self) -> bool:
        """Authenticate with Tor control port"""
        try:
            auth_methods = self.protocol_info.get('auth_methods', ['PASSWORD'])

            if 'NULL' in auth_methods:
                # No authentication required
                response = await self._send_command("AUTHENTICATE")
                return "250 OK" in "\n".join(response)

            elif 'PASSWORD' in auth_methods and self.control_password:
                # Password authentication
                password_hex = self.control_password.encode().hex()
                response = await self._send_command(f"AUTHENTICATE {password_hex}")
                return "250 OK" in "\n".join(response)

            elif 'COOKIE' in auth_methods:
                # Cookie authentication (would need to read cookie file)
                logger.warning("Cookie authentication not implemented")
                return False

            else:
                logger.error("No suitable authentication method available")
                return False

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

    async def _enable_events(self):
        """Enable Tor events we want to monitor"""
        try:
            events = [
                "CIRC", "STREAM", "ORCONN", "BW",
                "NEWDESC", "ADDRMAP", "STATUS_CLIENT"
            ]

            await self._send_command(f"SETEVENTS {' '.join(events)}")
            logger.info("Enabled Tor event monitoring")

        except Exception as e:
            logger.error(f"Failed to enable events: {e}")

    async def _send_command(self, command: str) -> List[str]:
        """Send command to Tor control port"""
        if not self.socket:
            raise Exception("Not connected to Tor control port")

        try:
            # Send command
            command_bytes = (command + "\r\n").encode()
            await asyncio.get_event_loop().run_in_executor(
                None, self.socket.send, command_bytes
            )

            # Read response
            response_lines = []
            while True:
                data = await asyncio.get_event_loop().run_in_executor(
                    None, self.socket.recv, 4096
                )

                if not data:
                    break

                lines = data.decode().strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        response_lines.append(line)

                        # Check for command completion
                        if line.startswith("250 ") or line.startswith("250-"):
                            if line.startswith("250 "):
                                return response_lines
                        elif line.startswith(("4", "5")):  # Error responses
                            raise Exception(f"Tor command error: {line}")

        except Exception as e:
            logger.error(f"Error sending command '{command}': {e}")
            raise

    async def get_circuits(self) -> List[TorCircuit]:
        """Get current Tor circuits"""
        try:
            response = await self._send_command("GETINFO circuit-status")
            circuits = []

            for line in response:
                if line.startswith("250+circuit-status=") or line.startswith("250-circuit-status="):
                    continue
                elif line.startswith("250 ") or line == ".":
                    continue
                else:
                    # Parse circuit line
                    circuit = self._parse_circuit_line(line)
                    if circuit:
                        circuits.append(circuit)
                        self.circuits[circuit.circuit_id] = circuit

            return circuits

        except Exception as e:
            logger.error(f"Error getting circuits: {e}")
            return []

    def _parse_circuit_line(self, line: str) -> Optional[TorCircuit]:
        """Parse a circuit status line"""
        try:
            parts = line.split()
            if len(parts) < 3:
                return None

            circuit_id = parts[0]
            status = parts[1]
            path_part = parts[2]

            # Parse path
            path = []
            if path_part != "":
                relays = path_part.split(",")
                for relay in relays:
                    # Remove flags like ~, +, etc.
                    clean_relay = relay.lstrip("~+$")
                    if "=" in clean_relay:
                        clean_relay = clean_relay.split("=")[0]
                    path.append(clean_relay)

            # Parse additional information
            build_flags = []
            purpose = TorCircuitPurpose.GENERAL
            time_created = datetime.now()

            if len(parts) > 3:
                for part in parts[3:]:
                    if part.startswith("BUILD_FLAGS="):
                        build_flags = part.split("=")[1].split(",")
                    elif part.startswith("PURPOSE="):
                        purpose_str = part.split("=")[1]
                        try:
                            purpose = TorCircuitPurpose(purpose_str.lower())
                        except ValueError:
                            purpose = TorCircuitPurpose.GENERAL
                    elif part.startswith("TIME_CREATED="):
                        try:
                            timestamp = float(part.split("=")[1])
                            time_created = datetime.fromtimestamp(timestamp)
                        except:
                            pass

            return TorCircuit(
                circuit_id=circuit_id,
                status=status,
                path=path,
                build_flags=build_flags,
                purpose=purpose,
                time_created=time_created
            )

        except Exception as e:
            logger.error(f"Error parsing circuit line '{line}': {e}")
            return None

    async def get_streams(self) -> List[TorStream]:
        """Get current Tor streams"""
        try:
            response = await self._send_command("GETINFO stream-status")
            streams = []

            for line in response:
                if line.startswith("250+stream-status=") or line.startswith("250-stream-status="):
                    continue
                elif line.startswith("250 ") or line == ".":
                    continue
                else:
                    # Parse stream line
                    stream = self._parse_stream_line(line)
                    if stream:
                        streams.append(stream)
                        self.streams[stream.stream_id] = stream

            return streams

        except Exception as e:
            logger.error(f"Error getting streams: {e}")
            return []

    def _parse_stream_line(self, line: str) -> Optional[TorStream]:
        """Parse a stream status line"""
        try:
            parts = line.split()
            if len(parts) < 4:
                return None

            stream_id = parts[0]
            status = parts[1]
            circuit_id = parts[2]
            target = parts[3]

            # Default values
            source = ""
            purpose = "USER"
            isolation_fields = {}

            # Parse additional fields
            if len(parts) > 4:
                for part in parts[4:]:
                    if part.startswith("SOURCE="):
                        source = part.split("=")[1]
                    elif part.startswith("PURPOSE="):
                        purpose = part.split("=")[1]
                    elif "=" in part:
                        key, value = part.split("=", 1)
                        isolation_fields[key] = value

            return TorStream(
                stream_id=stream_id,
                status=status,
                circuit_id=circuit_id,
                target=target,
                source=source,
                purpose=purpose,
                isolation_fields=isolation_fields
            )

        except Exception as e:
            logger.error(f"Error parsing stream line '{line}': {e}")
            return None

    async def new_circuit(self,
                         purpose: str = "general",
                         guard_fingerprints: Optional[List[str]] = None) -> Optional[str]:
        """Create a new Tor circuit"""
        try:
            command = "EXTENDCIRCUIT 0"

            if guard_fingerprints:
                # Specify path
                command += " " + ",".join(guard_fingerprints)

            if purpose != "general":
                command += f" purpose={purpose}"

            response = await self._send_command(command)

            # Parse response to get circuit ID
            for line in response:
                if line.startswith("250 EXTENDED"):
                    circuit_id = line.split()[1]
                    logger.info(f"Created new circuit: {circuit_id}")
                    return circuit_id

            return None

        except Exception as e:
            logger.error(f"Error creating new circuit: {e}")
            return None

    async def close_circuit(self, circuit_id: str) -> bool:
        """Close a Tor circuit"""
        try:
            response = await self._send_command(f"CLOSECIRCUIT {circuit_id}")

            success = any("250 OK" in line for line in response)
            if success:
                logger.info(f"Closed circuit: {circuit_id}")
                if circuit_id in self.circuits:
                    del self.circuits[circuit_id]

            return success

        except Exception as e:
            logger.error(f"Error closing circuit {circuit_id}: {e}")
            return False

    async def new_identity(self) -> bool:
        """Signal Tor to use a new identity (new circuits)"""
        try:
            response = await self._send_command("SIGNAL NEWNYM")

            success = any("250 OK" in line for line in response)
            if success:
                logger.info("Requested new Tor identity")
                # Clear circuit cache as they will be rebuilt
                self.circuits.clear()
                self.streams.clear()

            return success

        except Exception as e:
            logger.error(f"Error requesting new identity: {e}")
            return False

    async def set_conf(self, config_options: Dict[str, str]) -> bool:
        """Set Tor configuration options"""
        try:
            options = []
            for key, value in config_options.items():
                options.append(f"{key}={value}")

            command = "SETCONF " + " ".join(options)
            response = await self._send_command(command)

            success = any("250 OK" in line for line in response)
            if success:
                logger.info(f"Set configuration: {config_options}")

            return success

        except Exception as e:
            logger.error(f"Error setting configuration: {e}")
            return False

    async def get_conf(self, config_keys: List[str]) -> Dict[str, str]:
        """Get Tor configuration values"""
        try:
            command = "GETCONF " + " ".join(config_keys)
            response = await self._send_command(command)

            config = {}
            for line in response:
                if "=" in line and not line.startswith("250"):
                    key, value = line.split("=", 1)
                    config[key] = value

            return config

        except Exception as e:
            logger.error(f"Error getting configuration: {e}")
            return {}

class TorProxyIntegration:
    """Integration between Tor network and proxy management"""

    def __init__(self,
                 tor_socks_host: str = "127.0.0.1",
                 tor_socks_port: int = 9050,
                 tor_control_host: str = "127.0.0.1",
                 tor_control_port: int = 9051,
                 tor_control_password: Optional[str] = None):

        self.tor_socks_host = tor_socks_host
        self.tor_socks_port = tor_socks_port

        # Tor controller
        self.tor_controller = TorController(
            tor_control_host, tor_control_port, tor_control_password
        )

        # Circuit management
        self.circuit_pool = {}
        self.circuit_rotation_interval = 600  # 10 minutes
        self.max_circuits = 10

        # Stream isolation
        self.isolation_sessions = {}
        self.session_circuits = {}

        # Health monitoring
        self.last_health_check = None
        self.health_status = "unknown"

        logger.info("TorProxyIntegration initialized")

    async def initialize(self) -> bool:
        """Initialize Tor integration"""
        try:
            # Connect to Tor control port
            if not await self.tor_controller.connect():
                logger.error("Failed to connect to Tor control port")
                return False

            # Test SOCKS proxy
            if not await self._test_socks_proxy():
                logger.error("Tor SOCKS proxy not responding")
                return False

            # Start background tasks
            asyncio.create_task(self._circuit_maintenance())
            asyncio.create_task(self._health_monitoring())

            logger.info("Tor integration initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Tor integration: {e}")
            return False

    async def shutdown(self):
        """Shutdown Tor integration"""
        logger.info("Shutting down Tor integration...")

        try:
            # Close all managed circuits
            for circuit_id in list(self.circuit_pool.keys()):
                await self.tor_controller.close_circuit(circuit_id)

            # Disconnect from control port
            await self.tor_controller.disconnect()

        except Exception as e:
            logger.error(f"Error during Tor shutdown: {e}")

    async def _test_socks_proxy(self) -> bool:
        """Test Tor SOCKS proxy connectivity"""
        try:
            # Simple test by connecting to the SOCKS port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)

            result = await asyncio.get_event_loop().run_in_executor(
                None, sock.connect_ex, (self.tor_socks_host, self.tor_socks_port)
            )

            sock.close()
            return result == 0

        except Exception as e:
            logger.error(f"SOCKS proxy test failed: {e}")
            return False

    async def get_tor_proxy_endpoint(self,
                                   isolation_key: Optional[str] = None,
                                   purpose: str = "general") -> Optional[ProxyEndpoint]:
        """Get Tor proxy endpoint with optional stream isolation"""

        try:
            # Determine SOCKS port for isolation
            socks_port = self.tor_socks_port

            if isolation_key:
                # Use stream isolation
                socks_port = await self._get_isolated_port(isolation_key)

            # Create proxy endpoint
            endpoint = ProxyEndpoint(
                host=self.tor_socks_host,
                port=socks_port,
                proxy_type=ProxyType.TOR,
                region=ProxyRegion.GLOBAL,
                provider="tor_internal",
                weight=0.8,  # Lower weight due to potential latency
                max_connections=20,  # Conservative limit for Tor
                rotation_interval=self.circuit_rotation_interval
            )

            # Set status based on health
            endpoint.status = (
                ProxyStatus.HEALTHY if self.health_status == "healthy"
                else ProxyStatus.DEGRADED if self.health_status == "degraded"
                else ProxyStatus.UNHEALTHY
            )

            # Add Tor-specific metadata
            endpoint.provider_pool_id = isolation_key or "default"

            return endpoint

        except Exception as e:
            logger.error(f"Error getting Tor proxy endpoint: {e}")
            return None

    async def _get_isolated_port(self, isolation_key: str) -> int:
        """Get SOCKS port for stream isolation"""
        # For basic implementation, return the standard port
        # In production, you might configure multiple Tor instances
        # or use Tor's stream isolation features
        return self.tor_socks_port

    async def rotate_circuits(self, force: bool = False) -> bool:
        """Rotate Tor circuits"""
        try:
            if force:
                # Force new identity
                success = await self.tor_controller.new_identity()
                if success:
                    logger.info("Forced circuit rotation (new identity)")
                    # Wait a moment for circuits to rebuild
                    await asyncio.sleep(2)
                return success
            else:
                # Gradual rotation of specific circuits
                circuits = await self.tor_controller.get_circuits()
                old_circuits = [
                    c for c in circuits
                    if (datetime.now() - c.time_created).total_seconds() > self.circuit_rotation_interval
                ]

                rotated = 0
                for circuit in old_circuits[:3]:  # Rotate up to 3 at a time
                    if await self.tor_controller.close_circuit(circuit.circuit_id):
                        rotated += 1

                if rotated > 0:
                    logger.info(f"Rotated {rotated} old circuits")

                return rotated > 0

        except Exception as e:
            logger.error(f"Error rotating circuits: {e}")
            return False

    async def _circuit_maintenance(self):
        """Background task for circuit maintenance"""
        while True:
            try:
                # Check circuit health and rotate if needed
                await self.rotate_circuits(force=False)

                # Clean up closed circuits
                circuits = await self.tor_controller.get_circuits()
                active_ids = {c.circuit_id for c in circuits}

                for circuit_id in list(self.circuit_pool.keys()):
                    if circuit_id not in active_ids:
                        del self.circuit_pool[circuit_id]

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Circuit maintenance error: {e}")
                await asyncio.sleep(30)

    async def _health_monitoring(self):
        """Background task for health monitoring"""
        while True:
            try:
                # Test Tor connectivity
                socks_ok = await self._test_socks_proxy()

                # Check circuit count
                circuits = await self.tor_controller.get_circuits()
                circuit_count = len([c for c in circuits if c.status == "BUILT"])

                # Determine health status
                if socks_ok and circuit_count >= 3:
                    self.health_status = "healthy"
                elif socks_ok and circuit_count >= 1:
                    self.health_status = "degraded"
                else:
                    self.health_status = "unhealthy"

                self.last_health_check = datetime.now()

                logger.debug(
                    f"Tor health check: {self.health_status} "
                    f"(circuits: {circuit_count}, socks: {socks_ok})"
                )

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                self.health_status = "unhealthy"
                await asyncio.sleep(60)

    async def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive Tor network status"""
        try:
            circuits = await self.tor_controller.get_circuits()
            streams = await self.tor_controller.get_streams()

            # Circuit statistics
            circuit_stats = {
                'total': len(circuits),
                'built': len([c for c in circuits if c.status == "BUILT"]),
                'building': len([c for c in circuits if c.status == "EXTENDING"]),
                'failed': len([c for c in circuits if c.status == "FAILED"]),
                'by_purpose': {}
            }

            for circuit in circuits:
                purpose = circuit.purpose.value
                circuit_stats['by_purpose'][purpose] = circuit_stats['by_purpose'].get(purpose, 0) + 1

            # Stream statistics
            stream_stats = {
                'total': len(streams),
                'connected': len([s for s in streams if s.status == "SUCCEEDED"]),
                'connecting': len([s for s in streams if s.status in ["NEW", "NEWRESOLVE", "REMAP"]]),
                'failed': len([s for s in streams if s.status == "FAILED"])
            }

            return {
                'health_status': self.health_status,
                'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
                'socks_proxy': f"{self.tor_socks_host}:{self.tor_socks_port}",
                'control_authenticated': self.tor_controller.authenticated,
                'circuits': circuit_stats,
                'streams': stream_stats,
                'rotation_interval': self.circuit_rotation_interval
            }

        except Exception as e:
            logger.error(f"Error getting network status: {e}")
            return {
                'health_status': 'error',
                'error': str(e)
            }

    async def configure_for_target(self, target: str) -> Dict[str, Any]:
        """Configure Tor settings for specific target"""
        try:
            # Basic target analysis
            is_onion = target.endswith('.onion')

            config_recommendations = {
                'stream_isolation': True,
                'circuit_timeout': 60,
                'use_exit_nodes': not is_onion
            }

            if is_onion:
                # Hidden service specific configuration
                config_recommendations.update({
                    'circuit_type': 'hidden_service',
                    'suggested_timeout': 120,
                    'retry_attempts': 3
                })
            else:
                # Regular web target
                config_recommendations.update({
                    'circuit_type': 'general',
                    'suggested_timeout': 60,
                    'retry_attempts': 2
                })

            return config_recommendations

        except Exception as e:
            logger.error(f"Error configuring for target {target}: {e}")
            return {'error': str(e)}

# Factory function
async def create_tor_integration(
    tor_socks_host: str = "127.0.0.1",
    tor_socks_port: int = 9050,
    tor_control_host: str = "127.0.0.1",
    tor_control_port: int = 9051,
    tor_control_password: Optional[str] = None
) -> TorProxyIntegration:
    """Create and initialize Tor integration"""

    integration = TorProxyIntegration(
        tor_socks_host, tor_socks_port,
        tor_control_host, tor_control_port, tor_control_password
    )

    await integration.initialize()
    return integration

if __name__ == "__main__":
    # Example usage
    async def main():
        try:
            # Create Tor integration
            tor_integration = await create_tor_integration(
                tor_control_password=os.getenv('DB_PASSWORD', 'dev_password')  # From docker-compose
            )

            # Get status
            status = await tor_integration.get_network_status()
            print(f"Tor network status: {status}")

            # Get proxy endpoint
            proxy = await tor_integration.get_tor_proxy_endpoint()
            if proxy:
                print(f"Tor proxy: {proxy.proxy_url}")

            # Test circuit rotation
            await tor_integration.rotate_circuits(force=False)

            # Configure for specific target
            config = await tor_integration.configure_for_target("facebook.com")
            print(f"Target configuration: {config}")

            # Hidden service example
            onion_config = await tor_integration.configure_for_target("duckduckgogg42ts.onion")
            print(f"Onion service configuration: {onion_config}")

        finally:
            await tor_integration.shutdown()

    asyncio.run(main())