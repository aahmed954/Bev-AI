#!/usr/bin/env python3
"""
DRM Research Worker for ORACLE1
Advanced Digital Rights Management analysis and intelligence gathering
Supports Widevine, PlayReady, FairPlay, and other DRM systems
"""

import asyncio
import json
import time
import hashlib
import base64
import struct
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import aiohttp
import aiofiles
import re
import binascii

# Crypto libraries for DRM analysis
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import struct

# Media analysis
import requests
from urllib.parse import urlparse, parse_qs
import xml.etree.ElementTree as ET

# ORACLE integration
import redis
from influxdb_client import InfluxDBClient, Point, WritePrecision

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DRMSystem(Enum):
    WIDEVINE = "widevine"
    PLAYREADY = "playready"
    FAIRPLAY = "fairplay"
    PRIMETIME = "primetime"
    MARLIN = "marlin"
    VUDU = "vudu"
    UNKNOWN = "unknown"

class DRMLevel(Enum):
    L1 = "L1"  # Hardware-based
    L2 = "L2"  # Software-based with hardware crypto
    L3 = "L3"  # Software-based
    UNKNOWN = "unknown"

@dataclass
class DRMMetadata:
    """DRM system metadata"""
    system: DRMSystem
    version: str
    security_level: DRMLevel
    key_system: str
    license_url: str
    content_id: str
    pssh_data: Optional[str]
    encryption_scheme: str
    key_ids: List[str]
    content_type: str

@dataclass
class DRMAnalysisResult:
    """Result of DRM analysis"""
    timestamp: datetime
    content_url: str
    drm_metadata: DRMMetadata
    vulnerabilities: List[str]
    bypass_methods: List[str]
    protection_strength: str  # WEAK, MEDIUM, STRONG, MILITARY
    analysis_confidence: float
    technical_details: Dict[str, Any]
    recommendations: List[str]

class WidevineAnalyzer:
    """Widevine DRM system analyzer"""

    def __init__(self):
        self.known_vulnerabilities = [
            "CDM downgrade attack",
            "HDCP bypass methods",
            "Key extraction via debug interfaces",
            "Screen recording bypass",
            "EME API manipulation",
            "License server spoofing"
        ]

    async def analyze_widevine_content(self, manifest_url: str, headers: Dict = None) -> DRMAnalysisResult:
        """Analyze Widevine-protected content"""
        try:
            logger.info(f"Analyzing Widevine content: {manifest_url}")

            # Fetch manifest
            async with aiohttp.ClientSession() as session:
                async with session.get(manifest_url, headers=headers or {}) as response:
                    manifest_data = await response.text()

            # Parse manifest for DRM information
            drm_info = await self._parse_widevine_manifest(manifest_data)

            # Extract PSSH data
            pssh_data = await self._extract_pssh_data(manifest_data)

            # Analyze security level
            security_level = await self._determine_security_level(drm_info)

            # Check for known vulnerabilities
            vulnerabilities = await self._check_widevine_vulnerabilities(drm_info)

            # Analyze bypass methods
            bypass_methods = await self._analyze_bypass_methods(drm_info, security_level)

            # Determine protection strength
            protection_strength = await self._assess_protection_strength(security_level, vulnerabilities)

            # Create DRM metadata
            drm_metadata = DRMMetadata(
                system=DRMSystem.WIDEVINE,
                version=drm_info.get('version', 'unknown'),
                security_level=security_level,
                key_system=drm_info.get('key_system', 'com.widevine.alpha'),
                license_url=drm_info.get('license_url', ''),
                content_id=drm_info.get('content_id', ''),
                pssh_data=pssh_data,
                encryption_scheme=drm_info.get('encryption', 'cenc'),
                key_ids=drm_info.get('key_ids', []),
                content_type=drm_info.get('content_type', 'video')
            )

            return DRMAnalysisResult(
                timestamp=datetime.now(),
                content_url=manifest_url,
                drm_metadata=drm_metadata,
                vulnerabilities=vulnerabilities,
                bypass_methods=bypass_methods,
                protection_strength=protection_strength,
                analysis_confidence=0.85,
                technical_details={
                    'manifest_size': len(manifest_data),
                    'drm_systems_detected': ['widevine'],
                    'analysis_method': 'manifest_parsing'
                },
                recommendations=await self._generate_recommendations(vulnerabilities, bypass_methods)
            )

        except Exception as e:
            logger.error(f"Widevine analysis failed: {e}")
            raise

    async def _parse_widevine_manifest(self, manifest_data: str) -> Dict:
        """Parse manifest for Widevine DRM information"""
        drm_info = {}

        try:
            # Check if it's DASH manifest
            if 'xmlns="urn:mpeg:dash:schema:mpd:2011"' in manifest_data:
                drm_info.update(await self._parse_dash_manifest(manifest_data))

            # Check if it's HLS manifest
            elif '#EXTM3U' in manifest_data:
                drm_info.update(await self._parse_hls_manifest(manifest_data))

            # Check if it's Smooth Streaming
            elif 'SmoothStreamingMedia' in manifest_data:
                drm_info.update(await self._parse_smooth_manifest(manifest_data))

        except Exception as e:
            logger.error(f"Manifest parsing error: {e}")

        return drm_info

    async def _parse_dash_manifest(self, manifest_data: str) -> Dict:
        """Parse DASH manifest for DRM information"""
        drm_info = {}

        try:
            root = ET.fromstring(manifest_data)

            # Find ContentProtection elements
            for cp in root.iter('{urn:mpeg:dash:schema:mpd:2011}ContentProtection'):
                scheme_id = cp.get('schemeIdUri', '')

                if 'widevine' in scheme_id.lower() or 'edef8ba9-79d6-4ace-a3c8-27dcd51d21ed' in scheme_id:
                    drm_info['key_system'] = 'com.widevine.alpha'
                    drm_info['scheme_id'] = scheme_id

                    # Extract PSSH data
                    pssh_elem = cp.find('.//*[@*="pssh"]')
                    if pssh_elem is not None:
                        drm_info['pssh'] = pssh_elem.text

                    # Extract license URL
                    license_elem = cp.find('.//*[@*="Laurl"]')
                    if license_elem is not None:
                        drm_info['license_url'] = license_elem.text

        except ET.ParseError as e:
            logger.error(f"DASH manifest parse error: {e}")

        return drm_info

    async def _parse_hls_manifest(self, manifest_data: str) -> Dict:
        """Parse HLS manifest for DRM information"""
        drm_info = {}

        # Look for EXT-X-KEY tags with Widevine
        for line in manifest_data.split('\n'):
            if line.startswith('#EXT-X-KEY:'):
                if 'KEYFORMAT="urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed"' in line:
                    drm_info['key_system'] = 'com.widevine.alpha'

                    # Extract URI
                    uri_match = re.search(r'URI="([^"]+)"', line)
                    if uri_match:
                        drm_info['license_url'] = uri_match.group(1)

        return drm_info

    async def _parse_smooth_manifest(self, manifest_data: str) -> Dict:
        """Parse Smooth Streaming manifest for DRM information"""
        drm_info = {}

        try:
            root = ET.fromstring(manifest_data)

            # Find Protection elements
            for protection in root.iter('Protection'):
                system_id = protection.get('SystemID', '')

                if system_id.lower() == 'edef8ba9-79d6-4ace-a3c8-27dcd51d21ed':
                    drm_info['key_system'] = 'com.widevine.alpha'
                    drm_info['system_id'] = system_id

                    # Extract ProtectionHeader
                    header = protection.find('ProtectionHeader')
                    if header is not None:
                        drm_info['protection_header'] = header.text

        except ET.ParseError as e:
            logger.error(f"Smooth manifest parse error: {e}")

        return drm_info

    async def _extract_pssh_data(self, manifest_data: str) -> Optional[str]:
        """Extract PSSH (Protection System Specific Header) data"""
        # Look for base64-encoded PSSH boxes
        pssh_pattern = r'([A-Za-z0-9+/]{100,}={0,2})'
        matches = re.findall(pssh_pattern, manifest_data)

        for match in matches:
            try:
                decoded = base64.b64decode(match)
                # Check if it's a valid PSSH box (starts with size and 'pssh')
                if len(decoded) >= 8 and decoded[4:8] == b'pssh':
                    return match
            except:
                continue

        return None

    async def _determine_security_level(self, drm_info: Dict) -> DRMLevel:
        """Determine Widevine security level"""
        # This would require actual CDM interaction in practice
        # For research purposes, analyze available indicators

        if 'hardware' in str(drm_info).lower():
            return DRMLevel.L1
        elif 'secure' in str(drm_info).lower():
            return DRMLevel.L2
        else:
            return DRMLevel.L3

    async def _check_widevine_vulnerabilities(self, drm_info: Dict) -> List[str]:
        """Check for known Widevine vulnerabilities"""
        vulnerabilities = []

        # Check for common vulnerability indicators
        if drm_info.get('security_level') == DRMLevel.L3:
            vulnerabilities.append("Software-only implementation vulnerable to debugging")

        if not drm_info.get('license_url', '').startswith('https://'):
            vulnerabilities.append("Unencrypted license transmission")

        # Check for weak key derivation
        if 'weak' in str(drm_info).lower():
            vulnerabilities.append("Weak key derivation detected")

        return vulnerabilities

    async def _analyze_bypass_methods(self, drm_info: Dict, security_level: DRMLevel) -> List[str]:
        """Analyze potential DRM bypass methods"""
        bypass_methods = []

        if security_level == DRMLevel.L3:
            bypass_methods.extend([
                "CDM emulation possible",
                "Key extraction via memory dumping",
                "Browser automation bypass",
                "Screen recording tools"
            ])

        if security_level == DRMLevel.L2:
            bypass_methods.extend([
                "HDCP bypass methods",
                "Hardware debugging interfaces",
                "Firmware modification"
            ])

        # Check for specific bypass indicators
        if not drm_info.get('license_url'):
            bypass_methods.append("Missing license server - potential offline crack")

        return bypass_methods

    async def _assess_protection_strength(self, security_level: DRMLevel, vulnerabilities: List[str]) -> str:
        """Assess overall protection strength"""
        if len(vulnerabilities) > 3:
            return "WEAK"
        elif security_level == DRMLevel.L1 and len(vulnerabilities) == 0:
            return "STRONG"
        elif security_level == DRMLevel.L2:
            return "MEDIUM"
        else:
            return "WEAK"

    async def _generate_recommendations(self, vulnerabilities: List[str], bypass_methods: List[str]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []

        if "Software-only implementation" in str(vulnerabilities):
            recommendations.append("Upgrade to hardware-based DRM (Widevine L1)")

        if "Unencrypted license transmission" in vulnerabilities:
            recommendations.append("Implement HTTPS for all license communications")

        if len(bypass_methods) > 2:
            recommendations.append("Implement additional content protection measures")

        if not recommendations:
            recommendations.append("Current DRM implementation appears robust")

        return recommendations

class PlayReadyAnalyzer:
    """Microsoft PlayReady DRM system analyzer"""

    def __init__(self):
        self.system_id = "9a04f079-9840-4286-ab92-e65be0885f95"

    async def analyze_playready_content(self, manifest_url: str, headers: Dict = None) -> DRMAnalysisResult:
        """Analyze PlayReady-protected content"""
        logger.info(f"Analyzing PlayReady content: {manifest_url}")

        try:
            # Fetch and parse manifest
            async with aiohttp.ClientSession() as session:
                async with session.get(manifest_url, headers=headers or {}) as response:
                    manifest_data = await response.text()

            # Extract PlayReady header
            pr_header = await self._extract_playready_header(manifest_data)

            # Parse license acquisition URL
            license_url = await self._extract_license_url(pr_header)

            # Analyze security features
            security_features = await self._analyze_security_features(pr_header)

            # Create DRM metadata
            drm_metadata = DRMMetadata(
                system=DRMSystem.PLAYREADY,
                version=pr_header.get('version', '4.0'),
                security_level=DRMLevel.UNKNOWN,
                key_system='com.microsoft.playready',
                license_url=license_url,
                content_id=pr_header.get('kid', ''),
                pssh_data=None,
                encryption_scheme='cenc',
                key_ids=[pr_header.get('kid', '')],
                content_type='video'
            )

            # Assess vulnerabilities
            vulnerabilities = await self._check_playready_vulnerabilities(pr_header)
            bypass_methods = await self._analyze_playready_bypass_methods(security_features)

            return DRMAnalysisResult(
                timestamp=datetime.now(),
                content_url=manifest_url,
                drm_metadata=drm_metadata,
                vulnerabilities=vulnerabilities,
                bypass_methods=bypass_methods,
                protection_strength="MEDIUM",
                analysis_confidence=0.80,
                technical_details=security_features,
                recommendations=["Implement additional output protection", "Use secure clock verification"]
            )

        except Exception as e:
            logger.error(f"PlayReady analysis failed: {e}")
            raise

    async def _extract_playready_header(self, manifest_data: str) -> Dict:
        """Extract PlayReady header from manifest"""
        pr_header = {}

        try:
            # Look for PlayReady protection header
            root = ET.fromstring(manifest_data)

            for protection in root.iter():
                if 'Protection' in protection.tag:
                    system_id = protection.get('SystemID', '')
                    if system_id.lower() == self.system_id:
                        header_elem = protection.find('.//ProtectionHeader')
                        if header_elem is not None:
                            # Decode base64 header
                            header_data = base64.b64decode(header_elem.text)
                            pr_header = await self._parse_playready_object(header_data)

        except Exception as e:
            logger.error(f"PlayReady header extraction failed: {e}")

        return pr_header

    async def _parse_playready_object(self, header_data: bytes) -> Dict:
        """Parse PlayReady Object from binary data"""
        pr_object = {}

        try:
            # PlayReady Object has specific binary structure
            if len(header_data) < 10:
                return pr_object

            # Parse basic structure
            length = struct.unpack('<I', header_data[:4])[0]
            record_count = struct.unpack('<H', header_data[4:6])[0]

            offset = 6
            for _ in range(record_count):
                if offset + 4 > len(header_data):
                    break

                record_type = struct.unpack('<H', header_data[offset:offset+2])[0]
                record_length = struct.unpack('<H', header_data[offset+2:offset+4])[0]

                if record_type == 1:  # Rights Management Header
                    record_data = header_data[offset+4:offset+4+record_length]
                    # Parse XML within the record
                    try:
                        xml_start = record_data.find(b'<WRMHEADER')
                        if xml_start >= 0:
                            xml_data = record_data[xml_start:].decode('utf-16le', errors='ignore')
                            pr_object.update(await self._parse_playready_xml(xml_data))
                    except:
                        pass

                offset += 4 + record_length

        except Exception as e:
            logger.error(f"PlayReady object parsing failed: {e}")

        return pr_object

    async def _parse_playready_xml(self, xml_data: str) -> Dict:
        """Parse PlayReady XML header"""
        pr_data = {}

        try:
            # Clean up XML
            xml_data = xml_data.split('\x00')[0]  # Remove null terminators
            root = ET.fromstring(xml_data)

            # Extract key information
            data_elem = root.find('.//DATA')
            if data_elem is not None:
                pr_data['kid'] = data_elem.find('KID')
                pr_data['checksum'] = data_elem.find('CHECKSUM')
                pr_data['la_url'] = data_elem.find('LA_URL')

            # Extract version
            version = root.get('version')
            if version:
                pr_data['version'] = version

        except Exception as e:
            logger.error(f"PlayReady XML parsing failed: {e}")

        return pr_data

    async def _extract_license_url(self, pr_header: Dict) -> str:
        """Extract license acquisition URL"""
        return pr_header.get('la_url', '')

    async def _analyze_security_features(self, pr_header: Dict) -> Dict:
        """Analyze PlayReady security features"""
        features = {
            'output_protection': False,
            'secure_clock': False,
            'secure_stop': False,
            'hardware_drm': False
        }

        # Analyze based on header content
        header_str = str(pr_header).lower()

        if 'outputprotection' in header_str:
            features['output_protection'] = True
        if 'secureclock' in header_str:
            features['secure_clock'] = True
        if 'securestop' in header_str:
            features['secure_stop'] = True

        return features

    async def _check_playready_vulnerabilities(self, pr_header: Dict) -> List[str]:
        """Check for PlayReady vulnerabilities"""
        vulnerabilities = []

        if not pr_header.get('la_url', '').startswith('https://'):
            vulnerabilities.append("Insecure license acquisition URL")

        if not pr_header.get('checksum'):
            vulnerabilities.append("Missing content integrity protection")

        return vulnerabilities

    async def _analyze_playready_bypass_methods(self, security_features: Dict) -> List[str]:
        """Analyze PlayReady bypass methods"""
        bypass_methods = []

        if not security_features.get('output_protection'):
            bypass_methods.append("Screen recording possible")

        if not security_features.get('hardware_drm'):
            bypass_methods.append("Software-based key extraction")

        bypass_methods.extend([
            "Memory dumping techniques",
            "Debug interface exploitation",
            "CDM downgrade attacks"
        ])

        return bypass_methods

class FairPlayAnalyzer:
    """Apple FairPlay DRM system analyzer"""

    def __init__(self):
        self.system_id = "94ce86fb-07ff-4f43-adb8-93d2fa968ca2"

    async def analyze_fairplay_content(self, manifest_url: str, headers: Dict = None) -> DRMAnalysisResult:
        """Analyze FairPlay-protected content"""
        logger.info(f"Analyzing FairPlay content: {manifest_url}")

        try:
            # Fetch HLS manifest
            async with aiohttp.ClientSession() as session:
                async with session.get(manifest_url, headers=headers or {}) as response:
                    manifest_data = await response.text()

            # Parse FairPlay information from HLS
            fp_info = await self._parse_fairplay_hls(manifest_data)

            # Create DRM metadata
            drm_metadata = DRMMetadata(
                system=DRMSystem.FAIRPLAY,
                version="1.0",
                security_level=DRMLevel.L1,  # FairPlay is typically hardware-based
                key_system='com.apple.fps.1_0',
                license_url=fp_info.get('key_uri', ''),
                content_id=fp_info.get('content_id', ''),
                pssh_data=None,
                encryption_scheme='cbcs',
                key_ids=[fp_info.get('keyformat', '')],
                content_type='video'
            )

            # Assess vulnerabilities
            vulnerabilities = await self._check_fairplay_vulnerabilities(fp_info)
            bypass_methods = ["Hardware extraction required", "Secure Enclave bypass", "iOS jailbreak exploitation"]

            return DRMAnalysisResult(
                timestamp=datetime.now(),
                content_url=manifest_url,
                drm_metadata=drm_metadata,
                vulnerabilities=vulnerabilities,
                bypass_methods=bypass_methods,
                protection_strength="STRONG",
                analysis_confidence=0.75,
                technical_details=fp_info,
                recommendations=["Monitor for jailbreak detection bypasses", "Implement certificate pinning"]
            )

        except Exception as e:
            logger.error(f"FairPlay analysis failed: {e}")
            raise

    async def _parse_fairplay_hls(self, manifest_data: str) -> Dict:
        """Parse FairPlay information from HLS manifest"""
        fp_info = {}

        for line in manifest_data.split('\n'):
            if line.startswith('#EXT-X-KEY:'):
                if 'com.apple.streamingkeydelivery' in line:
                    # Parse FairPlay key information
                    fp_info['method'] = 'SAMPLE-AES'

                    # Extract URI
                    uri_match = re.search(r'URI="([^"]+)"', line)
                    if uri_match:
                        fp_info['key_uri'] = uri_match.group(1)

                    # Extract KEYFORMAT
                    keyformat_match = re.search(r'KEYFORMAT="([^"]+)"', line)
                    if keyformat_match:
                        fp_info['keyformat'] = keyformat_match.group(1)

        return fp_info

    async def _check_fairplay_vulnerabilities(self, fp_info: Dict) -> List[str]:
        """Check for FairPlay vulnerabilities"""
        vulnerabilities = []

        if not fp_info.get('key_uri', '').startswith('https://'):
            vulnerabilities.append("Insecure key delivery")

        # FairPlay is generally more secure due to hardware integration
        vulnerabilities.append("Requires specialized hardware attack")

        return vulnerabilities

class DRMResearchWorker:
    """Main DRM research worker for ORACLE1"""

    def __init__(self):
        self.widevine_analyzer = WidevineAnalyzer()
        self.playready_analyzer = PlayReadyAnalyzer()
        self.fairplay_analyzer = FairPlayAnalyzer()

        # Data storage
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.influx_client = InfluxDBClient(
            url="http://localhost:8086",
            token="oracle-research-token",
            org="bev-research"
        )

        self.running = True

    async def analyze_content(self, content_url: str, drm_system: str = None, headers: Dict = None) -> DRMAnalysisResult:
        """Analyze DRM-protected content"""
        logger.info(f"Starting DRM analysis for: {content_url}")

        try:
            # Auto-detect DRM system if not specified
            if not drm_system:
                drm_system = await self._detect_drm_system(content_url, headers)

            # Route to appropriate analyzer
            if drm_system.lower() == 'widevine':
                result = await self.widevine_analyzer.analyze_widevine_content(content_url, headers)
            elif drm_system.lower() == 'playready':
                result = await self.playready_analyzer.analyze_playready_content(content_url, headers)
            elif drm_system.lower() == 'fairplay':
                result = await self.fairplay_analyzer.analyze_fairplay_content(content_url, headers)
            else:
                raise ValueError(f"Unsupported DRM system: {drm_system}")

            # Store results
            await self._store_analysis_result(result)

            logger.info(f"DRM analysis completed for {content_url}")
            return result

        except Exception as e:
            logger.error(f"DRM analysis failed: {e}")
            raise

    async def _detect_drm_system(self, content_url: str, headers: Dict = None) -> str:
        """Auto-detect DRM system from content"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(content_url, headers=headers or {}) as response:
                    content = await response.text()

            # Check for Widevine indicators
            if any(indicator in content.lower() for indicator in [
                'widevine', 'edef8ba9-79d6-4ace-a3c8-27dcd51d21ed', 'com.widevine.alpha'
            ]):
                return 'widevine'

            # Check for PlayReady indicators
            if any(indicator in content.lower() for indicator in [
                'playready', '9a04f079-9840-4286-ab92-e65be0885f95', 'com.microsoft.playready'
            ]):
                return 'playready'

            # Check for FairPlay indicators
            if any(indicator in content.lower() for indicator in [
                'fairplay', 'com.apple.fps', 'streamingkeydelivery'
            ]):
                return 'fairplay'

            return 'unknown'

        except Exception as e:
            logger.error(f"DRM detection failed: {e}")
            return 'unknown'

    async def _store_analysis_result(self, result: DRMAnalysisResult):
        """Store analysis result in databases"""
        try:
            # Store in Redis for quick access
            key = f"drm:analysis:{int(time.time())}"
            data = asdict(result)
            data['timestamp'] = result.timestamp.isoformat()
            data['drm_metadata'] = asdict(result.drm_metadata)

            self.redis_client.hset(key, mapping={k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                                                for k, v in data.items()})
            self.redis_client.expire(key, 86400 * 7)  # 7 days

            # Store in InfluxDB for time-series analysis
            point = Point("drm_analysis") \
                .tag("drm_system", result.drm_metadata.system.value) \
                .tag("security_level", result.drm_metadata.security_level.value) \
                .tag("protection_strength", result.protection_strength) \
                .field("vulnerability_count", len(result.vulnerabilities)) \
                .field("bypass_method_count", len(result.bypass_methods)) \
                .field("analysis_confidence", result.analysis_confidence) \
                .time(result.timestamp, WritePrecision.NS)

            write_api = self.influx_client.write_api()
            write_api.write(bucket="oracle-research", org="bev-research", record=point)

        except Exception as e:
            logger.error(f"Failed to store analysis result: {e}")

    async def research_drm_trends(self, days: int = 30) -> Dict:
        """Analyze DRM trends over time"""
        try:
            # Query InfluxDB for trend data
            query = f'''
            from(bucket: "oracle-research")
                |> range(start: -{days}d)
                |> filter(fn: (r) => r._measurement == "drm_analysis")
                |> group(columns: ["drm_system"])
                |> count()
            '''

            query_api = self.influx_client.query_api()
            tables = query_api.query(query)

            trends = {}
            for table in tables:
                for record in table.records:
                    drm_system = record.values.get('drm_system')
                    count = record.values.get('_value')
                    trends[drm_system] = count

            return trends

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {}

if __name__ == "__main__":
    async def main():
        worker = DRMResearchWorker()

        # Example analysis
        test_url = "https://example.com/manifest.mpd"
        result = await worker.analyze_content(test_url, 'widevine')

        print(f"Analysis completed: {result.protection_strength} protection")
        print(f"Vulnerabilities found: {len(result.vulnerabilities)}")

    asyncio.run(main())