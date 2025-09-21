#!/usr/bin/env python3
"""
DRM Circumvention Research Framework
Advanced techniques for DRM analysis and educational research
FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY
"""

import struct
import hashlib
import hmac
from Crypto.Cipher import AES, DES3, Blowfish
from Crypto.PublicKey import RSA
from Crypto.Util.Padding import unpad
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import binascii
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DRMAnalyzer:
    """Advanced DRM scheme analysis framework"""
    
    def __init__(self):
        self.known_schemes = {
            'widevine': self.analyze_widevine,
            'fairplay': self.analyze_fairplay,
            'playready': self.analyze_playready,
            'adobe_adept': self.analyze_adobe_adept,
            'steam_ceg': self.analyze_steam_ceg,
            'denuvo': self.analyze_denuvo,
            'vmprotect': self.analyze_vmprotect
        }
        self.crypto_signatures = self._load_crypto_signatures()
        
    def analyze_widevine(self, content: bytes, metadata: Dict) -> Dict:
        """Analyze Widevine DRM implementation"""
        analysis = {
            'scheme': 'widevine',
            'security_level': self._detect_security_level(content),
            'pssh_boxes': self._extract_pssh_boxes(content),
            'key_ids': [],
            'license_url': None,
            'cdm_version': None
        }
        
        # Parse PSSH boxes
        pssh_data = self._parse_pssh(content)
        if pssh_data:
            analysis['key_ids'] = pssh_data.get('key_ids', [])
            analysis['provider'] = pssh_data.get('provider')
            
        # Detect CDM version
        cdm_patterns = {
            b'com.widevine.alpha': 'L3',
            b'\x08\x01\x12\x10': 'L1',  # Hardware backed
            b'oemcrypto': 'L1'
        }
        
        for pattern, level in cdm_patterns.items():
            if pattern in content:
                analysis['security_level'] = level
                break
        
        # Extract license server
        license_patterns = [
            rb'https://.*\.widevine\.com/.*',
            b'https://.*license.*',
            b'X-AxDRM-Message'
        ]
        
        for pattern in license_patterns:
            import re
            match = re.search(pattern, content)
            if match:
                analysis['license_url'] = match.group(0).decode('utf-8', errors='ignore')
                break
        
        return analysis
    
    def analyze_fairplay(self, content: bytes, metadata: Dict) -> Dict:
        """Analyze Apple FairPlay DRM"""
        analysis = {
            'scheme': 'fairplay',
            'version': self._detect_fairplay_version(content),
            'sinf_atoms': [],
            'key_server': None,
            'asset_id': None
        }
        
        # Parse M4V/MOV atoms
        atoms = self._parse_atoms(content)
        
        # Extract SINF (protection info)
        sinf_data = atoms.get('sinf')
        if sinf_data:
            analysis['sinf_atoms'] = self._parse_sinf(sinf_data)
            
        # Find key server URL
        if b'fps.apple.com' in content:
            analysis['key_server'] = 'fps.apple.com'
            
        # Extract asset ID
        if b'assetId' in content:
            idx = content.find(b'assetId')
            analysis['asset_id'] = content[idx:idx+50].decode('utf-8', errors='ignore')
        
        return analysis
    
    def analyze_playready(self, content: bytes, metadata: Dict) -> Dict:
        """Analyze Microsoft PlayReady DRM"""
        analysis = {
            'scheme': 'playready',
            'version': None,
            'header': None,
            'license_url': None,
            'key_ids': []
        }
        
        # Find PlayReady header
        pr_header_start = content.find(b'<WRMHEADER')
        if pr_header_start != -1:
            pr_header_end = content.find(b'</WRMHEADER>', pr_header_start)
            if pr_header_end != -1:
                header = content[pr_header_start:pr_header_end+12]
                analysis['header'] = base64.b64encode(header).decode()
                
                # Parse header for LA_URL
                if b'LA_URL' in header:
                    url_start = header.find(b'LA_URL>') + 7
                    url_end = header.find(b'</LA_URL', url_start)
                    analysis['license_url'] = header[url_start:url_end].decode()
        
        # Extract KIDs
        kid_pattern = b'KID>(.*?)</KID'
        import re
        kids = re.findall(kid_pattern, content)
        analysis['key_ids'] = [base64.b64encode(kid).decode() for kid in kids]
        
        return analysis
    
    def analyze_adobe_adept(self, content: bytes, metadata: Dict) -> Dict:
        """Analyze Adobe ADEPT (Digital Editions) DRM"""
        analysis = {
            'scheme': 'adobe_adept',
            'encryption_type': None,
            'rights_xml': None,
            'device_id': None,
            'user_uuid': None
        }
        
        # Check for ADEPT signatures
        if b'adept:ns' in content or b'adobe-adept' in content:
            analysis['encryption_type'] = 'ADEPT'
            
        # Extract rights.xml
        rights_start = content.find(b'<rightsxml')
        if rights_start != -1:
            rights_end = content.find(b'</rightsxml>', rights_start)
            if rights_end != -1:
                analysis['rights_xml'] = content[rights_start:rights_end+12].decode('utf-8', errors='ignore')
        
        # Find device ID
        if b'deviceID' in content:
            idx = content.find(b'deviceID')
            analysis['device_id'] = content[idx:idx+50].decode('utf-8', errors='ignore')
        
        return analysis
    
    def analyze_steam_ceg(self, content: bytes, metadata: Dict) -> Dict:
        """Analyze Steam CEG (Custom Executable Generation)"""
        analysis = {
            'scheme': 'steam_ceg',
            'stub_version': None,
            'encrypted_sections': [],
            'steam_api_calls': [],
            'app_id': None
        }
        
        # Check for Steam stub
        if b'steam_api' in content.lower() or b'steamclient' in content.lower():
            analysis['has_steam_drm'] = True
            
        # Find encrypted sections
        section_markers = [b'.steam', b'.bind', b'.ceg']
        for marker in section_markers:
            if marker in content:
                analysis['encrypted_sections'].append(marker.decode())
        
        # Extract Steam API calls
        api_functions = [
            b'SteamAPI_Init',
            b'SteamAPI_RestartAppIfNecessary',
            b'SteamUser',
            b'SteamApps'
        ]
        
        for func in api_functions:
            if func in content:
                analysis['steam_api_calls'].append(func.decode())
        
        # Try to find App ID
        app_id_pattern = rb'steam_appid\.txt'
        if app_id_pattern in content:
            analysis['app_id'] = 'Found in steam_appid.txt'
        
        return analysis
    
    def analyze_denuvo(self, content: bytes, metadata: Dict) -> Dict:
        """Analyze Denuvo Anti-Tamper"""
        analysis = {
            'scheme': 'denuvo',
            'version': None,
            'vm_sections': [],
            'triggers': [],
            'hardware_id_checks': False
        }
        
        # Denuvo signatures
        denuvo_sigs = [
            b'\x48\x8D\x05',  # VM entry
            b'\x0F\x1F\x44\x00\x00',  # NOP padding
            b'denuvo',
            b'.00cfg'  # Config section
        ]
        
        for sig in denuvo_sigs:
            if sig in content:
                analysis['triggers'].append(f"Signature: {sig.hex()}")
        
        # Check for VM sections
        if b'.vmp0' in content or b'.vmp1' in content:
            analysis['vm_sections'].append('VMProtect layers detected')
            
        # Hardware ID checks
        hwid_funcs = [b'GetVolumeInformation', b'GetComputerName', b'cpuid']
        for func in hwid_funcs:
            if func in content:
                analysis['hardware_id_checks'] = True
                break
        
        return analysis
    
    def analyze_vmprotect(self, content: bytes, metadata: Dict) -> Dict:
        """Analyze VMProtect virtualization"""
        analysis = {
            'scheme': 'vmprotect',
            'version': None,
            'virtualized_functions': [],
            'mutation_level': None,
            'anti_debug': []
        }
        
        # VMProtect markers
        vmp_sigs = [
            b'VMProtect',
            b'.vmp0', b'.vmp1', b'.vmp2',
            b'VirtualProtect',
            b'\xE9\x00\x00\x00\x00'  # JMP pattern
        ]
        
        for sig in vmp_sigs:
            if sig in content:
                analysis['virtualized_functions'].append(f"Pattern: {sig.hex()}")
        
        # Anti-debug techniques
        debug_checks = [
            (b'IsDebuggerPresent', 'IsDebuggerPresent'),
            (b'CheckRemoteDebuggerPresent', 'CheckRemoteDebugger'),
            (b'NtQueryInformationProcess', 'NtQueryInfo'),
            (b'\xCC', 'INT3 breakpoints')
        ]
        
        for pattern, name in debug_checks:
            if pattern in content:
                analysis['anti_debug'].append(name)
        
        # Estimate mutation level
        entropy = self._calculate_entropy(content[:10000])
        if entropy > 7.5:
            analysis['mutation_level'] = 'Ultra'
        elif entropy > 6.5:
            analysis['mutation_level'] = 'High'
        else:
            analysis['mutation_level'] = 'Standard'
        
        return analysis
    
    def identify_scheme(self, content: bytes) -> List[str]:
        """Identify DRM schemes present in content"""
        detected_schemes = []
        
        # Signature-based detection
        signatures = {
            'widevine': [b'widevine', b'pssh', b'oemcrypto'],
            'fairplay': [b'sinf', b'schi', b'schm'],
            'playready': [b'WRMHEADER', b'PlayReady', b'microsoft.com'],
            'adobe_adept': [b'adept:ns', b'adobe-adept', b'AdobeID'],
            'steam_ceg': [b'steam_api', b'SteamAPI_', b'.steam'],
            'denuvo': [b'denuvo', b'.00cfg'],
            'vmprotect': [b'VMProtect', b'.vmp']
        }
        
        for scheme, sigs in signatures.items():
            for sig in sigs:
                if sig.lower() in content.lower():
                    detected_schemes.append(scheme)
                    break
        
        return list(set(detected_schemes))
    
    def comprehensive_analysis(self, file_path: str) -> Dict:
        """Perform comprehensive DRM analysis"""
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Identify schemes
        schemes = self.identify_scheme(content)
        logger.info(f"Detected schemes: {schemes}")
        
        # Analyze each scheme
        results = {
            'file': file_path,
            'schemes_detected': schemes,
            'detailed_analysis': {},
            'crypto_analysis': self._analyze_cryptography(content),
            'obfuscation_level': self._assess_obfuscation(content)
        }
        
        metadata = {'file_path': file_path, 'size': len(content)}
        
        for scheme in schemes:
            if scheme in self.known_schemes:
                try:
                    results['detailed_analysis'][scheme] = self.known_schemes[scheme](content, metadata)
                except Exception as e:
                    logger.error(f"Error analyzing {scheme}: {e}")
                    results['detailed_analysis'][scheme] = {'error': str(e)}
        
        return results
    
    def _detect_security_level(self, content: bytes) -> str:
        """Detect DRM security level"""
        if b'L1' in content or b'oemcrypto' in content:
            return 'L1 (Hardware)'
        elif b'L2' in content:
            return 'L2 (Hybrid)'
        else:
            return 'L3 (Software)'
    
    def _extract_pssh_boxes(self, content: bytes) -> List[Dict]:
        """Extract and parse PSSH boxes"""
        pssh_boxes = []
        pssh_marker = b'pssh'
        
        idx = 0
        while True:
            idx = content.find(pssh_marker, idx)
            if idx == -1:
                break
            
            # Read box size
            if idx >= 4:
                size = struct.unpack('>I', content[idx-4:idx])[0]
                box_data = content[idx-4:idx-4+size]
                
                pssh_boxes.append({
                    'offset': idx-4,
                    'size': size,
                    'data': base64.b64encode(box_data).decode()
                })
            
            idx += 4
        
        return pssh_boxes
    
    def _parse_pssh(self, content: bytes) -> Optional[Dict]:
        """Parse PSSH box data"""
        try:
            pssh_boxes = self._extract_pssh_boxes(content)
            if not pssh_boxes:
                return None
            
            # Parse first PSSH box
            pssh_data = base64.b64decode(pssh_boxes[0]['data'])
            
            result = {
                'version': pssh_data[8] if len(pssh_data) > 8 else 0,
                'system_id': pssh_data[12:28].hex() if len(pssh_data) > 28 else None,
                'key_ids': []
            }
            
            # Extract KIDs if present
            if len(pssh_data) > 32:
                kid_count = pssh_data[30]
                for i in range(kid_count):
                    start = 32 + (i * 16)
                    if start + 16 <= len(pssh_data):
                        result['key_ids'].append(pssh_data[start:start+16].hex())
            
            return result
        except:
            return None
    
    def _detect_fairplay_version(self, content: bytes) -> Optional[str]:
        """Detect FairPlay version"""
        version_markers = {
            b'fpshls': 'FairPlay Streaming',
            b'fps2': 'FairPlay 2.0',
            b'fps3': 'FairPlay 3.0',
            b'fps4': 'FairPlay 4.0'
        }
        
        for marker, version in version_markers.items():
            if marker in content:
                return version
        
        return 'Unknown'
    
    def _parse_atoms(self, content: bytes) -> Dict:
        """Parse MP4/MOV atoms"""
        atoms = {}
        idx = 0
        
        while idx < len(content) - 8:
            try:
                size = struct.unpack('>I', content[idx:idx+4])[0]
                atom_type = content[idx+4:idx+8].decode('ascii', errors='ignore')
                
                if size > 8 and size < len(content) - idx:
                    atoms[atom_type] = content[idx+8:idx+size]
                    idx += size
                else:
                    idx += 8
            except:
                idx += 8
        
        return atoms
    
    def _parse_sinf(self, sinf_data: bytes) -> List[Dict]:
        """Parse SINF atom for FairPlay"""
        sinf_info = []
        
        # Parse sub-atoms
        sub_atoms = ['frma', 'schm', 'schi']
        for atom in sub_atoms:
            atom_bytes = atom.encode()
            if atom_bytes in sinf_data:
                idx = sinf_data.find(atom_bytes)
                sinf_info.append({
                    'type': atom,
                    'offset': idx,
                    'data': sinf_data[idx:idx+20].hex()
                })
        
        return sinf_info
    
    def _load_crypto_signatures(self) -> Dict:
        """Load known cryptographic signatures"""
        return {
            'aes': [b'Rijndael', b'AES'],
            'des': [b'DES', b'TripleDES'],
            'rsa': [b'RSA', b'PKCS', b'-----BEGIN'],
            'ecc': [b'secp256', b'EC', b'elliptic'],
            'chacha': [b'ChaCha', b'Poly1305']
        }
    
    def _analyze_cryptography(self, content: bytes) -> Dict:
        """Analyze cryptographic implementations"""
        crypto_found = {
            'algorithms': [],
            'key_sizes': [],
            'modes': [],
            'padding': []
        }
        
        # Check for crypto algorithms
        for algo, sigs in self.crypto_signatures.items():
            for sig in sigs:
                if sig in content:
                    crypto_found['algorithms'].append(algo)
                    break
        
        # Check for crypto modes
        modes = {
            b'CBC': 'CBC',
            b'ECB': 'ECB',
            b'CTR': 'CTR',
            b'GCM': 'GCM',
            b'OFB': 'OFB',
            b'CFB': 'CFB'
        }
        
        for mode_sig, mode_name in modes.items():
            if mode_sig in content:
                crypto_found['modes'].append(mode_name)
        
        # Check for padding schemes
        padding_schemes = {
            b'PKCS7': 'PKCS7',
            b'PKCS5': 'PKCS5',
            b'ISO10126': 'ISO10126',
            b'ANSI X9.23': 'ANSI X9.23'
        }
        
        for pad_sig, pad_name in padding_schemes.items():
            if pad_sig in content:
                crypto_found['padding'].append(pad_name)
        
        # Detect key sizes
        key_patterns = [
            (b'\x00\x00\x00\x10', '128-bit'),
            (b'\x00\x00\x00\x18', '192-bit'),
            (b'\x00\x00\x00\x20', '256-bit')
        ]
        
        for pattern, size in key_patterns:
            if pattern in content:
                crypto_found['key_sizes'].append(size)
        
        return crypto_found
    
    def _assess_obfuscation(self, content: bytes) -> Dict:
        """Assess code obfuscation level"""
        obfuscation = {
            'entropy': self._calculate_entropy(content[:10000]),
            'techniques': [],
            'level': 'Low'
        }
        
        # Check for obfuscation techniques
        techniques = {
            'packing': [b'UPX', b'ASPack', b'Themida'],
            'virtualization': [b'.vmp', b'Code Virtualizer'],
            'encryption': [b'encrypted', b'crypt', b'cipher'],
            'anti_debug': [b'IsDebuggerPresent', b'int 3', b'\xCC'],
            'anti_vm': [b'VMware', b'VirtualBox', b'QEMU'],
            'control_flow': [b'\xEB\xFE', b'\xE9']  # Infinite loops, JMPs
        }
        
        for technique, sigs in techniques.items():
            for sig in sigs:
                if sig in content:
                    obfuscation['techniques'].append(technique)
                    break
        
        # Assess level
        if obfuscation['entropy'] > 7.5 or len(obfuscation['techniques']) > 4:
            obfuscation['level'] = 'Ultra'
        elif obfuscation['entropy'] > 6.5 or len(obfuscation['techniques']) > 2:
            obfuscation['level'] = 'High'
        elif obfuscation['entropy'] > 5.5 or len(obfuscation['techniques']) > 0:
            obfuscation['level'] = 'Medium'
        
        return obfuscation
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy"""
        if not data:
            return 0
        
        freq = {}
        for byte in data:
            freq[byte] = freq.get(byte, 0) + 1
        
        entropy = 0
        for count in freq.values():
            prob = count / len(data)
            entropy -= prob * np.log2(prob)
        
        return entropy


class KeyExtractor:
    """Advanced key extraction and derivation"""
    
    def __init__(self):
        self.extraction_methods = {
            'memory_dump': self.extract_from_memory,
            'network_capture': self.extract_from_network,
            'binary_analysis': self.extract_from_binary,
            'side_channel': self.extract_via_side_channel
        }
    
    def extract_from_memory(self, dump_path: str) -> List[Dict]:
        """Extract keys from memory dumps"""
        keys_found = []
        
        with open(dump_path, 'rb') as f:
            dump = f.read()
        
        # Common key patterns
        key_patterns = [
            # AES keys (16, 24, 32 bytes of high entropy)
            {'name': 'AES-128', 'size': 16},
            {'name': 'AES-192', 'size': 24},
            {'name': 'AES-256', 'size': 32},
            # RSA components
            {'name': 'RSA-2048', 'size': 256},
            {'name': 'RSA-4096', 'size': 512}
        ]
        
        for pattern in key_patterns:
            candidates = self._find_high_entropy_regions(dump, pattern['size'])
            
            for candidate in candidates:
                if self._validate_key(candidate, pattern['name']):
                    keys_found.append({
                        'type': pattern['name'],
                        'key': binascii.hexlify(candidate).decode(),
                        'offset': dump.find(candidate)
                    })
        
        return keys_found
    
    def extract_from_network(self, pcap_path: str) -> List[Dict]:
        """Extract keys from network captures"""
        # This would use scapy or similar for real implementation
        keys = []
        
        # Simplified example
        with open(pcap_path, 'rb') as f:
            data = f.read()
        
        # Look for TLS handshake, license exchanges
        tls_patterns = [
            b'\x16\x03',  # TLS handshake
            b'license',
            b'key_exchange'
        ]
        
        for pattern in tls_patterns:
            if pattern in data:
                idx = data.find(pattern)
                # Extract potential key material
                potential_key = data[idx:idx+64]
                
                keys.append({
                    'source': 'network',
                    'pattern': pattern.hex(),
                    'data': binascii.hexlify(potential_key).decode()
                })
        
        return keys
    
    def extract_from_binary(self, binary_path: str) -> List[Dict]:
        """Extract hardcoded keys from binaries"""
        keys = []
        
        with open(binary_path, 'rb') as f:
            binary = f.read()
        
        # Look for hardcoded keys
        # High entropy regions that are not code
        regions = self._find_data_sections(binary)
        
        for region_start, region_end in regions:
            section = binary[region_start:region_end]
            
            # Check for key-like patterns
            for size in [16, 24, 32, 64, 128, 256]:
                candidates = self._find_high_entropy_regions(section, size)
                
                for candidate in candidates:
                    keys.append({
                        'source': 'binary',
                        'offset': region_start + section.find(candidate),
                        'size': len(candidate),
                        'key': binascii.hexlify(candidate).decode()
                    })
        
        return keys
    
    def extract_via_side_channel(self, target_process: str) -> List[Dict]:
        """Side-channel key extraction (timing, cache, power)"""
        # This is a simplified demonstration
        keys = []
        
        # Timing attack simulation
        timing_data = self._collect_timing_data(target_process)
        
        # Analyze timing patterns for key bits
        key_bits = self._analyze_timing_patterns(timing_data)
        
        if key_bits:
            keys.append({
                'method': 'timing_analysis',
                'confidence': 0.7,
                'key_bits': key_bits
            })
        
        return keys
    
    def _find_high_entropy_regions(self, data: bytes, size: int) -> List[bytes]:
        """Find high entropy regions of specific size"""
        regions = []
        
        for i in range(0, len(data) - size, 1):
            chunk = data[i:i+size]
            entropy = self._calculate_entropy(chunk)
            
            if entropy > 7.0:  # High entropy threshold
                regions.append(chunk)
        
        return regions[:100]  # Limit results
    
    def _validate_key(self, candidate: bytes, key_type: str) -> bool:
        """Validate if candidate is likely a real key"""
        # Check entropy
        entropy = self._calculate_entropy(candidate)
        if entropy < 6.0:
            return False
        
        # Check for patterns that indicate non-keys
        if candidate.startswith(b'\x00' * 4) or candidate.startswith(b'\xFF' * 4):
            return False
        
        # Type-specific validation
        if 'AES' in key_type:
            # AES keys should be random
            return entropy > 7.0
        elif 'RSA' in key_type:
            # RSA components have specific structures
            return candidate[0] != 0 and candidate[-1] != 0
        
        return True
    
    def _find_data_sections(self, binary: bytes) -> List[Tuple[int, int]]:
        """Find data sections in binary"""
        sections = []
        
        # Simplified PE/ELF section finder
        # Look for section headers
        if binary.startswith(b'MZ'):  # PE
            # Find PE header
            pe_offset = struct.unpack('<I', binary[0x3C:0x40])[0]
            # This would parse PE sections properly
            sections.append((0x1000, min(0x10000, len(binary))))
        elif binary.startswith(b'\x7FELF'):  # ELF
            # This would parse ELF sections
            sections.append((0x1000, min(0x10000, len(binary))))
        
        return sections
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy"""
        if not data:
            return 0
        
        freq = {}
        for byte in data:
            freq[byte] = freq.get(byte, 0) + 1
        
        entropy = 0
        for count in freq.values():
            prob = count / len(data)
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _collect_timing_data(self, target: str) -> List[float]:
        """Simulate timing data collection"""
        # In reality, this would measure actual execution times
        import random
        return [random.uniform(0.001, 0.01) for _ in range(1000)]
    
    def _analyze_timing_patterns(self, timing_data: List[float]) -> Optional[str]:
        """Analyze timing patterns for key extraction"""
        # Simplified analysis
        if not timing_data:
            return None
        
        # Statistical analysis would reveal key bits
        mean_time = np.mean(timing_data)
        std_time = np.std(timing_data)
        
        # Threshold-based bit extraction
        key_bits = []
        for time in timing_data[:256]:  # Extract 256 bits max
            if time > mean_time + std_time:
                key_bits.append('1')
            else:
                key_bits.append('0')
        
        return ''.join(key_bits)


if __name__ == "__main__":
    # Example usage
    analyzer = DRMAnalyzer()
    extractor = KeyExtractor()
    
    # Analyze DRM
    # results = analyzer.comprehensive_analysis("protected_file.bin")
    
    # Extract keys
    # keys = extractor.extract_from_binary("protected.exe")
    
    print("DRM Research Framework loaded - Educational purposes only!")
