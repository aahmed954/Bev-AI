#!/usr/bin/env python3
"""
Watermark Analysis Worker for ORACLE1
Advanced watermark detection and analysis for digital content
Supports LSB, frequency domain, pattern-based, and invisible watermarks
"""

import asyncio
import json
import time
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import aiohttp
import aiofiles
import io
import base64

# Image processing libraries
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np
from scipy import fft, signal, ndimage
from scipy.stats import entropy
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Audio processing
import librosa
import soundfile as sf
from scipy.io import wavfile

# ML libraries
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Video processing
import tempfile
import subprocess

# ORACLE integration
import redis
from influxdb_client import InfluxDBClient, Point, WritePrecision

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WatermarkType(Enum):
    LSB = "lsb"  # Least Significant Bit
    DCT = "dct"  # Discrete Cosine Transform
    DWT = "dwt"  # Discrete Wavelet Transform
    SPREAD_SPECTRUM = "spread_spectrum"
    PATTERN_BASED = "pattern_based"
    INVISIBLE = "invisible"
    VISIBLE = "visible"
    AUDIO_ECHO = "audio_echo"
    AUDIO_PHASE = "audio_phase"
    VIDEO_TEMPORAL = "video_temporal"
    UNKNOWN = "unknown"

class WatermarkDomain(Enum):
    SPATIAL = "spatial"
    FREQUENCY = "frequency"
    TEMPORAL = "temporal"
    HYBRID = "hybrid"

@dataclass
class WatermarkMetadata:
    """Watermark metadata information"""
    watermark_type: WatermarkType
    domain: WatermarkDomain
    strength: float  # 0.0 to 1.0
    location: Tuple[int, int, int, int]  # x, y, width, height
    extraction_method: str
    confidence: float
    embedded_data: Optional[str]
    algorithm_fingerprint: str

@dataclass
class WatermarkAnalysisResult:
    """Result of watermark analysis"""
    timestamp: datetime
    content_url: str
    content_type: str  # image, audio, video
    watermarks_found: List[WatermarkMetadata]
    analysis_methods_used: List[str]
    processing_time: float
    content_hash: str
    technical_details: Dict[str, Any]
    recommendations: List[str]

class LSBWatermarkDetector:
    """Least Significant Bit watermark detector"""

    def __init__(self):
        self.name = "LSB Detector"

    async def detect_image_lsb(self, image_data: np.ndarray) -> List[WatermarkMetadata]:
        """Detect LSB watermarks in images"""
        watermarks = []

        try:
            # Convert to numpy array if needed
            if len(image_data.shape) == 3:
                # Analyze each color channel
                for channel in range(image_data.shape[2]):
                    channel_data = image_data[:, :, channel]
                    lsb_data = await self._extract_lsb_plane(channel_data)

                    if await self._analyze_lsb_randomness(lsb_data):
                        watermark = WatermarkMetadata(
                            watermark_type=WatermarkType.LSB,
                            domain=WatermarkDomain.SPATIAL,
                            strength=await self._calculate_lsb_strength(lsb_data),
                            location=(0, 0, image_data.shape[1], image_data.shape[0]),
                            extraction_method=f"LSB channel {channel}",
                            confidence=0.7,
                            embedded_data=await self._extract_lsb_message(lsb_data),
                            algorithm_fingerprint=await self._generate_lsb_fingerprint(lsb_data)
                        )
                        watermarks.append(watermark)

        except Exception as e:
            logger.error(f"LSB detection failed: {e}")

        return watermarks

    async def _extract_lsb_plane(self, channel_data: np.ndarray) -> np.ndarray:
        """Extract LSB plane from image channel"""
        return channel_data & 1

    async def _analyze_lsb_randomness(self, lsb_data: np.ndarray) -> bool:
        """Analyze if LSB plane contains hidden data"""
        try:
            # Calculate entropy
            hist, _ = np.histogram(lsb_data.flatten(), bins=2, range=(0, 1))
            ent = entropy(hist + 1e-10)  # Add small value to avoid log(0)

            # LSB with hidden data should have higher entropy
            # Natural images typically have entropy < 0.9 in LSB
            return ent > 0.95

        except Exception as e:
            logger.error(f"LSB randomness analysis failed: {e}")
            return False

    async def _calculate_lsb_strength(self, lsb_data: np.ndarray) -> float:
        """Calculate watermark strength in LSB plane"""
        try:
            # Calculate ratio of modified bits
            total_bits = lsb_data.size
            modified_bits = np.sum(lsb_data)

            # Strength based on deviation from 50/50 distribution
            ratio = modified_bits / total_bits
            strength = abs(ratio - 0.5) * 2  # Normalize to 0-1

            return min(strength, 1.0)

        except Exception as e:
            logger.error(f"LSB strength calculation failed: {e}")
            return 0.0

    async def _extract_lsb_message(self, lsb_data: np.ndarray) -> Optional[str]:
        """Extract embedded message from LSB data"""
        try:
            # Convert LSB plane to binary string
            binary_string = ''.join(lsb_data.flatten().astype(str))

            # Try to extract text (8-bit ASCII)
            message = ""
            for i in range(0, len(binary_string) - 7, 8):
                byte = binary_string[i:i+8]
                if len(byte) == 8:
                    char_code = int(byte, 2)
                    if 32 <= char_code <= 126:  # Printable ASCII
                        message += chr(char_code)
                    else:
                        break

            return message if len(message) > 4 else None

        except Exception as e:
            logger.error(f"LSB message extraction failed: {e}")
            return None

    async def _generate_lsb_fingerprint(self, lsb_data: np.ndarray) -> str:
        """Generate algorithm fingerprint for LSB watermark"""
        try:
            # Create fingerprint based on statistical properties
            features = [
                np.mean(lsb_data),
                np.std(lsb_data),
                entropy(np.histogram(lsb_data.flatten(), bins=10)[0] + 1e-10)
            ]

            fingerprint_data = json.dumps(features)
            return hashlib.md5(fingerprint_data.encode()).hexdigest()[:16]

        except Exception as e:
            logger.error(f"LSB fingerprint generation failed: {e}")
            return "unknown"

class FrequencyWatermarkDetector:
    """Frequency domain watermark detector (DCT, DWT, FFT)"""

    def __init__(self):
        self.name = "Frequency Domain Detector"

    async def detect_frequency_watermarks(self, image_data: np.ndarray) -> List[WatermarkMetadata]:
        """Detect frequency domain watermarks"""
        watermarks = []

        try:
            # Convert to grayscale if needed
            if len(image_data.shape) == 3:
                gray_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image_data

            # DCT-based detection
            dct_watermark = await self._detect_dct_watermark(gray_image)
            if dct_watermark:
                watermarks.append(dct_watermark)

            # DWT-based detection
            dwt_watermark = await self._detect_dwt_watermark(gray_image)
            if dwt_watermark:
                watermarks.append(dwt_watermark)

            # FFT-based detection
            fft_watermark = await self._detect_fft_watermark(gray_image)
            if fft_watermark:
                watermarks.append(fft_watermark)

        except Exception as e:
            logger.error(f"Frequency domain detection failed: {e}")

        return watermarks

    async def _detect_dct_watermark(self, image: np.ndarray) -> Optional[WatermarkMetadata]:
        """Detect DCT-based watermarks"""
        try:
            # Apply DCT to 8x8 blocks
            height, width = image.shape
            dct_blocks = []

            for i in range(0, height - 7, 8):
                for j in range(0, width - 7, 8):
                    block = image[i:i+8, j:j+8].astype(np.float32)
                    dct_block = cv2.dct(block)
                    dct_blocks.append(dct_block)

            if not dct_blocks:
                return None

            # Analyze DCT coefficients for watermark patterns
            coefficients = np.array(dct_blocks)

            # Look for systematic modifications in mid-frequency coefficients
            mid_freq_variance = np.var(coefficients[:, 2:6, 2:6])

            if mid_freq_variance > 100:  # Threshold for watermark detection
                return WatermarkMetadata(
                    watermark_type=WatermarkType.DCT,
                    domain=WatermarkDomain.FREQUENCY,
                    strength=min(mid_freq_variance / 1000, 1.0),
                    location=(0, 0, width, height),
                    extraction_method="DCT coefficient analysis",
                    confidence=0.6,
                    embedded_data=None,
                    algorithm_fingerprint=f"dct_{int(mid_freq_variance)}"
                )

        except Exception as e:
            logger.error(f"DCT watermark detection failed: {e}")

        return None

    async def _detect_dwt_watermark(self, image: np.ndarray) -> Optional[WatermarkMetadata]:
        """Detect DWT-based watermarks"""
        try:
            # Simple Haar wavelet transform
            coeffs = self._haar_dwt_2d(image)

            if coeffs is None:
                return None

            # Analyze high-frequency subbands for watermarks
            _, (lh, hl, hh) = coeffs

            # Calculate energy in high-frequency subbands
            hf_energy = np.mean(lh**2) + np.mean(hl**2) + np.mean(hh**2)

            if hf_energy > 50:  # Threshold for watermark detection
                return WatermarkMetadata(
                    watermark_type=WatermarkType.DWT,
                    domain=WatermarkDomain.FREQUENCY,
                    strength=min(hf_energy / 500, 1.0),
                    location=(0, 0, image.shape[1], image.shape[0]),
                    extraction_method="DWT subband analysis",
                    confidence=0.5,
                    embedded_data=None,
                    algorithm_fingerprint=f"dwt_{int(hf_energy)}"
                )

        except Exception as e:
            logger.error(f"DWT watermark detection failed: {e}")

        return None

    def _haar_dwt_2d(self, image: np.ndarray) -> Optional[Tuple]:
        """Simple 2D Haar wavelet transform"""
        try:
            h, w = image.shape
            if h % 2 != 0 or w % 2 != 0:
                return None

            # First level decomposition
            # Low-pass filter: [1, 1] / sqrt(2)
            # High-pass filter: [1, -1] / sqrt(2)

            # Row-wise filtering
            rows_filtered = np.zeros_like(image, dtype=np.float32)
            for i in range(h):
                row = image[i, :].astype(np.float32)
                # Downsample and filter
                low = (row[::2] + row[1::2]) / np.sqrt(2)
                high = (row[::2] - row[1::2]) / np.sqrt(2)
                rows_filtered[i, :w//2] = low
                rows_filtered[i, w//2:] = high

            # Column-wise filtering
            ll = np.zeros((h//2, w//2), dtype=np.float32)
            lh = np.zeros((h//2, w//2), dtype=np.float32)
            hl = np.zeros((h//2, w//2), dtype=np.float32)
            hh = np.zeros((h//2, w//2), dtype=np.float32)

            for j in range(w//2):
                col_low = rows_filtered[:, j]
                ll[:, j] = (col_low[::2] + col_low[1::2]) / np.sqrt(2)
                lh[:, j] = (col_low[::2] - col_low[1::2]) / np.sqrt(2)

            for j in range(w//2, w):
                col_high = rows_filtered[:, j]
                hl[:, j-w//2] = (col_high[::2] + col_high[1::2]) / np.sqrt(2)
                hh[:, j-w//2] = (col_high[::2] - col_high[1::2]) / np.sqrt(2)

            return ll, (lh, hl, hh)

        except Exception as e:
            logger.error(f"Haar DWT failed: {e}")
            return None

    async def _detect_fft_watermark(self, image: np.ndarray) -> Optional[WatermarkMetadata]:
        """Detect FFT-based watermarks"""
        try:
            # Apply 2D FFT
            fft_image = np.fft.fft2(image)
            fft_magnitude = np.abs(fft_image)

            # Analyze frequency domain for periodic patterns
            # Look for peaks in the magnitude spectrum
            threshold = np.mean(fft_magnitude) + 3 * np.std(fft_magnitude)
            peaks = fft_magnitude > threshold
            peak_count = np.sum(peaks)

            # Exclude DC component and natural image frequencies
            center_x, center_y = fft_magnitude.shape[0] // 2, fft_magnitude.shape[1] // 2
            peaks[center_x-5:center_x+5, center_y-5:center_y+5] = False

            significant_peaks = np.sum(peaks)

            if significant_peaks > 10:  # Threshold for watermark detection
                return WatermarkMetadata(
                    watermark_type=WatermarkType.SPREAD_SPECTRUM,
                    domain=WatermarkDomain.FREQUENCY,
                    strength=min(significant_peaks / 100, 1.0),
                    location=(0, 0, image.shape[1], image.shape[0]),
                    extraction_method="FFT peak analysis",
                    confidence=0.4,
                    embedded_data=None,
                    algorithm_fingerprint=f"fft_{significant_peaks}"
                )

        except Exception as e:
            logger.error(f"FFT watermark detection failed: {e}")

        return None

class PatternWatermarkDetector:
    """Pattern-based watermark detector"""

    def __init__(self):
        self.name = "Pattern Detector"
        self.known_patterns = self._load_known_patterns()

    def _load_known_patterns(self) -> Dict[str, np.ndarray]:
        """Load known watermark patterns"""
        patterns = {}

        # Create some common watermark patterns
        # Copyright symbol pattern
        patterns['copyright'] = self._create_copyright_pattern()

        # Company logo patterns (simplified)
        patterns['logo_generic'] = self._create_generic_logo_pattern()

        return patterns

    def _create_copyright_pattern(self) -> np.ndarray:
        """Create a copyright symbol pattern"""
        pattern = np.zeros((16, 16), dtype=np.uint8)
        # Simple circle with 'C' inside
        cv2.circle(pattern, (8, 8), 6, 255, 2)
        cv2.putText(pattern, 'C', (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        return pattern

    def _create_generic_logo_pattern(self) -> np.ndarray:
        """Create a generic logo pattern"""
        pattern = np.zeros((20, 20), dtype=np.uint8)
        cv2.rectangle(pattern, (2, 2), (18, 18), 255, 2)
        cv2.line(pattern, (6, 6), (14, 14), 255, 2)
        return pattern

    async def detect_pattern_watermarks(self, image_data: np.ndarray) -> List[WatermarkMetadata]:
        """Detect pattern-based watermarks"""
        watermarks = []

        try:
            # Convert to grayscale
            if len(image_data.shape) == 3:
                gray_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image_data

            # Search for known patterns
            for pattern_name, pattern in self.known_patterns.items():
                matches = await self._template_match(gray_image, pattern)

                for match in matches:
                    watermark = WatermarkMetadata(
                        watermark_type=WatermarkType.PATTERN_BASED,
                        domain=WatermarkDomain.SPATIAL,
                        strength=match['confidence'],
                        location=match['location'],
                        extraction_method=f"Template matching: {pattern_name}",
                        confidence=match['confidence'],
                        embedded_data=pattern_name,
                        algorithm_fingerprint=f"pattern_{pattern_name}"
                    )
                    watermarks.append(watermark)

            # Look for repeated patterns
            repeated_patterns = await self._detect_repeated_patterns(gray_image)
            watermarks.extend(repeated_patterns)

        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")

        return watermarks

    async def _template_match(self, image: np.ndarray, pattern: np.ndarray) -> List[Dict]:
        """Perform template matching"""
        matches = []

        try:
            # Multi-scale template matching
            scales = [0.5, 0.8, 1.0, 1.2, 1.5]

            for scale in scales:
                if scale != 1.0:
                    h, w = pattern.shape
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled_pattern = cv2.resize(pattern, (new_w, new_h))
                else:
                    scaled_pattern = pattern

                # Template matching
                result = cv2.matchTemplate(image, scaled_pattern, cv2.TM_CCOEFF_NORMED)

                # Find matches above threshold
                threshold = 0.7
                locations = np.where(result >= threshold)

                for pt in zip(*locations[::-1]):
                    confidence = result[pt[1], pt[0]]
                    h, w = scaled_pattern.shape

                    matches.append({
                        'location': (pt[0], pt[1], w, h),
                        'confidence': confidence
                    })

        except Exception as e:
            logger.error(f"Template matching failed: {e}")

        return matches

    async def _detect_repeated_patterns(self, image: np.ndarray) -> List[WatermarkMetadata]:
        """Detect repeated patterns in image"""
        watermarks = []

        try:
            # Use autocorrelation to find repeated patterns
            # Apply FFT-based autocorrelation
            fft_image = np.fft.fft2(image)
            autocorr = np.fft.ifft2(fft_image * np.conj(fft_image))
            autocorr = np.real(autocorr)

            # Find peaks in autocorrelation (excluding center)
            center_x, center_y = autocorr.shape[0] // 2, autocorr.shape[1] // 2
            autocorr[center_x-5:center_x+5, center_y-5:center_y+5] = 0

            threshold = np.max(autocorr) * 0.3
            peaks = np.where(autocorr > threshold)

            if len(peaks[0]) > 0:
                # Found repeated patterns
                strength = np.max(autocorr) / np.max(image)**2

                watermark = WatermarkMetadata(
                    watermark_type=WatermarkType.PATTERN_BASED,
                    domain=WatermarkDomain.SPATIAL,
                    strength=min(strength * 100, 1.0),
                    location=(0, 0, image.shape[1], image.shape[0]),
                    extraction_method="Autocorrelation analysis",
                    confidence=0.6,
                    embedded_data="repeated_pattern",
                    algorithm_fingerprint=f"autocorr_{len(peaks[0])}"
                )
                watermarks.append(watermark)

        except Exception as e:
            logger.error(f"Repeated pattern detection failed: {e}")

        return watermarks

class AudioWatermarkDetector:
    """Audio watermark detector"""

    def __init__(self):
        self.name = "Audio Watermark Detector"

    async def detect_audio_watermarks(self, audio_data: np.ndarray, sample_rate: int) -> List[WatermarkMetadata]:
        """Detect watermarks in audio data"""
        watermarks = []

        try:
            # Echo-based watermark detection
            echo_watermarks = await self._detect_echo_watermarks(audio_data, sample_rate)
            watermarks.extend(echo_watermarks)

            # Phase-based watermark detection
            phase_watermarks = await self._detect_phase_watermarks(audio_data, sample_rate)
            watermarks.extend(phase_watermarks)

            # Spread spectrum watermark detection
            ss_watermarks = await self._detect_spread_spectrum_audio(audio_data, sample_rate)
            watermarks.extend(ss_watermarks)

        except Exception as e:
            logger.error(f"Audio watermark detection failed: {e}")

        return watermarks

    async def _detect_echo_watermarks(self, audio_data: np.ndarray, sample_rate: int) -> List[WatermarkMetadata]:
        """Detect echo-based audio watermarks"""
        watermarks = []

        try:
            # Compute autocorrelation to find echo patterns
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]

            # Look for peaks that indicate echo delays
            # Typical echo delays: 0.5ms to 50ms
            min_delay = int(0.0005 * sample_rate)  # 0.5ms
            max_delay = int(0.05 * sample_rate)    # 50ms

            if max_delay < len(autocorr):
                echo_region = autocorr[min_delay:max_delay]
                threshold = np.max(echo_region) * 0.1

                peaks = []
                for i in range(1, len(echo_region) - 1):
                    if (echo_region[i] > echo_region[i-1] and
                        echo_region[i] > echo_region[i+1] and
                        echo_region[i] > threshold):
                        peaks.append(i + min_delay)

                if len(peaks) > 2:  # Multiple echo delays indicate watermark
                    delay_ms = np.mean(peaks) / sample_rate * 1000
                    strength = np.max(echo_region) / np.max(autocorr[:min_delay])

                    watermark = WatermarkMetadata(
                        watermark_type=WatermarkType.AUDIO_ECHO,
                        domain=WatermarkDomain.TEMPORAL,
                        strength=min(strength, 1.0),
                        location=(0, 0, len(audio_data), 1),
                        extraction_method=f"Echo detection ({delay_ms:.2f}ms delay)",
                        confidence=0.7,
                        embedded_data=f"echo_delay_{delay_ms:.2f}ms",
                        algorithm_fingerprint=f"echo_{len(peaks)}"
                    )
                    watermarks.append(watermark)

        except Exception as e:
            logger.error(f"Echo watermark detection failed: {e}")

        return watermarks

    async def _detect_phase_watermarks(self, audio_data: np.ndarray, sample_rate: int) -> List[WatermarkMetadata]:
        """Detect phase-based audio watermarks"""
        watermarks = []

        try:
            # Compute STFT to analyze phase information
            window_size = 2048
            hop_size = 512

            # Simple STFT implementation
            num_frames = (len(audio_data) - window_size) // hop_size + 1
            stft_data = np.zeros((window_size // 2 + 1, num_frames), dtype=complex)

            for frame in range(num_frames):
                start = frame * hop_size
                end = start + window_size
                windowed = audio_data[start:end] * np.hanning(window_size)
                fft_frame = np.fft.rfft(windowed)
                stft_data[:, frame] = fft_frame

            # Analyze phase consistency across frames
            phases = np.angle(stft_data)
            phase_diff = np.diff(phases, axis=1)

            # Look for systematic phase patterns
            phase_variance = np.var(phase_diff, axis=1)
            suspicious_bins = np.where(phase_variance < 0.1)[0]  # Low variance indicates systematic modification

            if len(suspicious_bins) > 10:
                strength = 1.0 - np.mean(phase_variance[suspicious_bins])

                watermark = WatermarkMetadata(
                    watermark_type=WatermarkType.AUDIO_PHASE,
                    domain=WatermarkDomain.FREQUENCY,
                    strength=min(strength, 1.0),
                    location=(0, 0, len(audio_data), 1),
                    extraction_method="Phase consistency analysis",
                    confidence=0.6,
                    embedded_data=f"phase_bins_{len(suspicious_bins)}",
                    algorithm_fingerprint=f"phase_{len(suspicious_bins)}"
                )
                watermarks.append(watermark)

        except Exception as e:
            logger.error(f"Phase watermark detection failed: {e}")

        return watermarks

    async def _detect_spread_spectrum_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[WatermarkMetadata]:
        """Detect spread spectrum audio watermarks"""
        watermarks = []

        try:
            # Look for high-frequency noise patterns that might indicate spread spectrum watermarks
            # Apply high-pass filter
            from scipy import signal as scipy_signal

            nyquist = sample_rate // 2
            high_cutoff = 8000  # 8kHz

            if high_cutoff < nyquist:
                b, a = scipy_signal.butter(4, high_cutoff / nyquist, btype='high')
                filtered_audio = scipy_signal.filtfilt(b, a, audio_data)

                # Analyze noise floor
                noise_power = np.mean(filtered_audio ** 2)

                # Compare with original signal power
                signal_power = np.mean(audio_data ** 2)
                noise_ratio = noise_power / signal_power if signal_power > 0 else 0

                if noise_ratio > 0.01:  # Elevated noise floor may indicate watermark
                    watermark = WatermarkMetadata(
                        watermark_type=WatermarkType.SPREAD_SPECTRUM,
                        domain=WatermarkDomain.FREQUENCY,
                        strength=min(noise_ratio * 10, 1.0),
                        location=(0, 0, len(audio_data), 1),
                        extraction_method="Spread spectrum noise analysis",
                        confidence=0.5,
                        embedded_data=f"noise_ratio_{noise_ratio:.4f}",
                        algorithm_fingerprint=f"ss_{int(noise_ratio * 10000)}"
                    )
                    watermarks.append(watermark)

        except Exception as e:
            logger.error(f"Spread spectrum audio detection failed: {e}")

        return watermarks

class WatermarkAnalysisWorker:
    """Main watermark analysis worker for ORACLE1"""

    def __init__(self):
        self.lsb_detector = LSBWatermarkDetector()
        self.frequency_detector = FrequencyWatermarkDetector()
        self.pattern_detector = PatternWatermarkDetector()
        self.audio_detector = AudioWatermarkDetector()

        # Data storage
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.influx_client = InfluxDBClient(
            url="http://localhost:8086",
            token="oracle-research-token",
            org="bev-research"
        )

        self.running = True

    async def analyze_content(self, content_url: str, content_type: str = None, headers: Dict = None) -> WatermarkAnalysisResult:
        """Analyze content for watermarks"""
        start_time = time.time()
        logger.info(f"Starting watermark analysis for: {content_url}")

        try:
            # Download content
            content_data = await self._download_content(content_url, headers)
            content_hash = hashlib.sha256(content_data).hexdigest()

            # Auto-detect content type if not provided
            if not content_type:
                content_type = await self._detect_content_type(content_data)

            watermarks_found = []
            analysis_methods = []

            # Route to appropriate analyzer based on content type
            if content_type.startswith('image/'):
                watermarks_found, analysis_methods = await self._analyze_image_content(content_data)
            elif content_type.startswith('audio/'):
                watermarks_found, analysis_methods = await self._analyze_audio_content(content_data)
            elif content_type.startswith('video/'):
                watermarks_found, analysis_methods = await self._analyze_video_content(content_data)
            else:
                logger.warning(f"Unsupported content type: {content_type}")

            processing_time = time.time() - start_time

            # Create analysis result
            result = WatermarkAnalysisResult(
                timestamp=datetime.now(),
                content_url=content_url,
                content_type=content_type,
                watermarks_found=watermarks_found,
                analysis_methods_used=analysis_methods,
                processing_time=processing_time,
                content_hash=content_hash,
                technical_details={
                    'content_size': len(content_data),
                    'watermarks_detected': len(watermarks_found),
                    'analysis_success': True
                },
                recommendations=await self._generate_recommendations(watermarks_found)
            )

            # Store results
            await self._store_analysis_result(result)

            logger.info(f"Watermark analysis completed: {len(watermarks_found)} watermarks found")
            return result

        except Exception as e:
            logger.error(f"Watermark analysis failed: {e}")
            raise

    async def _download_content(self, url: str, headers: Dict = None) -> bytes:
        """Download content from URL"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers or {}) as response:
                return await response.read()

    async def _detect_content_type(self, content_data: bytes) -> str:
        """Detect content type from data"""
        # Check magic numbers
        if content_data.startswith(b'\xFF\xD8\xFF'):
            return 'image/jpeg'
        elif content_data.startswith(b'\x89PNG'):
            return 'image/png'
        elif content_data.startswith(b'GIF8'):
            return 'image/gif'
        elif content_data.startswith(b'RIFF') and b'WAVE' in content_data[:12]:
            return 'audio/wav'
        elif content_data.startswith(b'ID3') or content_data.startswith(b'\xFF\xFB'):
            return 'audio/mp3'
        elif content_data.startswith(b'\x00\x00\x00\x20ftyp'):
            return 'video/mp4'
        else:
            return 'application/octet-stream'

    async def _analyze_image_content(self, content_data: bytes) -> Tuple[List[WatermarkMetadata], List[str]]:
        """Analyze image content for watermarks"""
        watermarks = []
        methods = []

        try:
            # Load image
            image = Image.open(io.BytesIO(content_data))
            image_array = np.array(image)

            # LSB analysis
            lsb_watermarks = await self.lsb_detector.detect_image_lsb(image_array)
            watermarks.extend(lsb_watermarks)
            if lsb_watermarks:
                methods.append("LSB Detection")

            # Frequency domain analysis
            freq_watermarks = await self.frequency_detector.detect_frequency_watermarks(image_array)
            watermarks.extend(freq_watermarks)
            if freq_watermarks:
                methods.append("Frequency Domain Analysis")

            # Pattern-based analysis
            pattern_watermarks = await self.pattern_detector.detect_pattern_watermarks(image_array)
            watermarks.extend(pattern_watermarks)
            if pattern_watermarks:
                methods.append("Pattern Detection")

        except Exception as e:
            logger.error(f"Image analysis failed: {e}")

        return watermarks, methods

    async def _analyze_audio_content(self, content_data: bytes) -> Tuple[List[WatermarkMetadata], List[str]]:
        """Analyze audio content for watermarks"""
        watermarks = []
        methods = []

        try:
            # Save to temporary file for audio processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(content_data)
                tmp_path = tmp_file.name

            try:
                # Load audio
                audio_data, sample_rate = sf.read(tmp_path)

                # Audio watermark detection
                audio_watermarks = await self.audio_detector.detect_audio_watermarks(audio_data, sample_rate)
                watermarks.extend(audio_watermarks)
                if audio_watermarks:
                    methods.append("Audio Watermark Detection")

            finally:
                # Clean up temporary file
                import os
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")

        return watermarks, methods

    async def _analyze_video_content(self, content_data: bytes) -> Tuple[List[WatermarkMetadata], List[str]]:
        """Analyze video content for watermarks"""
        watermarks = []
        methods = []

        try:
            # Video analysis would require extracting frames and audio
            # For now, return placeholder
            logger.info("Video watermark analysis not fully implemented")
            methods.append("Video Analysis (Limited)")

        except Exception as e:
            logger.error(f"Video analysis failed: {e}")

        return watermarks, methods

    async def _generate_recommendations(self, watermarks: List[WatermarkMetadata]) -> List[str]:
        """Generate recommendations based on found watermarks"""
        recommendations = []

        if not watermarks:
            recommendations.append("No watermarks detected - content appears clean")
            return recommendations

        # Analyze watermark types found
        types_found = set(w.watermark_type for w in watermarks)

        if WatermarkType.LSB in types_found:
            recommendations.append("LSB watermarks detected - consider steganography analysis tools")

        if WatermarkType.DCT in types_found or WatermarkType.DWT in types_found:
            recommendations.append("Frequency domain watermarks found - robust against compression")

        if any(w.strength > 0.7 for w in watermarks):
            recommendations.append("High-strength watermarks detected - removal may degrade quality")

        if len(watermarks) > 3:
            recommendations.append("Multiple watermarking systems detected - content heavily protected")

        return recommendations

    async def _store_analysis_result(self, result: WatermarkAnalysisResult):
        """Store analysis result in databases"""
        try:
            # Store in Redis
            key = f"watermark:analysis:{int(time.time())}"
            data = asdict(result)
            data['timestamp'] = result.timestamp.isoformat()

            self.redis_client.hset(key, mapping={k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                                                for k, v in data.items()})
            self.redis_client.expire(key, 86400 * 7)  # 7 days

            # Store in InfluxDB
            point = Point("watermark_analysis") \
                .tag("content_type", result.content_type) \
                .field("watermarks_found", len(result.watermarks_found)) \
                .field("processing_time", result.processing_time) \
                .field("content_size", result.technical_details.get('content_size', 0)) \
                .time(result.timestamp, WritePrecision.NS)

            write_api = self.influx_client.write_api()
            write_api.write(bucket="oracle-research", org="bev-research", record=point)

        except Exception as e:
            logger.error(f"Failed to store analysis result: {e}")

if __name__ == "__main__":
    async def main():
        worker = WatermarkAnalysisWorker()

        # Example analysis
        test_url = "https://example.com/image.jpg"
        result = await worker.analyze_content(test_url, 'image/jpeg')

        print(f"Analysis completed: {len(result.watermarks_found)} watermarks found")
        print(f"Processing time: {result.processing_time:.2f}s")

    asyncio.run(main())