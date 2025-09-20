"""Watermark Analyzer for IntelOwl  
Detects and analyzes digital watermarks
Integrates with watermark_research.py
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
from celery import shared_task
from api_app.analyzers_manager.file_analyzers import FileAnalyzer
import logging
from PIL import Image
import cv2
from scipy import signal
import pywt

# Import our existing watermark research module
sys.path.append('/opt/bev_src/enhancement')
from watermark_research import WatermarkDetector

logger = logging.getLogger(__name__)


class WatermarkAnalyzer(FileAnalyzer):
    """Detect and analyze digital watermarks in files"""
    
    def set_params(self, params):
        """Set analyzer parameters"""
        self.detect_visible = params.get('detect_visible', True)
        self.detect_invisible = params.get('detect_invisible', True)
        self.detect_steganographic = params.get('detect_steganographic', True)
        self.extract_watermark = params.get('extract_watermark', False)
        self.identify_drm = params.get('identify_drm', True)
        
    def run(self):
        """Execute watermark analysis"""
        file_path = self.filepath
        results = {
            'watermarks_detected': [],
            'drm_protection': {},
            'watermark_types': [],
            'extraction_results': [],
            'copyright_info': {},
            'tampering_evidence': [],
            'recommendations': []
        }
        
        try:
            # Determine file type
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                results.update(self._analyze_image_watermark(file_path))
                
            elif file_ext == '.pdf':
                results.update(self._analyze_pdf_watermark(file_path))
                
            elif file_ext in ['.mp3', '.mp4', '.avi', '.mov', '.wav']:
                results.update(self._analyze_media_watermark(file_path))
                
            # Use existing watermark detector for comprehensive analysis
            detector = WatermarkDetector()
            detection_results = detector.detect_all_watermarks(file_path)
            results['watermarks_detected'].extend(detection_results)
            
            # Identify DRM protection
            if self.identify_drm:
                results['drm_protection'] = self._detect_drm(file_path)
                
            # Check for tampering
            results['tampering_evidence'] = self._check_tampering(results)
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
        except Exception as e:
            logger.error(f"Watermark analysis failed: {str(e)}")
            return {'error': str(e)}
            
        return results
        
    def _analyze_image_watermark(self, file_path: str) -> Dict:
        """Analyze image for watermarks"""
        results = {
            'watermarks_detected': [],
            'watermark_types': [],
            'extraction_results': []
        }
        
        try:
            img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect visible watermarks
            if self.detect_visible:
                visible_wm = self._detect_visible_watermark(img_rgb)
                if visible_wm:
                    results['watermarks_detected'].append(visible_wm)
                    results['watermark_types'].append('visible')
                    
            # Detect invisible watermarks
            if self.detect_invisible:
                # LSB watermark detection
                lsb_wm = self._detect_lsb_watermark(img_rgb)
                if lsb_wm:
                    results['watermarks_detected'].append(lsb_wm)
                    results['watermark_types'].append('lsb')
                    
                # DCT watermark detection
                dct_wm = self._detect_dct_watermark(gray)
                if dct_wm:
                    results['watermarks_detected'].append(dct_wm)
                    results['watermark_types'].append('dct')
                    
                # DWT watermark detection
                dwt_wm = self._detect_dwt_watermark(gray)
                if dwt_wm:
                    results['watermarks_detected'].append(dwt_wm)
                    results['watermark_types'].append('dwt')
                    
            # Extract watermark if requested
            if self.extract_watermark and results['watermarks_detected']:
                for wm in results['watermarks_detected']:
                    extraction = self._extract_watermark(img_rgb, wm['type'])
                    if extraction:
                        results['extraction_results'].append(extraction)
                        
        except Exception as e:
            logger.error(f"Image watermark analysis failed: {str(e)}")
            
        return results
        
    def _detect_visible_watermark(self, image: np.ndarray) -> Optional[Dict]:
        """Detect visible watermarks using computer vision"""
        try:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Edge detection to find watermark boundaries
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for text-like regions (common for watermarks)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100 and area < image.shape[0] * image.shape[1] * 0.1:
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = gray[y:y+h, x:x+w]
                    
                    # Check for text patterns
                    if self._is_text_region(roi):
                        return {
                            'type': 'visible',
                            'location': {'x': x, 'y': y, 'width': w, 'height': h},
                            'confidence': 0.7,
                            'description': 'Visible watermark detected'
                        }
                        
            # Check corners and edges (common watermark locations)
            corner_regions = [
                gray[:100, :100],  # Top-left
                gray[:100, -100:],  # Top-right
                gray[-100:, :100],  # Bottom-left
                gray[-100:, -100:]  # Bottom-right
            ]
            
            for i, region in enumerate(corner_regions):
                if self._has_watermark_pattern(region):
                    corners = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
                    return {
                        'type': 'visible',
                        'location': corners[i],
                        'confidence': 0.6,
                        'description': f'Watermark pattern in {corners[i]} corner'
                    }
                    
        except Exception as e:
            logger.error(f"Visible watermark detection failed: {str(e)}")
            
        return None
        
    def _detect_lsb_watermark(self, image: np.ndarray) -> Optional[Dict]:
        """Detect LSB (Least Significant Bit) watermarks"""
        try:
            # Extract LSB plane
            lsb_plane = image & 1
            
            # Analyze LSB patterns
            unique_patterns = len(np.unique(lsb_plane.reshape(-1, 3), axis=0))
            
            # Statistical analysis of LSB
            lsb_mean = np.mean(lsb_plane)
            lsb_std = np.std(lsb_plane)
            
            # Check for non-random patterns (indication of watermark)
            if lsb_std < 0.4 or unique_patterns < 5:
                return {
                    'type': 'lsb',
                    'confidence': 0.8,
                    'description': 'LSB watermark detected',
                    'statistics': {
                        'mean': float(lsb_mean),
                        'std': float(lsb_std),
                        'unique_patterns': unique_patterns
                    }
                }
                
        except Exception as e:
            logger.error(f"LSB watermark detection failed: {str(e)}")
            
        return None
        
    def _detect_dct_watermark(self, image: np.ndarray) -> Optional[Dict]:
        """Detect DCT-based watermarks"""
        try:
            # Apply DCT
            dct = cv2.dct(np.float32(image))
            
            # Analyze mid-frequency coefficients (common for watermarks)
            h, w = dct.shape
            mid_band = dct[h//4:3*h//4, w//4:3*w//4]
            
            # Statistical analysis
            energy = np.sum(np.abs(mid_band))
            pattern_strength = np.std(mid_band)
            
            # Check for embedded patterns
            if pattern_strength > np.std(dct) * 1.5:
                return {
                    'type': 'dct',
                    'confidence': 0.7,
                    'description': 'DCT-based watermark detected',
                    'statistics': {
                        'mid_band_energy': float(energy),
                        'pattern_strength': float(pattern_strength)
                    }
                }
                
        except Exception as e:
            logger.error(f"DCT watermark detection failed: {str(e)}")
            
        return None
        
    def _detect_dwt_watermark(self, image: np.ndarray) -> Optional[Dict]:
        """Detect DWT (Discrete Wavelet Transform) watermarks"""
        try:
            # Apply DWT
            coeffs = pywt.dwt2(image, 'db4')
            cA, (cH, cV, cD) = coeffs
            
            # Analyze high-frequency components
            hf_energy = np.sum(np.abs(cH)) + np.sum(np.abs(cV)) + np.sum(np.abs(cD))
            lf_energy = np.sum(np.abs(cA))
            
            ratio = hf_energy / (lf_energy + 1e-10)
            
            # Unusual ratio might indicate watermark
            if ratio > 0.5:
                return {
                    'type': 'dwt',
                    'confidence': 0.65,
                    'description': 'DWT-based watermark detected',
                    'statistics': {
                        'hf_energy': float(hf_energy),
                        'lf_energy': float(lf_energy),
                        'ratio': float(ratio)
                    }
                }
                
        except Exception as e:
            logger.error(f"DWT watermark detection failed: {str(e)}")
            
        return None
        
    def _analyze_pdf_watermark(self, file_path: str) -> Dict:
        """Analyze PDF for watermarks"""
        results = {
            'watermarks_detected': [],
            'watermark_types': []
        }
        
        try:
            from PyPDF2 import PdfReader
            
            with open(file_path, 'rb') as f:
                pdf = PdfReader(f)
                
                for page_num, page in enumerate(pdf.pages):
                    # Check for watermark objects
                    if '/Watermark' in page:
                        results['watermarks_detected'].append({
                            'type': 'pdf_watermark',
                            'page': page_num,
                            'description': 'PDF watermark object found'
                        })
                        results['watermark_types'].append('pdf_object')
                        
                    # Check for transparent text (common watermark technique)
                    if '/Contents' in page:
                        contents = page['/Contents']
                        # Look for transparency settings
                        if b'gs' in str(contents) or b'CA' in str(contents):
                            results['watermarks_detected'].append({
                                'type': 'transparent_text',
                                'page': page_num,
                                'description': 'Transparent text watermark detected'
                            })
                            
        except Exception as e:
            logger.error(f"PDF watermark analysis failed: {str(e)}")
            
        return results
        
    def _analyze_media_watermark(self, file_path: str) -> Dict:
        """Analyze media files for watermarks"""
        results = {
            'watermarks_detected': [],
            'watermark_types': []
        }
        
        try:
            # Audio watermark detection (simplified)
            if file_path.endswith(('.mp3', '.wav')):
                # Check for echo-based watermarks
                import scipy.io.wavfile as wav
                
                rate, data = wav.read(file_path)
                
                # Analyze frequency spectrum
                freqs, times, spectrogram = signal.spectrogram(data, rate)
                
                # Look for patterns in high frequencies (often used for watermarks)
                high_freq_energy = np.sum(spectrogram[len(freqs)//2:, :])
                total_energy = np.sum(spectrogram)
                
                if high_freq_energy / total_energy > 0.1:
                    results['watermarks_detected'].append({
                        'type': 'audio_spectrum',
                        'confidence': 0.6,
                        'description': 'Audio spectrum watermark detected'
                    })
                    results['watermark_types'].append('audio')
                    
        except Exception as e:
            logger.error(f"Media watermark analysis failed: {str(e)}")
            
        return results
        
    def _extract_watermark(self, image: np.ndarray, watermark_type: str) -> Optional[Dict]:
        """Extract watermark content"""
        extraction = {
            'type': watermark_type,
            'extracted': False,
            'content': None
        }
        
        try:
            if watermark_type == 'lsb':
                # Extract LSB watermark
                lsb_data = image & 1
                # Convert to bytes
                watermark_bits = lsb_data.flatten()
                watermark_bytes = np.packbits(watermark_bits)
                
                # Try to decode as text
                try:
                    text = watermark_bytes.tobytes().decode('utf-8', errors='ignore')
                    if text.isprintable():
                        extraction['extracted'] = True
                        extraction['content'] = text[:100]  # First 100 chars
                except:
                    pass
                    
            elif watermark_type == 'visible':
                # For visible watermarks, we'd use OCR
                extraction['extracted'] = False
                extraction['content'] = 'OCR extraction not implemented'
                
        except Exception as e:
            logger.error(f"Watermark extraction failed: {str(e)}")
            
        return extraction
        
    def _detect_drm(self, file_path: str) -> Dict:
        """Detect DRM protection"""
        drm_info = {
            'protected': False,
            'drm_type': None,
            'restrictions': []
        }
        
        try:
            # Check file headers for DRM signatures
            with open(file_path, 'rb') as f:
                header = f.read(1024)
                
                # Check for common DRM signatures
                if b'DRM' in header or b'WMDRM' in header:
                    drm_info['protected'] = True
                    drm_info['drm_type'] = 'Windows Media DRM'
                    drm_info['restrictions'].append('Copy protected')
                    
                elif b'FairPlay' in header:
                    drm_info['protected'] = True
                    drm_info['drm_type'] = 'Apple FairPlay'
                    drm_info['restrictions'].append('Apple device only')
                    
                elif b'Marlin' in header:
                    drm_info['protected'] = True
                    drm_info['drm_type'] = 'Marlin DRM'
                    drm_info['restrictions'].append('Streaming protected')
                    
        except Exception as e:
            logger.error(f"DRM detection failed: {str(e)}")
            
        return drm_info
        
    def _check_tampering(self, results: Dict) -> List[Dict]:
        """Check for evidence of tampering with watermarks"""
        tampering = []
        
        # Multiple watermarks might indicate tampering
        if len(results.get('watermarks_detected', [])) > 2:
            tampering.append({
                'type': 'multiple_watermarks',
                'description': 'Multiple watermarks detected - possible tampering',
                'severity': 'MEDIUM'
            })
            
        # Conflicting watermark types
        types = results.get('watermark_types', [])
        if 'visible' in types and 'invisible' in types:
            tampering.append({
                'type': 'conflicting_watermarks',
                'description': 'Both visible and invisible watermarks present',
                'severity': 'HIGH'
            })
            
        return tampering
        
    def _is_text_region(self, region: np.ndarray) -> bool:
        """Check if region contains text patterns"""
        # Simplified text detection using variance
        variance = np.var(region)
        return 500 < variance < 5000
        
    def _has_watermark_pattern(self, region: np.ndarray) -> bool:
        """Check if region has watermark-like patterns"""
        # Check for repetitive patterns
        mean_val = np.mean(region)
        return 100 < mean_val < 200  # Semi-transparent range
        
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on findings"""
        recommendations = []
        
        if results.get('watermarks_detected'):
            recommendations.append('Watermarks detected - consider legal implications before use')
            
            if 'visible' in results.get('watermark_types', []):
                recommendations.append('Visible watermark present - removal may violate copyright')
                
            if 'invisible' in results.get('watermark_types', []):
                recommendations.append('Invisible watermark present - content is tracked')
                
        if results.get('drm_protection', {}).get('protected'):
            recommendations.append('DRM protection detected - circumvention may be illegal')
            
        if results.get('tampering_evidence'):
            recommendations.append('Evidence of tampering detected - verify content authenticity')
            
        if not results.get('watermarks_detected'):
            recommendations.append('No watermarks detected - content appears unmarked')
            
        return recommendations
        
    @classmethod
    def _monkeypatch(cls):
        """Register analyzer with IntelOwl"""
        patches = [
            {
                'model': 'analyzers_manager.AnalyzerConfig',
                'name': 'WatermarkAnalyzer',
                'description': 'Detect and analyze digital watermarks',
                'python_module': 'custom_analyzers.watermark_analyzer.WatermarkAnalyzer',
                'disabled': False,
                'type': 'file',
                'docker_based': False,
                'maximum_tlp': 'RED',
                'observable_supported': [],
                'supported_filetypes': [
                    'image/jpeg', 'image/png', 'image/bmp',
                    'application/pdf', 'audio/mpeg', 'video/mp4'
                ],
                'run_hash': False,
                'run_hash_type': '',
                'not_supported_filetypes': [],
                'parameters': {
                    'detect_visible': {
                        'type': 'bool',
                        'description': 'Detect visible watermarks',
                        'default': True
                    },
                    'detect_invisible': {
                        'type': 'bool',
                        'description': 'Detect invisible watermarks',
                        'default': True
                    },
                    'extract_watermark': {
                        'type': 'bool',
                        'description': 'Attempt to extract watermark content',
                        'default': False
                    },
                    'identify_drm': {
                        'type': 'bool',
                        'description': 'Identify DRM protection',
                        'default': True
                    }
                }
            }
        ]
        return patches
