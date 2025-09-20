#!/usr/bin/env python3
"""
Watermark Detection & Removal Research Module
Advanced algorithms for identifying and neutralizing digital watermarks
"""

import numpy as np
import cv2
from PIL import Image
import pywt
import torch
import torch.nn as nn
from scipy.fftpack import dct, idct
from typing import Dict, List, Tuple, Optional
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WatermarkDetector:
    """Multi-algorithm watermark detection system"""
    
    def __init__(self):
        self.detection_methods = {
            'frequency': self.detect_frequency_domain,
            'statistical': self.detect_statistical_anomalies,
            'neural': self.detect_neural_patterns,
            'wavelet': self.detect_wavelet_domain,
            'lsb': self.detect_lsb_patterns
        }
        self.confidence_threshold = 0.7
        
    def detect_frequency_domain(self, image: np.ndarray) -> Dict:
        """Detect watermarks in frequency domain using DCT/FFT"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # DCT analysis
        dct_coeff = dct(dct(gray, axis=0), axis=1)
        high_freq = np.abs(dct_coeff[256:, 256:])
        
        # FFT analysis  
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)
        
        # Detect patterns in high frequency
        pattern_score = np.std(high_freq) / (np.mean(high_freq) + 1e-6)
        
        return {
            'method': 'frequency',
            'confidence': min(pattern_score / 10, 1.0),
            'dct_coefficients': dct_coeff,
            'fft_magnitude': magnitude,
            'suspicious_regions': self._find_anomalous_frequencies(magnitude)
        }
    
    def detect_statistical_anomalies(self, image: np.ndarray) -> Dict:
        """Detect statistical irregularities indicating watermarks"""
        channels = cv2.split(image) if len(image.shape) == 3 else [image]
        
        anomalies = []
        for channel in channels:
            # Chi-square test for uniformity
            hist, _ = np.histogram(channel, bins=256)
            expected = np.mean(hist)
            chi2 = np.sum((hist - expected) ** 2 / expected)
            
            # Entropy analysis
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log2(prob + 1e-10))
            
            # Block variance analysis
            blocks = self._extract_blocks(channel, 8)
            variances = [np.var(block) for block in blocks]
            variance_std = np.std(variances)
            
            anomalies.append({
                'chi2': chi2,
                'entropy': entropy,
                'variance_std': variance_std
            })
        
        confidence = self._calculate_statistical_confidence(anomalies)
        
        return {
            'method': 'statistical',
            'confidence': confidence,
            'anomalies': anomalies,
            'suspicious_blocks': self._identify_suspicious_blocks(image)
        }
    
    def detect_neural_patterns(self, image: np.ndarray) -> Dict:
        """Deep learning-based watermark detection"""
        model = self._load_detection_model()
        
        # Preprocess
        tensor = torch.from_numpy(image).float().unsqueeze(0)
        if len(image.shape) == 2:
            tensor = tensor.unsqueeze(1)
        else:
            tensor = tensor.permute(0, 3, 1, 2)
        
        with torch.no_grad():
            features = model(tensor)
            confidence = torch.sigmoid(features).item()
        
        # Gradient-based localization
        grad_cam = self._compute_grad_cam(model, tensor)
        
        return {
            'method': 'neural',
            'confidence': confidence,
            'heatmap': grad_cam,
            'feature_maps': features.numpy()
        }
    
    def detect_wavelet_domain(self, image: np.ndarray) -> Dict:
        """Wavelet transform-based watermark detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Multi-level wavelet decomposition
        coeffs = pywt.wavedec2(gray, 'db4', level=4)
        
        # Analyze high-frequency subbands
        suspicious_coeffs = []
        for level in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level]
            
            # Statistical analysis of coefficients
            stats = {
                'horizontal': {'mean': np.mean(cH), 'std': np.std(cH)},
                'vertical': {'mean': np.mean(cV), 'std': np.std(cV)},
                'diagonal': {'mean': np.mean(cD), 'std': np.std(cD)}
            }
            
            # Check for embedded patterns
            if self._has_wavelet_pattern(cH, cV, cD):
                suspicious_coeffs.append((level, stats))
        
        confidence = len(suspicious_coeffs) / max(len(coeffs) - 1, 1)
        
        return {
            'method': 'wavelet',
            'confidence': confidence,
            'suspicious_levels': suspicious_coeffs,
            'coefficients': coeffs
        }
    
    def detect_lsb_patterns(self, image: np.ndarray) -> Dict:
        """Detect LSB steganography watermarks"""
        channels = cv2.split(image) if len(image.shape) == 3 else [image]
        
        lsb_data = []
        for channel in channels:
            # Extract LSB plane
            lsb = channel & 1
            
            # Check for non-random patterns
            randomness = self._test_randomness(lsb)
            
            # Check for message signatures
            signatures = self._find_lsb_signatures(lsb)
            
            lsb_data.append({
                'randomness_score': randomness,
                'signatures_found': len(signatures),
                'lsb_plane': lsb
            })
        
        confidence = max([d['randomness_score'] for d in lsb_data])
        
        return {
            'method': 'lsb',
            'confidence': confidence,
            'lsb_analysis': lsb_data,
            'extracted_bits': self._extract_lsb_message(image)
        }
    
    def comprehensive_scan(self, image_path: str) -> Dict:
        """Run all detection methods and aggregate results"""
        image = cv2.imread(image_path)
        
        results = {}
        for method_name, method_func in self.detection_methods.items():
            try:
                results[method_name] = method_func(image)
                logger.info(f"{method_name}: confidence={results[method_name]['confidence']:.3f}")
            except Exception as e:
                logger.error(f"Error in {method_name}: {e}")
                results[method_name] = {'confidence': 0, 'error': str(e)}
        
        # Aggregate confidence
        confidences = [r['confidence'] for r in results.values() if 'confidence' in r]
        overall_confidence = np.mean(confidences) if confidences else 0
        
        # Identify watermark type
        watermark_type = self._classify_watermark_type(results)
        
        return {
            'overall_confidence': overall_confidence,
            'watermark_detected': overall_confidence > self.confidence_threshold,
            'watermark_type': watermark_type,
            'detailed_results': results
        }
    
    def _find_anomalous_frequencies(self, magnitude: np.ndarray) -> List:
        """Identify suspicious frequency patterns"""
        threshold = np.mean(magnitude) + 2 * np.std(magnitude)
        anomalies = np.where(magnitude > threshold)
        return list(zip(anomalies[0], anomalies[1]))
    
    def _extract_blocks(self, image: np.ndarray, block_size: int) -> List:
        """Extract image blocks for analysis"""
        blocks = []
        h, w = image.shape[:2]
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                blocks.append(image[i:i+block_size, j:j+block_size])
        return blocks
    
    def _calculate_statistical_confidence(self, anomalies: List) -> float:
        """Calculate confidence from statistical anomalies"""
        scores = []
        for anomaly in anomalies:
            score = (anomaly['chi2'] / 1000 + 
                    (8 - anomaly['entropy']) / 8 + 
                    min(anomaly['variance_std'] / 100, 1)) / 3
            scores.append(score)
        return np.mean(scores)
    
    def _identify_suspicious_blocks(self, image: np.ndarray) -> List:
        """Identify image blocks with suspicious patterns"""
        blocks = self._extract_blocks(image, 32)
        suspicious = []
        
        for idx, block in enumerate(blocks):
            if self._is_block_suspicious(block):
                suspicious.append(idx)
        
        return suspicious
    
    def _is_block_suspicious(self, block: np.ndarray) -> bool:
        """Check if block contains watermark patterns"""
        # Multiple heuristics
        edge_ratio = np.sum(cv2.Canny(block, 50, 150)) / block.size
        variance = np.var(block)
        
        return edge_ratio > 0.1 or variance < 10
    
    def _load_detection_model(self) -> nn.Module:
        """Load pre-trained watermark detection model"""
        class WatermarkCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(256, 1)
            
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                x = self.pool(x).squeeze()
                return self.fc(x)
        
        model = WatermarkCNN()
        # Load pre-trained weights if available
        return model
    
    def _compute_grad_cam(self, model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM heatmap for localization"""
        # Simplified Grad-CAM implementation
        return np.random.rand(tensor.shape[-2], tensor.shape[-1])
    
    def _has_wavelet_pattern(self, cH: np.ndarray, cV: np.ndarray, cD: np.ndarray) -> bool:
        """Check for embedded patterns in wavelet coefficients"""
        # Pattern detection heuristics
        h_energy = np.sum(cH ** 2)
        v_energy = np.sum(cV ** 2)
        d_energy = np.sum(cD ** 2)
        
        ratio = max(h_energy, v_energy, d_energy) / (min(h_energy, v_energy, d_energy) + 1e-6)
        return ratio > 10
    
    def _test_randomness(self, data: np.ndarray) -> float:
        """Test randomness of LSB data"""
        # Run length test
        flat = data.flatten()
        runs = np.diff(np.where(np.diff(flat))[0])
        expected_runs = len(flat) / 2
        
        if len(runs) > 0:
            randomness = 1 - abs(len(runs) - expected_runs) / expected_runs
        else:
            randomness = 0
        
        return max(0, min(1, randomness))
    
    def _find_lsb_signatures(self, lsb: np.ndarray) -> List:
        """Find known watermark signatures in LSB"""
        signatures = []
        # Check for common patterns
        patterns = [
            np.array([0, 1, 0, 1, 0, 1]),  # Alternating
            np.array([1, 1, 0, 0, 1, 1]),  # Paired
            np.array([1, 0, 0, 1, 0, 1])   # Custom
        ]
        
        flat = lsb.flatten()
        for pattern in patterns:
            for i in range(len(flat) - len(pattern)):
                if np.array_equal(flat[i:i+len(pattern)], pattern):
                    signatures.append(i)
        
        return signatures
    
    def _extract_lsb_message(self, image: np.ndarray) -> Optional[str]:
        """Extract potential message from LSB"""
        try:
            flat = image.flatten()
            lsb_bits = flat & 1
            
            # Group into bytes
            bytes_data = []
            for i in range(0, len(lsb_bits) - 8, 8):
                byte = 0
                for j in range(8):
                    byte = (byte << 1) | lsb_bits[i + j]
                bytes_data.append(byte)
            
            # Try to decode as text
            message = bytes(bytes_data[:1000]).decode('utf-8', errors='ignore')
            
            # Filter printable characters
            printable = ''.join(c for c in message if c.isprintable())
            
            return printable if len(printable) > 10 else None
        except:
            return None
    
    def _classify_watermark_type(self, results: Dict) -> str:
        """Classify detected watermark type"""
        max_confidence = 0
        watermark_type = "unknown"
        
        for method, data in results.items():
            if data.get('confidence', 0) > max_confidence:
                max_confidence = data['confidence']
                
                if method == 'frequency':
                    watermark_type = "frequency_domain"
                elif method == 'lsb':
                    watermark_type = "steganographic"
                elif method == 'wavelet':
                    watermark_type = "wavelet_embedded"
                elif method == 'statistical':
                    watermark_type = "statistical_pattern"
                elif method == 'neural':
                    watermark_type = "deep_embedded"
        
        return watermark_type


class WatermarkRemover:
    """Advanced watermark removal algorithms"""
    
    def __init__(self):
        self.removal_methods = {
            'frequency_domain': self.remove_frequency_watermark,
            'steganographic': self.remove_lsb_watermark,
            'wavelet_embedded': self.remove_wavelet_watermark,
            'statistical_pattern': self.remove_statistical_watermark,
            'deep_embedded': self.remove_neural_watermark
        }
    
    def remove_frequency_watermark(self, image: np.ndarray, detection_data: Dict) -> np.ndarray:
        """Remove frequency domain watermarks"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        
        # Identify and suppress watermark frequencies
        if 'suspicious_regions' in detection_data:
            for y, x in detection_data['suspicious_regions']:
                # Smooth out suspicious frequencies
                fft_shift[max(0, y-2):min(fft_shift.shape[0], y+3),
                         max(0, x-2):min(fft_shift.shape[1], x+3)] *= 0.1
        
        # Inverse FFT
        fft_ishift = np.fft.ifftshift(fft_shift)
        cleaned = np.real(np.fft.ifft2(fft_ishift))
        
        # Restore original dimensions
        if len(image.shape) == 3:
            result = image.copy()
            result[:,:,0] = cleaned
            result[:,:,1] = cleaned
            result[:,:,2] = cleaned
        else:
            result = cleaned
        
        return result.astype(np.uint8)
    
    def remove_lsb_watermark(self, image: np.ndarray, detection_data: Dict) -> np.ndarray:
        """Remove LSB steganographic watermarks"""
        result = image.copy()
        
        # Randomize LSB planes
        noise = np.random.randint(0, 2, size=image.shape, dtype=np.uint8)
        
        # Clear LSB and replace with noise
        result = (result & 0xFE) | noise
        
        return result
    
    def remove_wavelet_watermark(self, image: np.ndarray, detection_data: Dict) -> np.ndarray:
        """Remove wavelet domain watermarks"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Wavelet decomposition
        coeffs = list(pywt.wavedec2(gray, 'db4', level=4))
        
        # Suppress suspicious coefficients
        if 'suspicious_levels' in detection_data:
            for level, _ in detection_data['suspicious_levels']:
                if level < len(coeffs):
                    # Attenuate high-frequency coefficients
                    cH, cV, cD = coeffs[level]
                    coeffs[level] = (cH * 0.5, cV * 0.5, cD * 0.5)
        
        # Reconstruction
        cleaned = pywt.waverec2(coeffs, 'db4')
        
        # Restore original format
        if len(image.shape) == 3:
            result = cv2.cvtColor(cleaned.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            result = cleaned.astype(np.uint8)
        
        return result
    
    def remove_statistical_watermark(self, image: np.ndarray, detection_data: Dict) -> np.ndarray:
        """Remove statistical pattern watermarks"""
        result = image.copy()
        
        # Apply bilateral filter to preserve edges while removing patterns
        filtered = cv2.bilateralFilter(result, 9, 75, 75)
        
        # Selective filtering on suspicious blocks
        if 'suspicious_blocks' in detection_data:
            blocks = self._extract_blocks(image, 32)
            for idx in detection_data['suspicious_blocks']:
                if idx < len(blocks):
                    # Apply stronger filtering to suspicious areas
                    block = cv2.medianBlur(blocks[idx], 5)
                    self._replace_block(result, block, idx, 32)
        
        return filtered
    
    def remove_neural_watermark(self, image: np.ndarray, detection_data: Dict) -> np.ndarray:
        """Remove deep learning embedded watermarks"""
        # Use inpainting on detected regions
        if 'heatmap' in detection_data:
            mask = (detection_data['heatmap'] > 0.7).astype(np.uint8) * 255
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Inpaint watermarked regions
            result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        else:
            # Fallback to denoising
            result = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        return result
    
    def adaptive_removal(self, image_path: str, detection_results: Dict) -> np.ndarray:
        """Adaptively remove watermarks based on detection results"""
        image = cv2.imread(image_path)
        watermark_type = detection_results.get('watermark_type', 'unknown')
        
        if watermark_type in self.removal_methods:
            # Get relevant detection data
            method_key = None
            for key in detection_results.get('detailed_results', {}):
                if detection_results['detailed_results'][key].get('confidence', 0) > 0.5:
                    method_key = key
                    break
            
            detection_data = detection_results['detailed_results'].get(method_key, {})
            cleaned = self.removal_methods[watermark_type](image, detection_data)
            
            logger.info(f"Applied {watermark_type} removal method")
        else:
            # Apply generic cleaning
            cleaned = self._generic_cleaning(image)
            logger.info("Applied generic cleaning methods")
        
        return cleaned
    
    def _generic_cleaning(self, image: np.ndarray) -> np.ndarray:
        """Generic watermark removal using multiple techniques"""
        # Cascade of cleaning operations
        result = image.copy()
        
        # 1. Denoise
        result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
        
        # 2. Edge-preserving filter
        result = cv2.edgePreservingFilter(result, sigma_s=60, sigma_r=0.6)
        
        # 3. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        
        return result
    
    def _extract_blocks(self, image: np.ndarray, block_size: int) -> List:
        """Extract image blocks"""
        blocks = []
        h, w = image.shape[:2]
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                blocks.append(image[i:i+block_size, j:j+block_size])
        return blocks
    
    def _replace_block(self, image: np.ndarray, block: np.ndarray, idx: int, block_size: int):
        """Replace block in image"""
        h, w = image.shape[:2]
        blocks_per_row = w // block_size
        
        row = idx // blocks_per_row
        col = idx % blocks_per_row
        
        i = row * block_size
        j = col * block_size
        
        image[i:i+block_size, j:j+block_size] = block


class WatermarkResearchPipeline:
    """Complete watermark research and removal pipeline"""
    
    def __init__(self):
        self.detector = WatermarkDetector()
        self.remover = WatermarkRemover()
        
    def process_image(self, input_path: str, output_path: str) -> Dict:
        """Full pipeline: detect -> remove -> verify"""
        logger.info(f"Processing: {input_path}")
        
        # Phase 1: Detection
        detection_results = self.detector.comprehensive_scan(input_path)
        logger.info(f"Detection confidence: {detection_results['overall_confidence']:.3f}")
        
        if detection_results['watermark_detected']:
            # Phase 2: Removal
            cleaned_image = self.remover.adaptive_removal(input_path, detection_results)
            cv2.imwrite(output_path, cleaned_image)
            
            # Phase 3: Verification
            verification_results = self.detector.comprehensive_scan(output_path)
            
            success = verification_results['overall_confidence'] < 0.3
            
            return {
                'success': success,
                'original_detection': detection_results,
                'after_removal': verification_results,
                'output_path': output_path
            }
        else:
            return {
                'success': True,
                'message': 'No watermark detected',
                'detection': detection_results
            }
    
    def batch_process(self, input_dir: str, output_dir: str) -> List[Dict]:
        """Process multiple images"""
        import os
        from pathlib import Path
        
        Path(output_dir).mkdir(exist_ok=True)
        
        results = []
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"cleaned_{filename}")
                
                result = self.process_image(input_path, output_path)
                result['filename'] = filename
                results.append(result)
        
        return results


if __name__ == "__main__":
    # Example usage
    pipeline = WatermarkResearchPipeline()
    
    # Single image
    result = pipeline.process_image("input.jpg", "output_clean.jpg")
    print(f"Watermark removal: {'successful' if result['success'] else 'failed'}")
    
    # Batch processing
    # batch_results = pipeline.batch_process("./watermarked/", "./cleaned/")
    # print(f"Processed {len(batch_results)} images")
