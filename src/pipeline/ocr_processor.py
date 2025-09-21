#!/usr/bin/env python3
"""
Advanced OCR Processing Pipeline
Multi-engine text extraction with enhancement and post-processing
"""

import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Advanced image preprocessing for OCR optimization"""
    
    def __init__(self):
        self.enhancement_pipeline = [
            self.remove_noise,
            self.deskew,
            self.remove_borders,
            self.binarize,
            self.enhance_text
        ]
    
    def preprocess(self, image: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """Apply preprocessing pipeline"""
        result = image.copy()
        
        for enhancement in self.enhancement_pipeline:
            try:
                result = enhancement(result)
            except Exception as e:
                logger.warning(f"Enhancement failed: {e}")
                continue
        
        if aggressive:
            result = self.aggressive_enhancement(result)
        
        return result
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using morphological operations"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Bilateral filter for edge-preserving smoothing
        filtered = cv2.bilateralFilter(morph, 9, 75, 75)
        
        return filtered
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew"""
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            # Calculate average angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                if abs(angle) < 45:  # Filter out vertical lines
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                
                # Rotate image
                if abs(median_angle) > 0.5:  # Only correct if skew is significant
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    rotated = cv2.warpAffine(image, M, (w, h), 
                                           flags=cv2.INTER_CUBIC,
                                           borderMode=cv2.BORDER_REPLICATE)
                    return rotated
        
        return image
    
    def remove_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove black borders and margins"""
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (main content)
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # Crop to content
            cropped = image[y:y+h, x:x+w]
            return cropped
        
        return image
    
    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Advanced binarization for text extraction"""
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding for varying lighting
        binary = cv2.adaptiveThreshold(gray, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Alternative: OTSU thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine both methods
        combined = cv2.bitwise_and(binary, otsu)
        
        return combined
    
    def enhance_text(self, image: np.ndarray) -> np.ndarray:
        """Enhance text clarity"""
        # Sharpen kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        if len(sharpened.shape) == 2:
            enhanced = clahe.apply(sharpened)
        else:
            # Apply to luminance channel
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def aggressive_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Aggressive enhancement for difficult images"""
        result = image.copy()
        
        # Super-resolution using interpolation
        result = cv2.resize(result, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Morphological tophat for background removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(result, cv2.MORPH_TOPHAT, kernel)
        
        # Add tophat to original
        result = cv2.add(result, tophat)
        
        # Aggressive denoising
        result = cv2.fastNlMeansDenoising(result, None, 30, 7, 21)
        
        return result


class MultiEngineOCR:
    """Multiple OCR engines for maximum accuracy"""
    
    def __init__(self):
        self.tesseract_available = self._check_tesseract()
        self.easyocr_reader = None
        self.preprocessor = ImagePreprocessor()
        
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available"""
        try:
            pytesseract.get_tesseract_version()
            return True
        except:
            logger.warning("Tesseract not found")
            return False
    
    def _init_easyocr(self):
        """Initialize EasyOCR reader"""
        if self.easyocr_reader is None:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=True)
            except:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
    
    def ocr_tesseract(self, image: np.ndarray, lang: str = 'eng') -> Dict:
        """OCR using Tesseract"""
        if not self.tesseract_available:
            return {'text': '', 'confidence': 0}
        
        try:
            # Get text with confidence scores
            data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
            
            # Extract text and confidence
            text_parts = []
            confidences = []
            
            for i, conf in enumerate(data['conf']):
                if conf > 0:  # Filter out non-text
                    text = data['text'][i].strip()
                    if text:
                        text_parts.append(text)
                        confidences.append(conf)
            
            full_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Also get layout information
            layout = self._extract_layout_tesseract(data)
            
            return {
                'text': full_text,
                'confidence': avg_confidence / 100,
                'layout': layout,
                'engine': 'tesseract'
            }
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {'text': '', 'confidence': 0, 'error': str(e)}
    
    def ocr_easyocr(self, image: np.ndarray) -> Dict:
        """OCR using EasyOCR"""
        self._init_easyocr()
        
        if self.easyocr_reader is None:
            return {'text': '', 'confidence': 0}
        
        try:
            # Run EasyOCR
            results = self.easyocr_reader.readtext(image)
            
            # Extract text and confidence
            text_parts = []
            confidences = []
            layout = []
            
            for bbox, text, conf in results:
                text_parts.append(text)
                confidences.append(conf)
                layout.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': conf
                })
            
            full_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                'text': full_text,
                'confidence': avg_confidence,
                'layout': layout,
                'engine': 'easyocr'
            }
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return {'text': '', 'confidence': 0, 'error': str(e)}
    
    def ocr_hybrid(self, image: np.ndarray, preprocess: bool = True) -> Dict:
        """Hybrid OCR using multiple engines"""
        if preprocess:
            processed = self.preprocessor.preprocess(image)
        else:
            processed = image
        
        results = []
        
        # Run all available engines
        if self.tesseract_available:
            results.append(self.ocr_tesseract(processed))
        
        results.append(self.ocr_easyocr(processed))
        
        # Merge results
        if not results:
            return {'text': '', 'confidence': 0, 'method': 'none'}
        
        # Choose best result or combine
        best_result = max(results, key=lambda x: x.get('confidence', 0))
        
        # Try to combine if multiple results
        if len(results) > 1:
            combined_text = self._merge_ocr_results(results)
            combined_confidence = np.mean([r.get('confidence', 0) for r in results])
            
            if combined_confidence > best_result.get('confidence', 0):
                return {
                    'text': combined_text,
                    'confidence': combined_confidence,
                    'method': 'hybrid',
                    'engines': [r.get('engine', 'unknown') for r in results]
                }
        
        return best_result
    
    def _extract_layout_tesseract(self, data: Dict) -> List[Dict]:
        """Extract layout information from Tesseract data"""
        layout = []
        
        for i in range(len(data['text'])):
            if data['conf'][i] > 0 and data['text'][i].strip():
                layout.append({
                    'text': data['text'][i],
                    'bbox': (data['left'][i], data['top'][i], 
                           data['width'][i], data['height'][i]),
                    'confidence': data['conf'][i] / 100,
                    'level': data['level'][i]
                })
        
        return layout
    
    def _merge_ocr_results(self, results: List[Dict]) -> str:
        """Intelligently merge OCR results from multiple engines"""
        # Simple voting mechanism - can be made more sophisticated
        all_words = []
        
        for result in results:
            text = result.get('text', '')
            words = text.split()
            all_words.append(words)
        
        # Find consensus
        merged = []
        max_len = max(len(words) for words in all_words)
        
        for i in range(max_len):
            word_candidates = []
            for words in all_words:
                if i < len(words):
                    word_candidates.append(words[i])
            
            if word_candidates:
                # Choose most common word
                from collections import Counter
                most_common = Counter(word_candidates).most_common(1)[0][0]
                merged.append(most_common)
        
        return ' '.join(merged)


class DocumentOCR:
    """OCR for complete documents"""
    
    def __init__(self):
        self.ocr_engine = MultiEngineOCR()
        self.preprocessor = ImagePreprocessor()
        
    def process_pdf(self, pdf_path: str, output_format: str = 'text') -> Dict:
        """Extract text from PDF"""
        import fitz  # PyMuPDF
        
        results = {
            'pages': [],
            'total_text': '',
            'metadata': {}
        }
        
        # Open PDF
        pdf = fitz.open(pdf_path)
        
        # Extract metadata
        results['metadata'] = pdf.metadata
        
        all_text = []
        
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            
            # Try direct text extraction first
            text = page.get_text()
            
            if text.strip():
                # Has embedded text
                page_result = {
                    'page': page_num + 1,
                    'text': text,
                    'method': 'embedded',
                    'confidence': 1.0
                }
            else:
                # Need OCR
                # Render page to image
                mat = fitz.Matrix(2, 2)  # 2x scaling
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to numpy array
                import io
                from PIL import Image
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img)
                
                # Run OCR
                ocr_result = self.ocr_engine.ocr_hybrid(img_array)
                
                page_result = {
                    'page': page_num + 1,
                    'text': ocr_result['text'],
                    'method': 'ocr',
                    'confidence': ocr_result.get('confidence', 0)
                }
            
            results['pages'].append(page_result)
            all_text.append(page_result['text'])
        
        pdf.close()
        
        results['total_text'] = '\n\n'.join(all_text)
        
        # Format output
        if output_format == 'text':
            return results['total_text']
        elif output_format == 'json':
            return results
        else:
            return results
    
    def process_image(self, image_path: str, languages: List[str] = None) -> Dict:
        """Process single image"""
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            return {'error': 'Failed to load image'}
        
        # Detect text regions
        regions = self._detect_text_regions(image)
        
        results = {
            'file': image_path,
            'regions': [],
            'full_text': ''
        }
        
        all_text = []
        
        for i, region in enumerate(regions):
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
            
            # OCR on region
            ocr_result = self.ocr_engine.ocr_hybrid(roi)
            
            results['regions'].append({
                'index': i,
                'bbox': region,
                'text': ocr_result['text'],
                'confidence': ocr_result.get('confidence', 0)
            })
            
            all_text.append(ocr_result['text'])
        
        results['full_text'] = ' '.join(all_text)
        
        return results
    
    def batch_process(self, input_dir: str, output_dir: str, 
                     formats: List[str] = None) -> List[Dict]:
        """Batch process documents"""
        if formats is None:
            formats = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
        results = []
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all documents
        files = []
        for ext in formats:
            files.extend([f for f in os.listdir(input_dir) 
                         if f.lower().endswith(ext)])
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for file in files:
                input_path = os.path.join(input_dir, file)
                output_path = os.path.join(output_dir, file + '.txt')
                
                future = executor.submit(self._process_single, 
                                        input_path, output_path)
                futures.append((file, future))
            
            for file, future in futures:
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                    logger.info(f"Processed {file}")
                except Exception as e:
                    logger.error(f"Failed to process {file}: {e}")
                    results.append({'file': file, 'error': str(e)})
        
        return results
    
    def _process_single(self, input_path: str, output_path: str) -> Dict:
        """Process single document"""
        if input_path.lower().endswith('.pdf'):
            text = self.process_pdf(input_path, 'text')
        else:
            result = self.process_image(input_path)
            text = result.get('full_text', '')
        
        # Save text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return {
            'file': input_path,
            'output': output_path,
            'text_length': len(text),
            'success': True
        }
    
    def _detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use MSER for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        # Get bounding boxes
        bboxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            
            # Filter small regions
            if w > 20 and h > 10:
                bboxes.append((x, y, w, h))
        
        # Merge overlapping boxes
        merged = self._merge_boxes(bboxes)
        
        # If no regions found, use whole image
        if not merged:
            h, w = image.shape[:2]
            return [(0, 0, w, h)]
        
        return merged
    
    def _merge_boxes(self, boxes: List[Tuple]) -> List[Tuple]:
        """Merge overlapping bounding boxes"""
        if not boxes:
            return []
        
        # Sort by x coordinate
        boxes = sorted(boxes, key=lambda b: b[0])
        
        merged = []
        current = boxes[0]
        
        for box in boxes[1:]:
            # Check if boxes overlap
            if self._boxes_overlap(current, box):
                # Merge
                x1 = min(current[0], box[0])
                y1 = min(current[1], box[1])
                x2 = max(current[0] + current[2], box[0] + box[2])
                y2 = max(current[1] + current[3], box[1] + box[3])
                current = (x1, y1, x2 - x1, y2 - y1)
            else:
                merged.append(current)
                current = box
        
        merged.append(current)
        
        return merged
    
    def _boxes_overlap(self, box1: Tuple, box2: Tuple) -> bool:
        """Check if two boxes overlap"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        return not (x1 + w1 < x2 or x2 + w2 < x1 or 
                   y1 + h1 < y2 or y2 + h2 < y1)


class OCRPostProcessor:
    """Post-processing for OCR results"""
    
    def __init__(self):
        self.spell_checker = None
        self.load_dictionaries()
    
    def load_dictionaries(self):
        """Load spell checking dictionaries"""
        try:
            from spellchecker import SpellChecker
            self.spell_checker = SpellChecker()
        except:
            logger.warning("SpellChecker not available")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize OCR text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR errors
        replacements = {
            'rn': 'm',
            'l1': 'll',
            'O0': '00',
            '|': 'I',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def spell_correct(self, text: str) -> str:
        """Spell correction for OCR text"""
        if not self.spell_checker:
            return text
        
        words = text.split()
        corrected = []
        
        for word in words:
            # Check if word is misspelled
            if word.lower() not in self.spell_checker:
                # Get correction
                correction = self.spell_checker.correction(word)
                if correction:
                    corrected.append(correction)
                else:
                    corrected.append(word)
            else:
                corrected.append(word)
        
        return ' '.join(corrected)
    
    def format_output(self, text: str, format_type: str = 'plain') -> str:
        """Format OCR output"""
        if format_type == 'plain':
            return text
        elif format_type == 'markdown':
            # Add markdown formatting
            lines = text.split('\n')
            formatted = []
            
            for line in lines:
                if line.isupper() and len(line) < 100:
                    # Likely a header
                    formatted.append(f"## {line}")
                else:
                    formatted.append(line)
            
            return '\n'.join(formatted)
        elif format_type == 'html':
            # Convert to HTML
            html = f"<html><body><p>{text.replace(chr(10), '</p><p>')}</p></body></html>"
            return html
        
        return text


if __name__ == "__main__":
    # Initialize components
    ocr = MultiEngineOCR()
    doc_ocr = DocumentOCR()
    post_processor = OCRPostProcessor()
    
    # Example usage
    # result = doc_ocr.process_pdf("document.pdf")
    # cleaned = post_processor.clean_text(result)
    # corrected = post_processor.spell_correct(cleaned)
    
    print("OCR Processing Pipeline initialized - Extracting text from everything!")
