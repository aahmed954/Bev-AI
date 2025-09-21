#!/usr/bin/env python3
"""
Advanced OCR Pipeline for Bev Research Framework
Multi-engine OCR with preprocessing, enhancement, and extraction
"""

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import pytesseract
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import pdfplumber
import fitz  # PyMuPDF
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

class OCREngine(Enum):
    """Available OCR engines"""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    TROCR = "trocr"  # Transformer-based OCR
    PADDLE = "paddle"  # PaddleOCR
    ALL = "all"  # Use all engines and vote

@dataclass
class OCRResult:
    """OCR extraction results"""
    text: str
    confidence: float
    engine: str
    language: str
    preprocessing_applied: List[str]
    bbox: Optional[List[Tuple[int, int, int, int]]] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ImagePreprocessor:
    """Advanced image preprocessing for OCR optimization"""
    
    @staticmethod
    def auto_enhance(image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Automatically enhance image for better OCR"""
        applied = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            applied.append("grayscale")
        
        # Denoise
        image = cv2.fastNlMeansDenoising(image, h=30)
        applied.append("denoise")
        
        # Adaptive thresholding for text
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        applied.append("adaptive_threshold")
        
        return image, applied
    
    @staticmethod
    def remove_skew(image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deskew image using Hough transform"""
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                if -45 <= angle <= 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated, median_angle
        
        return image, 0.0
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image.shape) == 2:
            return clahe.apply(image)
        else:
            # Apply to luminance channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def remove_noise(image: np.ndarray) -> np.ndarray:
        """Advanced noise removal"""
        # Morphological operations
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Median filter for salt-and-pepper noise
        image = cv2.medianBlur(image, 3)
        
        return image
    
    @staticmethod
    def upscale_image(image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """Upscale image for better OCR on small text"""
        height, width = image.shape[:2]
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))
        return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)

class TesseractEngine:
    """Tesseract OCR engine wrapper"""
    
    def __init__(self):
        self.languages = self._get_available_languages()
        self.custom_config = '--oem 3 --psm 11'  # LSTM engine, sparse text
    
    def _get_available_languages(self) -> List[str]:
        """Get available Tesseract languages"""
        try:
            langs = pytesseract.get_languages()
            return [l for l in langs if not l.startswith('osd')]
        except:
            return ['eng']
    
    def extract(self, image: np.ndarray, language: str = 'eng') -> OCRResult:
        """Extract text using Tesseract"""
        try:
            # Get text with confidence scores
            data = pytesseract.image_to_data(
                image, 
                lang=language,
                config=self.custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Filter and combine text
            text_parts = []
            total_conf = 0
            valid_boxes = 0
            bboxes = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:
                    text = data['text'][i].strip()
                    if text:
                        text_parts.append(text)
                        total_conf += int(data['conf'][i])
                        valid_boxes += 1
                        
                        bbox = (
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        )
                        bboxes.append(bbox)
            
            avg_conf = total_conf / valid_boxes if valid_boxes > 0 else 0
            
            return OCRResult(
                text=' '.join(text_parts),
                confidence=avg_conf / 100,
                engine="tesseract",
                language=language,
                preprocessing_applied=[],
                bbox=bboxes
            )
            
        except Exception as e:
            print(f"Tesseract error: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine="tesseract",
                language=language,
                preprocessing_applied=[]
            )

class EasyOCREngine:
    """EasyOCR engine wrapper"""
    
    def __init__(self):
        self.reader = None
        self.supported_langs = ['en', 'ch_sim', 'ch_tra', 'ja', 'ko', 'ru', 'ar']
    
    def _initialize_reader(self, languages: List[str]):
        """Lazy load EasyOCR reader"""
        if self.reader is None or set(languages) != set(self.reader.lang_list):
            self.reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())
    
    def extract(self, image: np.ndarray, language: str = 'en') -> OCRResult:
        """Extract text using EasyOCR"""
        try:
            # Map language codes
            lang_map = {'eng': 'en', 'chi_sim': 'ch_sim', 'jpn': 'ja'}
            lang = lang_map.get(language, language)
            
            if lang not in self.supported_langs:
                lang = 'en'
            
            self._initialize_reader([lang])
            
            # Perform OCR
            results = self.reader.readtext(image, detail=1)
            
            if not results:
                return OCRResult(
                    text="",
                    confidence=0.0,
                    engine="easyocr",
                    language=language,
                    preprocessing_applied=[]
                )
            
            # Extract text and bboxes
            text_parts = []
            confidences = []
            bboxes = []
            
            for (bbox, text, confidence) in results:
                text_parts.append(text)
                confidences.append(confidence)
                
                # Convert bbox format
                points = np.array(bbox).astype(int)
                x_min = min(points[:, 0])
                y_min = min(points[:, 1])
                x_max = max(points[:, 0])
                y_max = max(points[:, 1])
                bboxes.append((x_min, y_min, x_max, y_max))
            
            return OCRResult(
                text=' '.join(text_parts),
                confidence=np.mean(confidences) if confidences else 0.0,
                engine="easyocr",
                language=language,
                preprocessing_applied=[],
                bbox=bboxes
            )
            
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine="easyocr",
                language=language,
                preprocessing_applied=[]
            )

class TrOCREngine:
    """Transformer-based OCR using Microsoft's TrOCR"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _initialize_model(self):
        """Lazy load TrOCR model"""
        if self.model is None:
            self.processor = TrOCRProcessor.from_pretrained(
                'microsoft/trocr-base-printed'
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(
                'microsoft/trocr-base-printed'
            ).to(self.device)
    
    def extract(self, image: np.ndarray, language: str = 'eng') -> OCRResult:
        """Extract text using TrOCR"""
        try:
            self._initialize_model()
            
            # Convert to PIL Image
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Process image
            pixel_values = self.processor(
                images=pil_image, 
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate text
            generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            return OCRResult(
                text=text,
                confidence=0.85,  # TrOCR doesn't provide confidence
                engine="trocr",
                language=language,
                preprocessing_applied=[]
            )
            
        except Exception as e:
            print(f"TrOCR error: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine="trocr",
                language=language,
                preprocessing_applied=[]
            )

class OCRPipeline:
    """Main OCR processing pipeline"""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.tesseract = TesseractEngine()
        self.easyocr = EasyOCREngine()
        self.trocr = TrOCREngine()
        
        # Results cache
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _get_image_hash(self, image: np.ndarray) -> str:
        """Generate hash for image caching"""
        return hashlib.md5(image.tobytes()).hexdigest()
    
    async def process_image(
        self,
        image_path: str,
        engine: OCREngine = OCREngine.ALL,
        preprocess: bool = True,
        language: str = 'eng'
    ) -> OCRResult:
        """Process image through OCR pipeline"""
        
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
        
        # Check cache
        img_hash = self._get_image_hash(image)
        cache_key = f"{img_hash}_{engine.value}_{language}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Preprocessing
        preprocessing_applied = []
        if preprocess:
            # Auto enhance
            image, enhancements = self.preprocessor.auto_enhance(image)
            preprocessing_applied.extend(enhancements)
            
            # Deskew
            image, skew_angle = self.preprocessor.remove_skew(image)
            if skew_angle != 0:
                preprocessing_applied.append(f"deskew_{skew_angle:.2f}")
            
            # Upscale if image is small
            height, width = image.shape[:2]
            if width < 1000 or height < 1000:
                image = self.preprocessor.upscale_image(image, 2.0)
                preprocessing_applied.append("upscale_2x")
        
        # Run OCR based on selected engine
        if engine == OCREngine.TESSERACT:
            result = await self._run_tesseract(image, language)
        elif engine == OCREngine.EASYOCR:
            result = await self._run_easyocr(image, language)
        elif engine == OCREngine.TROCR:
            result = await self._run_trocr(image, language)
        elif engine == OCREngine.ALL:
            result = await self._run_ensemble(image, language)
        else:
            result = await self._run_tesseract(image, language)
        
        # Add preprocessing info
        result.preprocessing_applied = preprocessing_applied
        
        # Cache result
        self.cache[cache_key] = result
        
        return result
    
    async def _run_tesseract(self, image: np.ndarray, language: str) -> OCRResult:
        """Run Tesseract OCR asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.tesseract.extract,
            image,
            language
        )
    
    async def _run_easyocr(self, image: np.ndarray, language: str) -> OCRResult:
        """Run EasyOCR asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.easyocr.extract,
            image,
            language
        )
    
    async def _run_trocr(self, image: np.ndarray, language: str) -> OCRResult:
        """Run TrOCR asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.trocr.extract,
            image,
            language
        )
    
    async def _run_ensemble(self, image: np.ndarray, language: str) -> OCRResult:
        """Run all engines and combine results"""
        # Run all engines in parallel
        results = await asyncio.gather(
            self._run_tesseract(image, language),
            self._run_easyocr(image, language),
            self._run_trocr(image, language),
            return_exceptions=True
        )
        
        # Filter out failed results
        valid_results = [r for r in results if not isinstance(r, Exception) and r.text]
        
        if not valid_results:
            return OCRResult(
                text="",
                confidence=0.0,
                engine="ensemble",
                language=language,
                preprocessing_applied=[]
            )
        
        # Voting mechanism
        texts = [r.text for r in valid_results]
        confidences = [r.confidence for r in valid_results]
        
        # Use highest confidence result as primary
        best_idx = np.argmax(confidences)
        best_result = valid_results[best_idx]
        
        # Calculate ensemble confidence
        ensemble_confidence = np.mean(confidences)
        
        return OCRResult(
            text=best_result.text,
            confidence=ensemble_confidence,
            engine=f"ensemble_{best_result.engine}",
            language=language,
            preprocessing_applied=[],
            metadata={
                "all_results": [
                    {"engine": r.engine, "text": r.text[:100], "confidence": r.confidence}
                    for r in valid_results
                ]
            }
        )
    
    async def process_pdf(
        self,
        pdf_path: str,
        engine: OCREngine = OCREngine.ALL,
        pages: Optional[List[int]] = None
    ) -> List[OCRResult]:
        """Process PDF document"""
        results = []
        
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        
        # Process specified pages or all
        page_range = pages if pages else range(len(pdf_document))
        
        for page_num in page_range:
            page = pdf_document[page_num]
            
            # First try to extract text directly
            text = page.get_text()
            
            if text.strip():
                # Text extraction successful
                results.append(OCRResult(
                    text=text,
                    confidence=1.0,
                    engine="pdf_native",
                    language="auto",
                    preprocessing_applied=["direct_extraction"],
                    metadata={"page": page_num + 1}
                ))
            else:
                # Convert page to image for OCR
                mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to numpy array
                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Process with OCR
                result = await self.process_image(image, engine=engine)
                result.metadata["page"] = page_num + 1
                results.append(result)
        
        pdf_document.close()
        return results
    
    async def extract_structured_data(self, image_path: str) -> Dict:
        """Extract structured data like tables, forms, etc."""
        # Process image
        result = await self.process_image(image_path, engine=OCREngine.ALL)
        
        # Extract patterns
        extracted = {
            "emails": re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', result.text),
            "phones": re.findall(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,4}', result.text),
            "urls": re.findall(r'https?://(?:[-\w.])+(?:\:\d+)?(?:[/\w\s\-.%]*)', result.text),
            "dates": re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', result.text),
            "amounts": re.findall(r'\$[\d,]+\.?\d*', result.text),
            "raw_text": result.text,
            "confidence": result.confidence,
            "engine": result.engine
        }
        
        return extracted
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.cache.clear()


async def main():
    """Example usage"""
    pipeline = OCRPipeline()
    
    # Process image with all engines
    result = await pipeline.process_image(
        "/path/to/image.jpg",
        engine=OCREngine.ALL,
        preprocess=True
    )
    print(f"Text: {result.text[:200]}...")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Engine: {result.engine}")
    
    # Process PDF
    pdf_results = await pipeline.process_pdf(
        "/path/to/document.pdf",
        engine=OCREngine.TESSERACT
    )
    for i, result in enumerate(pdf_results):
        print(f"Page {i+1}: {result.text[:100]}...")
    
    # Extract structured data
    data = await pipeline.extract_structured_data("/path/to/form.jpg")
    print(f"Extracted emails: {data['emails']}")
    print(f"Extracted phones: {data['phones']}")
    
    pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
