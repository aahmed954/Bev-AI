#!/usr/bin/env python3
"""
Enhanced OCR Pipeline with Tesseract Integration
Complete document processing with multi-language support
"""

import asyncio
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pdf2image
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple, Optional
import json
import hashlib
from dataclasses import dataclass
import logging
from pathlib import Path
import easyocr
import boto3
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """OCR processing result"""
    text: str
    confidence: float
    language: str
    bounding_boxes: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class EnhancedOCRPipeline:
    """
    Advanced OCR pipeline with multiple engines and preprocessing
    """
    
    def __init__(self):
        # Initialize OCR engines
        self.tesseract_langs = ['eng', 'fra', 'deu', 'spa', 'chi_sim', 'jpn', 'rus', 'ara']
        self.easyocr_reader = easyocr.Reader(
            ['en', 'fr', 'de', 'es', 'ch_sim', 'ja', 'ru', 'ar'],
            gpu=torch.cuda.is_available()
        )
        
        # LayoutLM for document understanding
        self.layoutlm_processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base"
        )
        self.layoutlm_model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base"
        )
        
        # AWS Textract client (for complex documents)
        self.textract_client = boto3.client('textract', region_name='us-east-1')
        
        # Preprocessing configurations
        self.preprocessing_configs = {
            'standard': self.standard_preprocessing,
            'handwritten': self.handwritten_preprocessing,
            'scanned': self.scanned_preprocessing,
            'screenshot': self.screenshot_preprocessing,
            'dark_mode': self.dark_mode_preprocessing
        }
        
        # Language detection model
        from langdetect import detect_langs
        self.detect_langs = detect_langs
    
    async def process_document(self, file_path: str, 
                              doc_type: str = 'auto') -> OCRResult:
        """
        Process document with automatic preprocessing and OCR
        """
        logger.info(f"Processing document: {file_path}")
        
        # Determine file type
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            images = await self.pdf_to_images(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            images = [Image.open(file_path)]
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Process each page/image
        all_results = []
        
        for idx, image in enumerate(images):
            # Detect document type if auto
            if doc_type == 'auto':
                doc_type = await self.detect_document_type(image)
            
            # Preprocess image
            processed_image = await self.preprocess_image(image, doc_type)
            
            # Run multiple OCR engines
            results = await asyncio.gather(
                self.tesseract_ocr(processed_image),
                self.easyocr_ocr(processed_image),
                self.layoutlm_ocr(processed_image) if doc_type in ['form', 'invoice'] else asyncio.sleep(0)
            )
            
            # Merge and validate results
            merged_result = await self.merge_ocr_results(results)
            all_results.append(merged_result)
        
        # Combine all pages
        final_result = await self.combine_page_results(all_results)
        
        # Post-process text
        final_result.text = await self.post_process_text(final_result.text)
        
        return final_result
    
    async def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF to images for OCR"""
        
        # Try PyMuPDF first (faster)
        try:
            pdf_document = fitz.open(pdf_path)
            images = []
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            
            pdf_document.close()
            return images
            
        except Exception as e:
            logger.warning(f"PyMuPDF failed, falling back to pdf2image: {e}")
            
            # Fallback to pdf2image
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=300,
                fmt='png',
                thread_count=4
            )
            return images
    
    async def detect_document_type(self, image: Image.Image) -> str:
        """Detect document type using image analysis"""
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Analyze image characteristics
        height, width = img_array.shape[:2]
        aspect_ratio = width / height
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Detect lines (forms often have many lines)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        line_count = len(lines) if lines is not None else 0
        
        # Calculate text density
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_pixels = np.sum(binary == 0)
        text_density = text_pixels / (height * width)
        
        # Classify document type
        if line_count > 50:
            return 'form'
        elif text_density < 0.1:
            return 'handwritten'
        elif aspect_ratio > 1.5:
            return 'screenshot'
        else:
            return 'standard'
    
    async def preprocess_image(self, image: Image.Image, doc_type: str) -> Image.Image:
        """Apply preprocessing based on document type"""
        
        preprocessor = self.preprocessing_configs.get(doc_type, self.standard_preprocessing)
        return await preprocessor(image)
    
    async def standard_preprocessing(self, image: Image.Image) -> Image.Image:
        """Standard preprocessing for clean documents"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)
        
        # Denoise
        img_array = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        
        return Image.fromarray(denoised)
    
    async def handwritten_preprocessing(self, image: Image.Image) -> Image.Image:
        """Preprocessing for handwritten documents"""
        
        img_array = np.array(image.convert('L'))  # Convert to grayscale
        
        # Binarization with adaptive threshold
        binary = cv2.adaptiveThreshold(
            img_array,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Morphological operations to connect text
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Skew correction
        coords = np.column_stack(np.where(processed > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
        
        if abs(angle) > 0.5:
            (h, w) = processed.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            processed = cv2.warpAffine(
                processed,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
        
        return Image.fromarray(processed)
    
    async def scanned_preprocessing(self, image: Image.Image) -> Image.Image:
        """Preprocessing for scanned documents"""
        
        img_array = np.array(image)
        
        # Remove shadows
        rgb_planes = cv2.split(img_array)
        result_planes = []
        
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(
                diff_img, None, alpha=0, beta=255, 
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
            )
            result_planes.append(norm_img)
        
        result = cv2.merge(result_planes)
        
        # Enhance quality
        kernel_sharpen = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
        result = cv2.filter2D(result, -1, kernel_sharpen)
        
        return Image.fromarray(result)
    
    async def screenshot_preprocessing(self, image: Image.Image) -> Image.Image:
        """Preprocessing for screenshots"""
        
        # Upscale if resolution is low
        width, height = image.size
        if width < 1920:
            scale = 1920 / width
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.LANCZOS)
        
        # Enhance text clarity
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        return image
    
    async def dark_mode_preprocessing(self, image: Image.Image) -> Image.Image:
        """Preprocessing for dark mode images"""
        
        # Invert colors
        img_array = np.array(image)
        inverted = 255 - img_array
        
        # Enhance contrast
        image = Image.fromarray(inverted)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        return image
    
    async def tesseract_ocr(self, image: Image.Image) -> Dict[str, Any]:
        """Run Tesseract OCR with multiple languages"""
        
        # Detect language
        sample_text = pytesseract.image_to_string(image, lang='eng')[:500]
        
        try:
            detected_langs = self.detect_langs(sample_text)
            primary_lang = detected_langs[0].lang if detected_langs else 'en'
            
            # Map to Tesseract language code
            lang_map = {
                'en': 'eng', 'fr': 'fra', 'de': 'deu',
                'es': 'spa', 'zh': 'chi_sim', 'ja': 'jpn',
                'ru': 'rus', 'ar': 'ara'
            }
            
            tesseract_lang = lang_map.get(primary_lang, 'eng')
            
        except:
            tesseract_lang = 'eng'
        
        # Run OCR with detected language
        custom_config = r'--oem 3 --psm 6'
        
        # Get detailed data
        ocr_data = pytesseract.image_to_data(
            image,
            lang=tesseract_lang,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Get plain text
        text = pytesseract.image_to_string(
            image,
            lang=tesseract_lang,
            config=custom_config
        )
        
        # Calculate average confidence
        confidences = [
            int(conf) for conf in ocr_data['conf'] 
            if int(conf) > 0
        ]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Extract bounding boxes
        bounding_boxes = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 0:
                bounding_boxes.append({
                    'text': ocr_data['text'][i],
                    'confidence': ocr_data['conf'][i],
                    'bbox': {
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i]
                    }
                })
        
        return {
            'engine': 'tesseract',
            'text': text,
            'confidence': avg_confidence / 100,
            'language': tesseract_lang,
            'bounding_boxes': bounding_boxes
        }
    
    async def easyocr_ocr(self, image: Image.Image) -> Dict[str, Any]:
        """Run EasyOCR for additional validation"""
        
        # Convert PIL to numpy
        img_array = np.array(image)
        
        # Run EasyOCR
        results = self.easyocr_reader.readtext(img_array)
        
        # Extract text and bounding boxes
        text_parts = []
        bounding_boxes = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            text_parts.append(text)
            confidences.append(confidence)
            
            # Convert bbox format
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            bounding_boxes.append({
                'text': text,
                'confidence': confidence * 100,
                'bbox': {
                    'x': min(x_coords),
                    'y': min(y_coords),
                    'width': max(x_coords) - min(x_coords),
                    'height': max(y_coords) - min(y_coords)
                }
            })
        
        return {
            'engine': 'easyocr',
            'text': ' '.join(text_parts),
            'confidence': np.mean(confidences) if confidences else 0,
            'language': 'multi',
            'bounding_boxes': bounding_boxes
        }
    
    async def layoutlm_ocr(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """Use LayoutLM for structured document understanding"""
        
        # This is particularly good for forms and invoices
        # Requires additional setup for full implementation
        
        return None  # Placeholder for now
    
    async def textract_ocr(self, image_bytes: bytes) -> Dict[str, Any]:
        """Use AWS Textract for complex documents"""
        
        try:
            response = self.textract_client.detect_document_text(
                Document={'Bytes': image_bytes}
            )
            
            # Extract text
            text_parts = []
            bounding_boxes = []
            
            for block in response['Blocks']:
                if block['BlockType'] == 'LINE':
                    text_parts.append(block['Text'])
                    
                    bbox = block['Geometry']['BoundingBox']
                    bounding_boxes.append({
                        'text': block['Text'],
                        'confidence': block['Confidence'],
                        'bbox': {
                            'x': bbox['Left'],
                            'y': bbox['Top'],
                            'width': bbox['Width'],
                            'height': bbox['Height']
                        }
                    })
            
            return {
                'engine': 'textract',
                'text': ' '.join(text_parts),
                'confidence': 0.95,  # Textract is generally very accurate
                'language': 'auto',
                'bounding_boxes': bounding_boxes
            }
            
        except Exception as e:
            logger.error(f"Textract failed: {e}")
            return None
    
    async def merge_ocr_results(self, results: List[Dict[str, Any]]) -> OCRResult:
        """Merge results from multiple OCR engines"""
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        if not results:
            return OCRResult(
                text="",
                confidence=0,
                language="unknown",
                bounding_boxes=[],
                metadata={}
            )
        
        # Use voting for best text
        if len(results) == 1:
            best_result = results[0]
        else:
            # Compare results and use highest confidence
            best_result = max(results, key=lambda x: x['confidence'])
            
            # Validate with other engines
            for other in results:
                if other != best_result:
                    # Use fuzzy matching to validate
                    from difflib import SequenceMatcher
                    
                    similarity = SequenceMatcher(
                        None,
                        best_result['text'],
                        other['text']
                    ).ratio()
                    
                    # If significantly different, log warning
                    if similarity < 0.8:
                        logger.warning(
                            f"OCR results differ significantly between "
                            f"{best_result['engine']} and {other['engine']}"
                        )
        
        return OCRResult(
            text=best_result['text'],
            confidence=best_result['confidence'],
            language=best_result['language'],
            bounding_boxes=best_result['bounding_boxes'],
            metadata={
                'engine': best_result['engine'],
                'all_results': results
            }
        )
    
    async def combine_page_results(self, results: List[OCRResult]) -> OCRResult:
        """Combine results from multiple pages"""
        
        if not results:
            return OCRResult(
                text="",
                confidence=0,
                language="unknown",
                bounding_boxes=[],
                metadata={}
            )
        
        # Combine text with page separators
        combined_text = '\n\n--- Page Break ---\n\n'.join(
            r.text for r in results
        )
        
        # Average confidence
        avg_confidence = np.mean([r.confidence for r in results])
        
        # Combine bounding boxes with page info
        all_boxes = []
        for page_num, result in enumerate(results):
            for box in result.bounding_boxes:
                box_with_page = box.copy()
                box_with_page['page'] = page_num + 1
                all_boxes.append(box_with_page)
        
        # Determine primary language
        languages = [r.language for r in results]
        primary_language = max(set(languages), key=languages.count)
        
        return OCRResult(
            text=combined_text,
            confidence=avg_confidence,
            language=primary_language,
            bounding_boxes=all_boxes,
            metadata={
                'page_count': len(results),
                'page_confidences': [r.confidence for r in results]
            }
        )
    
    async def post_process_text(self, text: str) -> str:
        """Clean and format extracted text"""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix common OCR errors
        replacements = {
            'rn': 'm',  # Common OCR error
            '0': 'O',  # Zero to O in appropriate contexts
            '1': 'I',  # One to I in appropriate contexts
        }
        
        # Context-aware replacements would go here
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        return text.strip()

# Integration with existing pipeline
class OCRIntegration:
    """Integrate enhanced OCR with BEV pipeline"""
    
    def __init__(self):
        self.ocr_pipeline = EnhancedOCRPipeline()
        self.vector_db = None  # Connect to Qdrant
        self.storage_client = boto3.client('s3')
    
    async def process_document_batch(self, document_urls: List[str]) -> List[OCRResult]:
        """Process multiple documents in parallel"""
        
        tasks = [
            self.process_single_document(url) 
            for url in document_urls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle errors
        processed_results = []
        for url, result in zip(document_urls, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {url}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def process_single_document(self, document_url: str) -> OCRResult:
        """Process single document from URL or path"""
        
        # Download if URL
        if document_url.startswith('http'):
            local_path = await self.download_document(document_url)
        else:
            local_path = document_url
        
        # Process with OCR
        result = await self.ocr_pipeline.process_document(local_path)
        
        # Store in vector database
        if result.text:
            await self.store_in_vector_db(result, document_url)
        
        # Store processed text in S3
        await self.store_processed_text(result, document_url)
        
        return result
    
    async def download_document(self, url: str) -> str:
        """Download document to local storage"""
        
        import aiohttp
        import tempfile
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                content = await response.read()
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
                    f.write(content)
                    return f.name
    
    async def store_in_vector_db(self, result: OCRResult, source_url: str):
        """Store OCR result in vector database for search"""
        
        # Generate embedding
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(result.text)
        
        # Store in Qdrant
        # Implementation here
        
    async def store_processed_text(self, result: OCRResult, source_url: str):
        """Store processed text in S3"""
        
        # Generate unique key
        doc_hash = hashlib.sha256(source_url.encode()).hexdigest()
        key = f"ocr-results/{doc_hash}/extracted_text.json"
        
        # Prepare data
        data = {
            'source_url': source_url,
            'text': result.text,
            'confidence': result.confidence,
            'language': result.language,
            'metadata': result.metadata,
            'processed_at': datetime.utcnow().isoformat()
        }
        
        # Upload to S3
        self.storage_client.put_object(
            Bucket='bev-documents',
            Key=key,
            Body=json.dumps(data),
            ContentType='application/json'
        )
        
        logger.info(f"Stored OCR result to S3: {key}")

# CLI for testing
async def main():
    """Test OCR pipeline"""
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_ocr_pipeline.py <document_path>")
        return
    
    pipeline = EnhancedOCRPipeline()
    result = await pipeline.process_document(sys.argv[1])
    
    print(f"Extracted Text ({result.confidence*100:.1f}% confidence):")
    print("-" * 50)
    print(result.text)
    print("-" * 50)
    print(f"Language: {result.language}")
    print(f"Bounding Boxes: {len(result.bounding_boxes)}")

if __name__ == "__main__":
    asyncio.run(main())
