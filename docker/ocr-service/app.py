#!/usr/bin/env python3
"""
OCR Service with Tesseract and FastAPI
Multi-language document processing with RabbitMQ integration
"""

import asyncio
import io
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import aiofiles
import cv2
import magic
import numpy as np
import pika
import pytesseract
import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from langdetect import detect
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance, ImageFilter
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
ocr_requests_total = Counter('ocr_requests_total', 'Total OCR requests', ['language', 'status'])
ocr_processing_time = Histogram('ocr_processing_seconds', 'OCR processing time')
ocr_queue_size = Counter('ocr_queue_size', 'OCR queue size')

class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Service configuration
    service_name: str = "ocr-service"
    debug: bool = False
    max_file_size: int = 50 * 1024 * 1024  # 50MB

    # RabbitMQ configuration
    rabbitmq_host: str = "rabbitmq"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "admin"
    rabbitmq_password: str = "admin123"
    rabbitmq_vhost: str = "/"
    ocr_queue: str = "ocr_processing"
    result_queue: str = "ocr_results"

    # OCR configuration
    supported_languages: List[str] = [
        "eng", "fra", "deu", "spa", "ita", "por", "rus",
        "chi_sim", "chi_tra", "jpn", "kor", "ara"
    ]
    ocr_config: str = "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[]{}\"'-/@#$%^&*+=<>|\\~`"

    # File processing
    temp_dir: str = "/tmp/ocr"
    upload_dir: str = "/app/uploads"
    processed_dir: str = "/app/processed"

    class Config:
        env_prefix = "OCR_"

settings = Settings()

# Create necessary directories
for directory in [settings.temp_dir, settings.upload_dir, settings.processed_dir]:
    Path(directory).mkdir(parents=True, exist_ok=True)

class OCRRequest(BaseModel):
    """OCR processing request model"""
    file_id: str = Field(..., description="Unique file identifier")
    language: Optional[str] = Field("auto", description="Language code or 'auto' for detection")
    preprocessing: bool = Field(True, description="Apply image preprocessing")
    extract_tables: bool = Field(False, description="Extract table structures")
    output_format: str = Field("text", description="Output format: text, hocr, pdf")

class OCRResult(BaseModel):
    """OCR processing result model"""
    file_id: str
    text: str
    confidence: float
    language: str
    page_count: int
    processing_time: float
    metadata: Dict
    tables: Optional[List[Dict]] = None
    timestamp: datetime

class OCRService:
    """Main OCR service class"""

    def __init__(self):
        self.connection = None
        self.channel = None
        self.setup_rabbitmq()

    def setup_rabbitmq(self):
        """Setup RabbitMQ connection and queues"""
        try:
            credentials = pika.PlainCredentials(
                settings.rabbitmq_user,
                settings.rabbitmq_password
            )
            parameters = pika.ConnectionParameters(
                host=settings.rabbitmq_host,
                port=settings.rabbitmq_port,
                virtual_host=settings.rabbitmq_vhost,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )

            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            # Declare queues
            self.channel.queue_declare(queue=settings.ocr_queue, durable=True)
            self.channel.queue_declare(queue=settings.result_queue, durable=True)

            logger.info("RabbitMQ connection established")

        except Exception as e:
            logger.error("Failed to setup RabbitMQ", error=str(e))
            raise

    def detect_language(self, text: str) -> str:
        """Detect text language"""
        if not text.strip():
            return "eng"

        try:
            detected = detect(text)
            # Map common language codes to Tesseract codes
            lang_mapping = {
                "en": "eng", "fr": "fra", "de": "deu", "es": "spa",
                "it": "ita", "pt": "por", "ru": "rus", "zh": "chi_sim",
                "ja": "jpn", "ko": "kor", "ar": "ara"
            }
            return lang_mapping.get(detected, "eng")
        except:
            return "eng"

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply image preprocessing for better OCR"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')

            # Convert to numpy array
            img_array = np.array(image)

            # Apply Gaussian blur to reduce noise
            img_array = cv2.GaussianBlur(img_array, (1, 1), 0)

            # Apply threshold to get a binary image
            _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Convert back to PIL Image
            processed_image = Image.fromarray(img_array)

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(processed_image)
            processed_image = enhancer.enhance(2.0)

            return processed_image

        except Exception as e:
            logger.warning("Image preprocessing failed", error=str(e))
            return image

    def extract_text_from_image(self, image: Image.Image, language: str, preprocessing: bool = True) -> Dict:
        """Extract text from image using Tesseract"""
        try:
            if preprocessing:
                image = self.preprocess_image(image)

            # Auto-detect language if needed
            if language == "auto":
                # Quick OCR for language detection
                sample_text = pytesseract.image_to_string(image, config="--psm 6")
                language = self.detect_language(sample_text)

            # Ensure language is supported
            if language not in settings.supported_languages:
                language = "eng"

            # Extract text with confidence
            ocr_data = pytesseract.image_to_data(
                image,
                lang=language,
                config=settings.ocr_config,
                output_type=pytesseract.Output.DICT
            )

            # Calculate confidence
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Extract text
            text = pytesseract.image_to_string(image, lang=language, config=settings.ocr_config)

            return {
                "text": text.strip(),
                "confidence": avg_confidence,
                "language": language,
                "word_count": len(text.split()),
                "char_count": len(text)
            }

        except Exception as e:
            logger.error("OCR extraction failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"OCR processing failed: {str(e)}"
            )

    def process_pdf(self, pdf_bytes: bytes, language: str, preprocessing: bool = True) -> Dict:
        """Process PDF document"""
        try:
            # Convert PDF to images
            images = convert_from_bytes(pdf_bytes, dpi=300)

            all_text = []
            total_confidence = 0
            page_count = len(images)

            for i, image in enumerate(images):
                logger.info(f"Processing PDF page {i+1}/{page_count}")

                result = self.extract_text_from_image(image, language, preprocessing)
                all_text.append(result["text"])
                total_confidence += result["confidence"]

            combined_text = "\n\n".join(all_text)
            avg_confidence = total_confidence / page_count if page_count > 0 else 0

            # Auto-detect language from combined text if needed
            if language == "auto":
                language = self.detect_language(combined_text)

            return {
                "text": combined_text,
                "confidence": avg_confidence,
                "language": language,
                "page_count": page_count,
                "word_count": len(combined_text.split()),
                "char_count": len(combined_text)
            }

        except Exception as e:
            logger.error("PDF processing failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"PDF processing failed: {str(e)}"
            )

    async def process_file(self, file_content: bytes, filename: str, language: str = "auto", preprocessing: bool = True) -> Dict:
        """Process uploaded file"""
        start_time = datetime.now()

        try:
            # Detect file type
            file_type = magic.from_buffer(file_content, mime=True)
            logger.info("Processing file", filename=filename, file_type=file_type)

            result = None

            if file_type == "application/pdf":
                result = self.process_pdf(file_content, language, preprocessing)

            elif file_type.startswith("image/"):
                image = Image.open(io.BytesIO(file_content))
                result = self.extract_text_from_image(image, language, preprocessing)
                result["page_count"] = 1

            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file_type}"
                )

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Add metadata
            result.update({
                "filename": filename,
                "file_type": file_type,
                "file_size": len(file_content),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            })

            # Update metrics
            ocr_requests_total.labels(language=result["language"], status="success").inc()
            ocr_processing_time.observe(processing_time)

            return result

        except HTTPException:
            raise
        except Exception as e:
            # Update error metrics
            ocr_requests_total.labels(language=language, status="error").inc()
            logger.error("File processing failed", filename=filename, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"File processing failed: {str(e)}"
            )

    def publish_result(self, result: Dict):
        """Publish OCR result to RabbitMQ"""
        try:
            if self.channel and not self.connection.is_closed:
                self.channel.basic_publish(
                    exchange='',
                    routing_key=settings.result_queue,
                    body=json.dumps(result, default=str),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Make message persistent
                        content_type='application/json'
                    )
                )
                logger.info("Result published to queue", file_id=result.get("file_id"))
            else:
                logger.warning("RabbitMQ connection not available")

        except Exception as e:
            logger.error("Failed to publish result", error=str(e))

# Initialize OCR service
ocr_service = OCRService()

# FastAPI application
app = FastAPI(
    title="OCR Service",
    description="Multi-language OCR service with Tesseract and RabbitMQ integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("OCR Service starting up")

    # Verify Tesseract installation
    try:
        version = pytesseract.get_tesseract_version()
        logger.info("Tesseract version", version=str(version))
    except Exception as e:
        logger.error("Tesseract not available", error=str(e))
        raise

    # Test language support
    available_langs = pytesseract.get_languages()
    logger.info("Available languages", languages=available_langs)

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("OCR Service shutting down")
    if ocr_service.connection and not ocr_service.connection.is_closed:
        ocr_service.connection.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Tesseract
        pytesseract.get_tesseract_version()

        # Check RabbitMQ
        rabbitmq_status = "connected" if (
            ocr_service.connection and
            not ocr_service.connection.is_closed
        ) else "disconnected"

        return {
            "status": "healthy",
            "service": settings.service_name,
            "timestamp": datetime.now().isoformat(),
            "tesseract": "available",
            "rabbitmq": rabbitmq_status,
            "supported_languages": settings.supported_languages
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/ocr/process", response_model=OCRResult)
async def process_ocr_request(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    preprocessing: bool = Form(True),
    extract_tables: bool = Form(False),
    output_format: str = Form("text")
):
    """Process file for OCR extraction"""

    # Validate file size
    file_content = await file.read()
    if len(file_content) > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {settings.max_file_size} bytes"
        )

    # Generate unique file ID
    file_id = str(uuid.uuid4())

    try:
        # Process the file
        result = await ocr_service.process_file(
            file_content,
            file.filename,
            language,
            preprocessing
        )

        # Create OCR result
        ocr_result = OCRResult(
            file_id=file_id,
            text=result["text"],
            confidence=result["confidence"],
            language=result["language"],
            page_count=result["page_count"],
            processing_time=result["processing_time"],
            metadata={
                "filename": result["filename"],
                "file_type": result["file_type"],
                "file_size": result["file_size"],
                "word_count": result["word_count"],
                "char_count": result["char_count"]
            },
            timestamp=datetime.now()
        )

        # Publish result to queue
        ocr_service.publish_result(ocr_result.dict())

        return ocr_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("OCR processing failed", file_id=file_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )

@app.get("/ocr/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "supported_languages": settings.supported_languages,
        "available_languages": pytesseract.get_languages()
    }

@app.get("/ocr/status")
async def get_service_status():
    """Get detailed service status"""
    return {
        "service": settings.service_name,
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "max_file_size": settings.max_file_size,
            "supported_languages": len(settings.supported_languages),
            "rabbitmq_host": settings.rabbitmq_host,
            "debug_mode": settings.debug
        },
        "tesseract_version": str(pytesseract.get_tesseract_version()),
        "available_languages": pytesseract.get_languages()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=settings.debug,
        workers=4 if not settings.debug else 1
    )