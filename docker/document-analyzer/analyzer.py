#!/usr/bin/env python3
"""
Document Analyzer with NLP Processing and Neo4j Integration
Extracts entities, relationships, and metadata from documents
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import nltk
import pandas as pd
import pika
import spacy
import structlog
from fastapi import FastAPI, HTTPException, status
from flair.data import Sentence
from flair.models import SequenceTagger
from neo4j import GraphDatabase
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer
from textblob import TextBlob

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
analysis_requests_total = Counter('analysis_requests_total', 'Total analysis requests', ['type', 'status'])
analysis_processing_time = Histogram('analysis_processing_seconds', 'Analysis processing time')
entities_extracted_total = Counter('entities_extracted_total', 'Total entities extracted', ['entity_type'])
relationships_extracted_total = Counter('relationships_extracted_total', 'Total relationships extracted')

class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Service configuration
    service_name: str = "document-analyzer"
    debug: bool = False

    # Neo4j configuration
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "admin123"
    neo4j_database: str = "neo4j"

    # RabbitMQ configuration
    rabbitmq_host: str = "rabbitmq"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "admin"
    rabbitmq_password: str = "admin123"
    rabbitmq_vhost: str = "/"
    analysis_queue: str = "document_analysis"
    result_queue: str = "analysis_results"

    # NLP configuration
    spacy_model: str = "en_core_web_lg"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    max_text_length: int = 1000000  # 1MB
    chunk_size: int = 1000
    overlap_size: int = 100

    # Processing configuration
    extract_entities: bool = True
    extract_relationships: bool = True
    extract_keywords: bool = True
    extract_topics: bool = True
    sentiment_analysis: bool = True

    class Config:
        env_prefix = "ANALYZER_"

settings = Settings()

class DocumentRequest(BaseModel):
    """Document analysis request model"""
    document_id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., description="Document text content")
    metadata: Optional[Dict] = Field(default_factory=dict, description="Document metadata")
    analysis_types: List[str] = Field(
        default=["entities", "relationships", "keywords", "sentiment"],
        description="Types of analysis to perform"
    )

class Entity(BaseModel):
    """Entity model"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    metadata: Optional[Dict] = None

class Relationship(BaseModel):
    """Relationship model"""
    subject: str
    predicate: str
    object: str
    confidence: float
    context: str
    metadata: Optional[Dict] = None

class AnalysisResult(BaseModel):
    """Document analysis result model"""
    document_id: str
    entities: List[Entity]
    relationships: List[Relationship]
    keywords: List[str]
    topics: List[Dict]
    sentiment: Dict
    summary: str
    metadata: Dict
    processing_time: float
    timestamp: datetime

class DocumentAnalyzer:
    """Main document analyzer class"""

    def __init__(self):
        self.nlp = None
        self.sentence_transformer = None
        self.ner_tagger = None
        self.neo4j_driver = None
        self.connection = None
        self.channel = None

        self.setup_nlp_models()
        self.setup_neo4j()
        self.setup_rabbitmq()

    def setup_nlp_models(self):
        """Initialize NLP models"""
        try:
            logger.info("Loading NLP models")

            # Load spaCy model
            self.nlp = spacy.load(settings.spacy_model)

            # Add custom pipeline components
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

            # Load sentence transformer for embeddings
            self.sentence_transformer = SentenceTransformer(settings.sentence_transformer_model)

            # Load Flair NER model for additional entity types
            self.ner_tagger = SequenceTagger.load("flair/ner-english-large")

            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')

            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')

            logger.info("NLP models loaded successfully")

        except Exception as e:
            logger.error("Failed to load NLP models", error=str(e))
            raise

    def setup_neo4j(self):
        """Setup Neo4j database connection"""
        try:
            self.neo4j_driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )

            # Test connection
            with self.neo4j_driver.session(database=settings.neo4j_database) as session:
                session.run("RETURN 1")

            # Create indexes and constraints
            self.create_neo4j_schema()

            logger.info("Neo4j connection established")

        except Exception as e:
            logger.error("Failed to setup Neo4j", error=str(e))
            raise

    def create_neo4j_schema(self):
        """Create Neo4j schema (indexes and constraints)"""
        try:
            with self.neo4j_driver.session(database=settings.neo4j_database) as session:
                # Create indexes for better performance
                queries = [
                    "CREATE INDEX entity_text_index IF NOT EXISTS FOR (e:Entity) ON (e.text)",
                    "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                    "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id)",
                    "CREATE CONSTRAINT entity_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.text, e.type) IS UNIQUE",
                    "CREATE CONSTRAINT document_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"
                ]

                for query in queries:
                    try:
                        session.run(query)
                    except Exception as e:
                        # Ignore if constraint/index already exists
                        if "already exists" not in str(e).lower():
                            logger.warning("Schema creation warning", query=query, error=str(e))

        except Exception as e:
            logger.error("Failed to create Neo4j schema", error=str(e))

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
            self.channel.queue_declare(queue=settings.analysis_queue, durable=True)
            self.channel.queue_declare(queue=settings.result_queue, durable=True)

            logger.info("RabbitMQ connection established")

        except Exception as e:
            logger.error("Failed to setup RabbitMQ", error=str(e))
            raise

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text"""
        entities = []

        try:
            # Process with spaCy
            doc = self.nlp(text)

            for ent in doc.ents:
                entities.append(Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8,  # spaCy doesn't provide confidence scores
                    metadata={
                        "source": "spacy",
                        "description": spacy.explain(ent.label_)
                    }
                ))

            # Process with Flair for additional entities
            sentence = Sentence(text)
            self.ner_tagger.predict(sentence)

            for entity in sentence.get_spans('ner'):
                # Avoid duplicates
                overlapping = any(
                    abs(e.start - entity.start_position) < 5 and
                    abs(e.end - entity.end_position) < 5
                    for e in entities
                )

                if not overlapping:
                    entities.append(Entity(
                        text=entity.text,
                        label=entity.get_label("ner").value,
                        start=entity.start_position,
                        end=entity.end_position,
                        confidence=entity.get_label("ner").score,
                        metadata={
                            "source": "flair"
                        }
                    ))

            # Update metrics
            for entity in entities:
                entities_extracted_total.labels(entity_type=entity.label).inc()

            logger.info(f"Extracted {len(entities)} entities")
            return entities

        except Exception as e:
            logger.error("Entity extraction failed", error=str(e))
            return []

    def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities"""
        relationships = []

        try:
            doc = self.nlp(text)

            # Simple dependency parsing approach
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ in ["nsubj", "dobj", "pobj"]:
                        # Find subject-verb-object patterns
                        subj = None
                        obj = None
                        verb = token.head

                        # Find subject
                        for child in verb.children:
                            if child.dep_ in ["nsubj", "nsubjpass"]:
                                subj = child.text

                        # Find object
                        for child in verb.children:
                            if child.dep_ in ["dobj", "pobj"]:
                                obj = child.text

                        if subj and obj and subj != obj:
                            relationships.append(Relationship(
                                subject=subj,
                                predicate=verb.lemma_,
                                object=obj,
                                confidence=0.7,
                                context=sent.text,
                                metadata={
                                    "source": "dependency_parsing",
                                    "sentence_start": sent.start_char,
                                    "sentence_end": sent.end_char
                                }
                            ))

            # Update metrics
            relationships_extracted_total.inc(len(relationships))

            logger.info(f"Extracted {len(relationships)} relationships")
            return relationships

        except Exception as e:
            logger.error("Relationship extraction failed", error=str(e))
            return []

    def extract_keywords(self, text: str, top_k: int = 20) -> List[str]:
        """Extract keywords using TF-IDF and named entities"""
        try:
            doc = self.nlp(text)

            # Get noun phrases and named entities
            keywords = set()

            # Add named entities
            for ent in doc.ents:
                if len(ent.text.split()) <= 3:  # Limit to short phrases
                    keywords.add(ent.text.lower())

            # Add significant noun phrases
            for chunk in doc.noun_chunks:
                if (len(chunk.text.split()) <= 3 and
                    chunk.root.pos_ in ["NOUN", "PROPN"] and
                    not chunk.root.is_stop):
                    keywords.add(chunk.text.lower())

            # Filter and rank keywords
            filtered_keywords = []
            for keyword in keywords:
                if (len(keyword) > 2 and
                    not keyword.isdigit() and
                    keyword.replace(" ", "").isalpha()):
                    filtered_keywords.append(keyword)

            return sorted(filtered_keywords)[:top_k]

        except Exception as e:
            logger.error("Keyword extraction failed", error=str(e))
            return []

    def extract_topics(self, text: str, num_topics: int = 5) -> List[Dict]:
        """Extract topics using LDA-based approach"""
        try:
            # Simple topic extraction using noun phrases frequency
            doc = self.nlp(text)

            # Count noun phrases
            phrase_counts = {}
            for chunk in doc.noun_chunks:
                phrase = chunk.text.lower().strip()
                if (len(phrase.split()) <= 3 and
                    not chunk.root.is_stop and
                    len(phrase) > 2):
                    phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

            # Get top phrases as topics
            topics = []
            sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)

            for i, (phrase, count) in enumerate(sorted_phrases[:num_topics]):
                topics.append({
                    "topic_id": i,
                    "keywords": phrase.split(),
                    "weight": count / len(sorted_phrases) if sorted_phrases else 0,
                    "representative_phrase": phrase
                })

            return topics

        except Exception as e:
            logger.error("Topic extraction failed", error=str(e))
            return []

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze document sentiment"""
        try:
            blob = TextBlob(text)

            # Calculate sentiment scores
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1

            # Determine sentiment label
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"

            return {
                "polarity": polarity,
                "subjectivity": subjectivity,
                "label": label,
                "confidence": abs(polarity)
            }

        except Exception as e:
            logger.error("Sentiment analysis failed", error=str(e))
            return {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "label": "neutral",
                "confidence": 0.0
            }

    def generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate document summary"""
        try:
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]

            if len(sentences) <= max_sentences:
                return text

            # Simple extractive summarization
            # Score sentences by entity density and position
            scored_sentences = []

            for i, sentence in enumerate(sentences):
                sent_doc = self.nlp(sentence)
                entity_count = len(sent_doc.ents)
                position_score = 1 - (i / len(sentences))  # Earlier sentences get higher scores
                score = entity_count + position_score

                scored_sentences.append((score, sentence))

            # Get top sentences
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            summary_sentences = [sent for score, sent in scored_sentences[:max_sentences]]

            # Maintain original order
            original_order = []
            for sent in sentences:
                if sent in summary_sentences:
                    original_order.append(sent)

            return " ".join(original_order)

        except Exception as e:
            logger.error("Summary generation failed", error=str(e))
            return text[:500] + "..." if len(text) > 500 else text

    def store_in_neo4j(self, document_id: str, entities: List[Entity],
                       relationships: List[Relationship], metadata: Dict):
        """Store analysis results in Neo4j graph database"""
        try:
            with self.neo4j_driver.session(database=settings.neo4j_database) as session:

                # Create document node
                session.run(
                    """
                    MERGE (d:Document {id: $doc_id})
                    SET d.created_at = datetime(),
                        d.metadata = $metadata
                    """,
                    doc_id=document_id,
                    metadata=metadata
                )

                # Create entity nodes and relationships to document
                for entity in entities:
                    session.run(
                        """
                        MERGE (e:Entity {text: $text, type: $type})
                        SET e.confidence = $confidence,
                            e.metadata = $metadata
                        WITH e
                        MATCH (d:Document {id: $doc_id})
                        MERGE (d)-[:CONTAINS_ENTITY]->(e)
                        """,
                        text=entity.text,
                        type=entity.label,
                        confidence=entity.confidence,
                        metadata=entity.metadata or {},
                        doc_id=document_id
                    )

                # Create relationships between entities
                for rel in relationships:
                    session.run(
                        """
                        MATCH (subj:Entity {text: $subject})
                        MATCH (obj:Entity {text: $object})
                        MERGE (subj)-[r:RELATES_TO {predicate: $predicate}]->(obj)
                        SET r.confidence = $confidence,
                            r.context = $context,
                            r.metadata = $metadata
                        """,
                        subject=rel.subject,
                        object=rel.object,
                        predicate=rel.predicate,
                        confidence=rel.confidence,
                        context=rel.context,
                        metadata=rel.metadata or {}
                    )

                logger.info("Results stored in Neo4j", document_id=document_id)

        except Exception as e:
            logger.error("Failed to store in Neo4j", document_id=document_id, error=str(e))

    async def analyze_document(self, request: DocumentRequest) -> AnalysisResult:
        """Analyze document and extract information"""
        start_time = datetime.now()

        try:
            logger.info("Starting document analysis", document_id=request.document_id)

            # Validate text length
            if len(request.text) > settings.max_text_length:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Text length exceeds maximum of {settings.max_text_length} characters"
                )

            # Initialize results
            entities = []
            relationships = []
            keywords = []
            topics = []
            sentiment = {}
            summary = ""

            # Perform requested analyses
            if "entities" in request.analysis_types:
                entities = self.extract_entities(request.text)

            if "relationships" in request.analysis_types and entities:
                relationships = self.extract_relationships(request.text, entities)

            if "keywords" in request.analysis_types:
                keywords = self.extract_keywords(request.text)

            if "topics" in request.analysis_types:
                topics = self.extract_topics(request.text)

            if "sentiment" in request.analysis_types:
                sentiment = self.analyze_sentiment(request.text)

            if "summary" in request.analysis_types:
                summary = self.generate_summary(request.text)

            # Store in Neo4j
            self.store_in_neo4j(request.document_id, entities, relationships, request.metadata)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Create result
            result = AnalysisResult(
                document_id=request.document_id,
                entities=entities,
                relationships=relationships,
                keywords=keywords,
                topics=topics,
                sentiment=sentiment,
                summary=summary,
                metadata={
                    **request.metadata,
                    "analysis_types": request.analysis_types,
                    "text_length": len(request.text),
                    "entity_count": len(entities),
                    "relationship_count": len(relationships)
                },
                processing_time=processing_time,
                timestamp=datetime.now()
            )

            # Update metrics
            analysis_requests_total.labels(type="document", status="success").inc()
            analysis_processing_time.observe(processing_time)

            logger.info("Document analysis completed",
                       document_id=request.document_id,
                       processing_time=processing_time)

            return result

        except HTTPException:
            raise
        except Exception as e:
            analysis_requests_total.labels(type="document", status="error").inc()
            logger.error("Document analysis failed",
                        document_id=request.document_id,
                        error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis failed: {str(e)}"
            )

    def publish_result(self, result: AnalysisResult):
        """Publish analysis result to RabbitMQ"""
        try:
            if self.channel and not self.connection.is_closed:
                self.channel.basic_publish(
                    exchange='',
                    routing_key=settings.result_queue,
                    body=result.json(),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Make message persistent
                        content_type='application/json'
                    )
                )
                logger.info("Result published to queue", document_id=result.document_id)
            else:
                logger.warning("RabbitMQ connection not available")

        except Exception as e:
            logger.error("Failed to publish result", error=str(e))

# Initialize analyzer
analyzer = DocumentAnalyzer()

# FastAPI application
app = FastAPI(
    title="Document Analyzer",
    description="NLP document analysis service with Neo4j integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Document Analyzer starting up")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Document Analyzer shutting down")
    if analyzer.neo4j_driver:
        analyzer.neo4j_driver.close()
    if analyzer.connection and not analyzer.connection.is_closed:
        analyzer.connection.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Neo4j connection
        with analyzer.neo4j_driver.session(database=settings.neo4j_database) as session:
            session.run("RETURN 1")

        # Check RabbitMQ
        rabbitmq_status = "connected" if (
            analyzer.connection and
            not analyzer.connection.is_closed
        ) else "disconnected"

        return {
            "status": "healthy",
            "service": settings.service_name,
            "timestamp": datetime.now().isoformat(),
            "neo4j": "connected",
            "rabbitmq": rabbitmq_status,
            "nlp_models": "loaded"
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

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_document_endpoint(request: DocumentRequest):
    """Analyze document and extract entities, relationships, and insights"""
    result = await analyzer.analyze_document(request)

    # Publish result to queue
    analyzer.publish_result(result)

    return result

@app.get("/status")
async def get_service_status():
    """Get detailed service status"""
    return {
        "service": settings.service_name,
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "neo4j_uri": settings.neo4j_uri,
            "spacy_model": settings.spacy_model,
            "max_text_length": settings.max_text_length,
            "debug_mode": settings.debug
        },
        "nlp_models": {
            "spacy": settings.spacy_model,
            "sentence_transformer": settings.sentence_transformer_model,
            "flair_ner": "flair/ner-english-large"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "analyzer:app",
        host="0.0.0.0",
        port=8081,
        reload=settings.debug,
        workers=2 if not settings.debug else 1
    )