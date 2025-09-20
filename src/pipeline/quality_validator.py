#!/usr/bin/env python3
"""
Quality Validator - Comprehensive Quality Assessment for Context Compression
Part of the BEV OSINT Context Compression Engine
"""

import asyncio
import json
import logging
import math
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import tiktoken
from collections import Counter, defaultdict
import re

# ML and NLP imports
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import torch
import torch.nn.functional as F

# Additional analysis imports
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import textstat
import difflib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityMetric:
    """Individual quality metric result"""
    name: str
    score: float
    threshold: float
    passes: bool
    description: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Comprehensive validation result"""
    overall_score: float
    passes_quality_thresholds: bool
    individual_metrics: List[QualityMetric]

    # Core quality dimensions
    information_preservation: float
    semantic_similarity: float
    structural_coherence: float
    linguistic_quality: float

    # Additional metrics
    readability_score: float
    compression_efficiency: float
    reconstruction_accuracy: float

    # Recommendations and insights
    recommendations: List[str]
    quality_insights: Dict[str, Any]

    # Processing metadata
    validation_time: float
    content_analysis: Dict[str, Any]

@dataclass
class ValidationConfig:
    """Configuration for quality validation"""
    # Core thresholds
    min_information_preservation: float = 0.95
    min_semantic_similarity: float = 0.90
    min_structural_coherence: float = 0.85
    min_linguistic_quality: float = 0.80

    # Detailed thresholds
    min_bleu_score: float = 0.8
    min_rouge_score: float = 0.8
    max_readability_degradation: float = 0.2
    min_reconstruction_accuracy: float = 0.95

    # Analysis settings
    enable_deep_analysis: bool = True
    enable_linguistic_analysis: bool = True
    enable_semantic_analysis: bool = True
    enable_structural_analysis: bool = True

    # Performance settings
    batch_size: int = 32
    max_content_length: int = 10000

class QualityValidator:
    """
    Comprehensive quality validator for context compression results
    Evaluates multiple dimensions of quality and provides detailed feedback
    """

    def __init__(self, config: ValidationConfig):
        self.config = config

        # Initialize NLP models
        self._initialize_models()

        # Initialize tokenizer
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Quality metrics registry
        self.metrics_registry = {
            'information_preservation': self._evaluate_information_preservation,
            'semantic_similarity': self._evaluate_semantic_similarity,
            'structural_coherence': self._evaluate_structural_coherence,
            'linguistic_quality': self._evaluate_linguistic_quality,
            'bleu_score': self._evaluate_bleu_score,
            'rouge_score': self._evaluate_rouge_score,
            'readability': self._evaluate_readability,
            'reconstruction_accuracy': self._evaluate_reconstruction_accuracy,
            'compression_efficiency': self._evaluate_compression_efficiency
        }

    def _initialize_models(self):
        """Initialize NLP models and components"""
        try:
            # Sentence transformer for semantic analysis
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

            # SpaCy for linguistic analysis
            self.nlp = spacy.load("en_core_web_sm")

            # TF-IDF for content analysis
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )

            # ROUGE scorer for summarization quality
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )

            # Download required NLTK data
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
            except:
                pass

            logger.info("Quality validation models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize quality validation models: {e}")
            raise

    async def validate_compression(self,
                                 original_content: List[str],
                                 compressed_content: List[str],
                                 compression_metadata: Dict[str, Any] = None) -> ValidationResult:
        """
        Comprehensive validation of compression results

        Args:
            original_content: Original content before compression
            compressed_content: Content after compression
            compression_metadata: Metadata from compression process

        Returns:
            ValidationResult with detailed quality assessment
        """
        start_time = time.time()

        logger.info(f"Starting quality validation: {len(original_content)} -> {len(compressed_content)} items")

        try:
            # Content preprocessing
            original_text = self._preprocess_content(original_content)
            compressed_text = self._preprocess_content(compressed_content)

            # Content analysis
            content_analysis = await self._analyze_content(original_text, compressed_text)

            # Evaluate all quality metrics
            metric_results = []

            for metric_name, metric_func in self.metrics_registry.items():
                try:
                    metric_result = await metric_func(
                        original_text, compressed_text, compression_metadata, content_analysis
                    )
                    metric_results.append(metric_result)
                except Exception as e:
                    logger.warning(f"Failed to evaluate metric {metric_name}: {e}")
                    # Create a failed metric
                    metric_results.append(QualityMetric(
                        name=metric_name,
                        score=0.0,
                        threshold=0.0,
                        passes=False,
                        description=f"Metric evaluation failed: {e}"
                    ))

            # Calculate overall scores and assessment
            overall_assessment = await self._calculate_overall_assessment(metric_results)

            # Generate recommendations
            recommendations = await self._generate_recommendations(metric_results, content_analysis)

            # Create validation result
            validation_time = time.time() - start_time

            result = ValidationResult(
                overall_score=overall_assessment['overall_score'],
                passes_quality_thresholds=overall_assessment['passes_thresholds'],
                individual_metrics=metric_results,
                information_preservation=self._get_metric_score(metric_results, 'information_preservation'),
                semantic_similarity=self._get_metric_score(metric_results, 'semantic_similarity'),
                structural_coherence=self._get_metric_score(metric_results, 'structural_coherence'),
                linguistic_quality=self._get_metric_score(metric_results, 'linguistic_quality'),
                readability_score=self._get_metric_score(metric_results, 'readability'),
                compression_efficiency=self._get_metric_score(metric_results, 'compression_efficiency'),
                reconstruction_accuracy=self._get_metric_score(metric_results, 'reconstruction_accuracy'),
                recommendations=recommendations,
                quality_insights=await self._generate_quality_insights(metric_results, content_analysis),
                validation_time=validation_time,
                content_analysis=content_analysis
            )

            logger.info(f"Quality validation completed: {result.overall_score:.3f} overall score, "
                       f"{'PASS' if result.passes_quality_thresholds else 'FAIL'}")

            return result

        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            raise

    def _preprocess_content(self, content: List[str]) -> str:
        """Preprocess content for analysis"""
        if not content:
            return ""

        # Join all content into single text
        combined_text = '\n'.join(str(item) for item in content if item)

        # Basic cleanup
        combined_text = re.sub(r'\s+', ' ', combined_text)
        combined_text = combined_text.strip()

        # Truncate if too long
        if len(combined_text) > self.config.max_content_length:
            combined_text = combined_text[:self.config.max_content_length] + "..."

        return combined_text

    async def _analyze_content(self, original: str, compressed: str) -> Dict[str, Any]:
        """Analyze content characteristics for validation"""
        analysis = {}

        # Basic statistics
        analysis['original_length'] = len(original)
        analysis['compressed_length'] = len(compressed)
        analysis['length_ratio'] = len(compressed) / max(1, len(original))

        # Token analysis
        original_tokens = self.encoding.encode(original)
        compressed_tokens = self.encoding.encode(compressed)
        analysis['original_token_count'] = len(original_tokens)
        analysis['compressed_token_count'] = len(compressed_tokens)
        analysis['token_ratio'] = len(compressed_tokens) / max(1, len(original_tokens))

        # Word analysis
        original_words = original.split()
        compressed_words = compressed.split()
        analysis['original_word_count'] = len(original_words)
        analysis['compressed_word_count'] = len(compressed_words)
        analysis['word_ratio'] = len(compressed_words) / max(1, len(original_words))

        # Vocabulary analysis
        original_vocab = set(original_words)
        compressed_vocab = set(compressed_words)
        analysis['vocabulary_overlap'] = len(original_vocab.intersection(compressed_vocab))
        analysis['vocabulary_preservation'] = analysis['vocabulary_overlap'] / max(1, len(original_vocab))

        # Sentence analysis
        original_sentences = len(re.findall(r'[.!?]+', original))
        compressed_sentences = len(re.findall(r'[.!?]+', compressed))
        analysis['original_sentence_count'] = original_sentences
        analysis['compressed_sentence_count'] = compressed_sentences
        analysis['sentence_ratio'] = compressed_sentences / max(1, original_sentences)

        return analysis

    async def _evaluate_information_preservation(self,
                                               original: str,
                                               compressed: str,
                                               metadata: Dict[str, Any],
                                               analysis: Dict[str, Any]) -> QualityMetric:
        """Evaluate information preservation quality"""

        # Token-based preservation
        token_preservation = analysis['token_ratio']

        # Vocabulary preservation
        vocab_preservation = analysis['vocabulary_preservation']

        # Content structure preservation
        structure_preservation = min(1.0, analysis['sentence_ratio'])

        # Weighted average
        info_preservation = (
            token_preservation * 0.4 +
            vocab_preservation * 0.4 +
            structure_preservation * 0.2
        )

        passes = info_preservation >= self.config.min_information_preservation

        return QualityMetric(
            name="information_preservation",
            score=info_preservation,
            threshold=self.config.min_information_preservation,
            passes=passes,
            description="Measures how much original information is preserved",
            details={
                'token_preservation': token_preservation,
                'vocabulary_preservation': vocab_preservation,
                'structure_preservation': structure_preservation
            }
        )

    async def _evaluate_semantic_similarity(self,
                                          original: str,
                                          compressed: str,
                                          metadata: Dict[str, Any],
                                          analysis: Dict[str, Any]) -> QualityMetric:
        """Evaluate semantic similarity between original and compressed content"""

        if not original or not compressed:
            similarity = 0.0 if original != compressed else 1.0
        else:
            try:
                # Generate embeddings
                embeddings = self.semantic_model.encode([original, compressed])

                # Calculate cosine similarity
                similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
                similarity = float(similarity_matrix[0][0])

            except Exception as e:
                logger.warning(f"Semantic similarity calculation failed: {e}")
                # Fallback to word overlap
                original_words = set(original.lower().split())
                compressed_words = set(compressed.lower().split())

                if not original_words:
                    similarity = 1.0 if not compressed_words else 0.0
                else:
                    intersection = len(original_words.intersection(compressed_words))
                    union = len(original_words.union(compressed_words))
                    similarity = intersection / max(1, union)

        passes = similarity >= self.config.min_semantic_similarity

        return QualityMetric(
            name="semantic_similarity",
            score=similarity,
            threshold=self.config.min_semantic_similarity,
            passes=passes,
            description="Measures semantic similarity between original and compressed content",
            details={'method': 'sentence_transformers'}
        )

    async def _evaluate_structural_coherence(self,
                                           original: str,
                                           compressed: str,
                                           metadata: Dict[str, Any],
                                           analysis: Dict[str, Any]) -> QualityMetric:
        """Evaluate structural coherence of compressed content"""

        # Sentence structure coherence
        sentence_coherence = min(1.0, analysis['sentence_ratio'] * 1.2)

        # Paragraph structure (if applicable)
        original_paragraphs = len(original.split('\n\n'))
        compressed_paragraphs = len(compressed.split('\n\n'))
        paragraph_coherence = min(1.0, compressed_paragraphs / max(1, original_paragraphs))

        # Logical flow preservation (simplified)
        logical_flow = await self._assess_logical_flow(original, compressed)

        # Combined coherence score
        coherence = (
            sentence_coherence * 0.4 +
            paragraph_coherence * 0.3 +
            logical_flow * 0.3
        )

        passes = coherence >= self.config.min_structural_coherence

        return QualityMetric(
            name="structural_coherence",
            score=coherence,
            threshold=self.config.min_structural_coherence,
            passes=passes,
            description="Measures structural and logical coherence of compressed content",
            details={
                'sentence_coherence': sentence_coherence,
                'paragraph_coherence': paragraph_coherence,
                'logical_flow': logical_flow
            }
        )

    async def _assess_logical_flow(self, original: str, compressed: str) -> float:
        """Assess logical flow preservation"""
        try:
            # Simple approach: check if key transition words are preserved
            transition_words = {
                'however', 'therefore', 'moreover', 'furthermore', 'consequently',
                'nevertheless', 'meanwhile', 'subsequently', 'additionally', 'similarly'
            }

            original_transitions = set()
            compressed_transitions = set()

            for word in transition_words:
                if word in original.lower():
                    original_transitions.add(word)
                if word in compressed.lower():
                    compressed_transitions.add(word)

            if not original_transitions:
                return 1.0  # No transitions to preserve

            preserved_ratio = len(compressed_transitions.intersection(original_transitions)) / len(original_transitions)
            return preserved_ratio

        except Exception:
            return 0.8  # Default reasonable score

    async def _evaluate_linguistic_quality(self,
                                         original: str,
                                         compressed: str,
                                         metadata: Dict[str, Any],
                                         analysis: Dict[str, Any]) -> QualityMetric:
        """Evaluate linguistic quality of compressed content"""

        # Grammar and syntax quality
        grammar_score = await self._assess_grammar_quality(compressed)

        # Readability preservation
        readability_score = await self._assess_readability_preservation(original, compressed)

        # Vocabulary richness
        vocab_richness = await self._assess_vocabulary_richness(compressed)

        # Combined linguistic quality
        linguistic_quality = (
            grammar_score * 0.4 +
            readability_score * 0.4 +
            vocab_richness * 0.2
        )

        passes = linguistic_quality >= self.config.min_linguistic_quality

        return QualityMetric(
            name="linguistic_quality",
            score=linguistic_quality,
            threshold=self.config.min_linguistic_quality,
            passes=passes,
            description="Measures overall linguistic quality of compressed content",
            details={
                'grammar_score': grammar_score,
                'readability_score': readability_score,
                'vocabulary_richness': vocab_richness
            }
        )

    async def _assess_grammar_quality(self, text: str) -> float:
        """Assess grammar quality using SpaCy"""
        try:
            if not text:
                return 0.0

            doc = self.nlp(text[:1000])  # Limit length for performance

            # Count various linguistic features
            total_tokens = len(doc)
            if total_tokens == 0:
                return 0.0

            # Check for basic grammatical structures
            has_subjects = sum(1 for token in doc if token.dep_ == "nsubj")
            has_objects = sum(1 for token in doc if token.dep_ in ["dobj", "iobj"])
            has_verbs = sum(1 for token in doc if token.pos_ == "VERB")

            # Grammar quality heuristic
            structure_score = min(1.0, (has_subjects + has_objects + has_verbs) / max(1, total_tokens * 0.3))

            return structure_score

        except Exception:
            return 0.7  # Default reasonable score

    async def _assess_readability_preservation(self, original: str, compressed: str) -> float:
        """Assess readability preservation"""
        try:
            if not original or not compressed:
                return 0.0

            # Use textstat for readability metrics
            original_ease = textstat.flesch_reading_ease(original)
            compressed_ease = textstat.flesch_reading_ease(compressed)

            # Calculate preservation ratio
            if original_ease <= 0:
                return 1.0 if compressed_ease <= 0 else 0.5

            readability_ratio = compressed_ease / original_ease

            # Penalize significant degradation
            if readability_ratio < (1 - self.config.max_readability_degradation):
                return readability_ratio / (1 - self.config.max_readability_degradation)

            return min(1.0, readability_ratio)

        except Exception:
            return 0.8  # Default reasonable score

    async def _assess_vocabulary_richness(self, text: str) -> float:
        """Assess vocabulary richness"""
        try:
            if not text:
                return 0.0

            words = text.lower().split()
            if len(words) < 10:
                return 1.0  # Too short to assess

            unique_words = len(set(words))
            total_words = len(words)

            # Type-token ratio (vocabulary richness)
            ttr = unique_words / total_words

            # Normalize to 0-1 scale (typical TTR ranges from 0.3 to 0.8)
            normalized_ttr = min(1.0, max(0.0, (ttr - 0.3) / 0.5))

            return normalized_ttr

        except Exception:
            return 0.6  # Default reasonable score

    async def _evaluate_bleu_score(self,
                                 original: str,
                                 compressed: str,
                                 metadata: Dict[str, Any],
                                 analysis: Dict[str, Any]) -> QualityMetric:
        """Evaluate BLEU score for compression quality"""

        try:
            if not original or not compressed:
                bleu = 0.0
            else:
                # Tokenize
                original_tokens = original.split()
                compressed_tokens = compressed.split()

                # Calculate BLEU score
                bleu = sentence_bleu([original_tokens], compressed_tokens)

        except Exception as e:
            logger.warning(f"BLEU score calculation failed: {e}")
            bleu = 0.0

        passes = bleu >= self.config.min_bleu_score

        return QualityMetric(
            name="bleu_score",
            score=bleu,
            threshold=self.config.min_bleu_score,
            passes=passes,
            description="BLEU score measuring n-gram overlap quality",
            details={'method': 'sentence_bleu'}
        )

    async def _evaluate_rouge_score(self,
                                  original: str,
                                  compressed: str,
                                  metadata: Dict[str, Any],
                                  analysis: Dict[str, Any]) -> QualityMetric:
        """Evaluate ROUGE score for compression quality"""

        try:
            if not original or not compressed:
                rouge = 0.0
            else:
                scores = self.rouge_scorer.score(original, compressed)
                # Use ROUGE-L F1 score
                rouge = scores['rougeL'].fmeasure

        except Exception as e:
            logger.warning(f"ROUGE score calculation failed: {e}")
            rouge = 0.0

        passes = rouge >= self.config.min_rouge_score

        return QualityMetric(
            name="rouge_score",
            score=rouge,
            threshold=self.config.min_rouge_score,
            passes=passes,
            description="ROUGE-L score measuring longest common subsequence",
            details={'method': 'rouge_l_f1'}
        )

    async def _evaluate_readability(self,
                                  original: str,
                                  compressed: str,
                                  metadata: Dict[str, Any],
                                  analysis: Dict[str, Any]) -> QualityMetric:
        """Evaluate readability preservation"""

        readability_score = await self._assess_readability_preservation(original, compressed)
        threshold = 1 - self.config.max_readability_degradation
        passes = readability_score >= threshold

        return QualityMetric(
            name="readability",
            score=readability_score,
            threshold=threshold,
            passes=passes,
            description="Readability preservation score",
            details={'max_degradation': self.config.max_readability_degradation}
        )

    async def _evaluate_reconstruction_accuracy(self,
                                              original: str,
                                              compressed: str,
                                              metadata: Dict[str, Any],
                                              analysis: Dict[str, Any]) -> QualityMetric:
        """Evaluate reconstruction accuracy"""

        # Use edit distance for accuracy
        try:
            if not original and not compressed:
                accuracy = 1.0
            elif not original or not compressed:
                accuracy = 0.0
            else:
                # Calculate normalized edit distance
                edit_ratio = difflib.SequenceMatcher(None, original, compressed).ratio()
                accuracy = edit_ratio

        except Exception:
            accuracy = analysis['vocabulary_preservation']  # Fallback

        passes = accuracy >= self.config.min_reconstruction_accuracy

        return QualityMetric(
            name="reconstruction_accuracy",
            score=accuracy,
            threshold=self.config.min_reconstruction_accuracy,
            passes=passes,
            description="Accuracy of content reconstruction",
            details={'method': 'sequence_matcher'}
        )

    async def _evaluate_compression_efficiency(self,
                                             original: str,
                                             compressed: str,
                                             metadata: Dict[str, Any],
                                             analysis: Dict[str, Any]) -> QualityMetric:
        """Evaluate compression efficiency"""

        # Calculate compression ratio
        compression_ratio = 1 - analysis['length_ratio']

        # Efficiency considers both compression and quality preservation
        # Higher compression with maintained quality = higher efficiency
        quality_estimate = (
            analysis['vocabulary_preservation'] * 0.5 +
            min(1.0, analysis['sentence_ratio']) * 0.3 +
            min(1.0, analysis['token_ratio'] * 1.2) * 0.2
        )

        # Efficiency is compression achieved while maintaining quality
        efficiency = compression_ratio * quality_estimate

        # No fixed threshold for efficiency - it's informational
        passes = True

        return QualityMetric(
            name="compression_efficiency",
            score=efficiency,
            threshold=0.0,  # No threshold
            passes=passes,
            description="Efficiency of compression (compression ratio * quality preservation)",
            details={
                'compression_ratio': compression_ratio,
                'quality_estimate': quality_estimate
            }
        )

    def _get_metric_score(self, metrics: List[QualityMetric], metric_name: str) -> float:
        """Get score for a specific metric"""
        for metric in metrics:
            if metric.name == metric_name:
                return metric.score
        return 0.0

    async def _calculate_overall_assessment(self, metrics: List[QualityMetric]) -> Dict[str, Any]:
        """Calculate overall quality assessment"""

        # Core metrics with weights
        core_metrics_weights = {
            'information_preservation': 0.3,
            'semantic_similarity': 0.25,
            'structural_coherence': 0.2,
            'linguistic_quality': 0.15,
            'reconstruction_accuracy': 0.1
        }

        # Calculate weighted average
        weighted_score = 0.0
        total_weight = 0.0

        for metric in metrics:
            if metric.name in core_metrics_weights:
                weight = core_metrics_weights[metric.name]
                weighted_score += metric.score * weight
                total_weight += weight

        overall_score = weighted_score / max(total_weight, 1.0)

        # Check if all core thresholds are met
        core_metrics_pass = all(
            metric.passes for metric in metrics
            if metric.name in core_metrics_weights
        )

        return {
            'overall_score': overall_score,
            'passes_thresholds': core_metrics_pass
        }

    async def _generate_recommendations(self,
                                      metrics: List[QualityMetric],
                                      analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality assessment"""

        recommendations = []

        # Check individual metrics for recommendations
        for metric in metrics:
            if not metric.passes:
                if metric.name == 'information_preservation':
                    recommendations.append(
                        "Reduce compression ratio to preserve more information"
                    )
                elif metric.name == 'semantic_similarity':
                    recommendations.append(
                        "Improve semantic similarity by adjusting deduplication threshold"
                    )
                elif metric.name == 'structural_coherence':
                    recommendations.append(
                        "Preserve document structure and logical flow"
                    )
                elif metric.name == 'linguistic_quality':
                    recommendations.append(
                        "Improve linguistic quality through better text processing"
                    )
                elif metric.name == 'bleu_score':
                    recommendations.append(
                        "Improve n-gram preservation for better BLEU score"
                    )
                elif metric.name == 'rouge_score':
                    recommendations.append(
                        "Preserve longer common subsequences for better ROUGE score"
                    )
                elif metric.name == 'reconstruction_accuracy':
                    recommendations.append(
                        "Improve reconstruction accuracy by reducing lossy compression"
                    )

        # General recommendations based on analysis
        if analysis['token_ratio'] < 0.7:
            recommendations.append(
                "Consider less aggressive compression to maintain content quality"
            )

        if analysis['vocabulary_preservation'] < 0.8:
            recommendations.append(
                "Preserve more vocabulary diversity to maintain semantic richness"
            )

        return recommendations

    async def _generate_quality_insights(self,
                                       metrics: List[QualityMetric],
                                       analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality insights and analysis"""

        insights = {
            'compression_summary': {
                'token_reduction': 1 - analysis['token_ratio'],
                'word_reduction': 1 - analysis['word_ratio'],
                'length_reduction': 1 - analysis['length_ratio'],
                'vocabulary_preserved': analysis['vocabulary_preservation']
            },
            'quality_distribution': {},
            'strength_areas': [],
            'improvement_areas': [],
            'overall_assessment': ""
        }

        # Quality distribution
        for metric in metrics:
            insights['quality_distribution'][metric.name] = {
                'score': metric.score,
                'threshold': metric.threshold,
                'passes': metric.passes
            }

        # Identify strengths and weaknesses
        for metric in metrics:
            if metric.passes and metric.score > 0.9:
                insights['strength_areas'].append(metric.name)
            elif not metric.passes or metric.score < 0.7:
                insights['improvement_areas'].append(metric.name)

        # Overall assessment
        overall_score = sum(m.score for m in metrics) / len(metrics)
        if overall_score >= 0.9:
            assessment = "Excellent compression quality"
        elif overall_score >= 0.8:
            assessment = "Good compression quality"
        elif overall_score >= 0.7:
            assessment = "Acceptable compression quality"
        else:
            assessment = "Poor compression quality - significant improvements needed"

        insights['overall_assessment'] = assessment

        return insights

    async def benchmark_quality_metrics(self,
                                      test_cases: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Benchmark quality metrics on test cases"""

        benchmark_results = {
            'test_count': len(test_cases),
            'metric_performance': {},
            'processing_times': {},
            'reliability_scores': {}
        }

        for metric_name, metric_func in self.metrics_registry.items():
            metric_scores = []
            processing_times = []

            for original, compressed in test_cases:
                start_time = time.time()

                try:
                    analysis = await self._analyze_content(original, compressed)
                    metric_result = await metric_func(original, compressed, {}, analysis)
                    metric_scores.append(metric_result.score)

                except Exception:
                    metric_scores.append(0.0)

                processing_times.append(time.time() - start_time)

            benchmark_results['metric_performance'][metric_name] = {
                'mean_score': np.mean(metric_scores),
                'std_score': np.std(metric_scores),
                'min_score': np.min(metric_scores),
                'max_score': np.max(metric_scores)
            }

            benchmark_results['processing_times'][metric_name] = {
                'mean_time': np.mean(processing_times),
                'total_time': np.sum(processing_times)
            }

            # Reliability as inverse of coefficient of variation
            cv = np.std(metric_scores) / max(np.mean(metric_scores), 0.01)
            benchmark_results['reliability_scores'][metric_name] = max(0.0, 1.0 - cv)

        return benchmark_results