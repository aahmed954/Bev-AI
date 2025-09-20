"""
Extended Reasoning Pipeline for BEV OSINT Framework
Handles 100K+ token contexts with intelligent chunking and memory management
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import json
import aiohttp
import numpy as np
from collections import defaultdict
import networkx as nx

# Custom imports
from .research_workflow import ResearchWorkflowEngine
from .counterfactual_analyzer import CounterfactualAnalyzer
from .knowledge_synthesizer import KnowledgeSynthesizer
from .integration_client import create_integration_client, IntegrationClient

logger = logging.getLogger(__name__)

class ReasoningPhase(Enum):
    """Reasoning pipeline phases"""
    EXPLORATION = "exploration"
    DEEP_DIVING = "deep_diving"
    CROSS_VERIFICATION = "cross_verification"
    SYNTHESIS = "synthesis"
    COUNTERFACTUAL = "counterfactual"

class ConfidenceLevel(Enum):
    """Confidence levels for reasoning outputs"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95

@dataclass
class ReasoningContext:
    """Context container for reasoning operations"""
    context_id: str
    raw_content: str
    tokens: int
    chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_started: datetime = field(default_factory=datetime.now)
    current_phase: ReasoningPhase = ReasoningPhase.EXPLORATION
    phase_results: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    uncertainty_factors: List[str] = field(default_factory=list)
    knowledge_graph: Optional[nx.DiGraph] = None

@dataclass
class ReasoningResult:
    """Result container for reasoning operations"""
    context_id: str
    final_synthesis: str
    confidence_score: float
    uncertainty_factors: List[str]
    phase_outputs: Dict[str, Any]
    knowledge_graph: nx.DiGraph
    processing_time: float
    token_efficiency: float
    verification_results: Dict[str, Any]
    counterfactual_analysis: Dict[str, Any]
    recommendations: List[str]

class ExtendedReasoningPipeline:
    """
    Extended reasoning pipeline for complex OSINT analysis
    Handles large contexts with multi-phase verification
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compression_endpoint = config.get('compression_endpoint', 'http://172.30.0.43:8000')
        self.vector_db_endpoint = config.get('vector_db_endpoint', 'http://172.30.0.44:8000')
        self.max_tokens = config.get('max_tokens', 100000)
        self.chunk_size = config.get('chunk_size', 8000)
        self.overlap_ratio = config.get('overlap_ratio', 0.1)
        self.min_confidence = config.get('min_confidence', 0.6)
        self.max_processing_time = config.get('max_processing_time', 600)  # 10 minutes

        # Initialize specialized components
        self.workflow_engine = ResearchWorkflowEngine(config)
        self.counterfactual_analyzer = CounterfactualAnalyzer(config)
        self.knowledge_synthesizer = KnowledgeSynthesizer(config)

        # Initialize integration client
        self.integration_client: Optional[IntegrationClient] = None

        # Memory management
        self.active_contexts: Dict[str, ReasoningContext] = {}
        self.context_cache: Dict[str, Any] = {}
        self.processing_metrics: Dict[str, List[float]] = defaultdict(list)

        # Phase configurations
        self.phase_configs = {
            ReasoningPhase.EXPLORATION: {
                'timeout': 120,
                'min_confidence': 0.4,
                'breadth_factor': 0.8
            },
            ReasoningPhase.DEEP_DIVING: {
                'timeout': 180,
                'min_confidence': 0.6,
                'depth_factor': 0.9
            },
            ReasoningPhase.CROSS_VERIFICATION: {
                'timeout': 150,
                'min_confidence': 0.7,
                'verification_threshold': 0.8
            },
            ReasoningPhase.SYNTHESIS: {
                'timeout': 120,
                'min_confidence': 0.8,
                'integration_threshold': 0.85
            },
            ReasoningPhase.COUNTERFACTUAL: {
                'timeout': 90,
                'min_confidence': 0.6,
                'hypothesis_count': 5
            }
        }

    async def process_context(self, content: str, context_id: str = None,
                            metadata: Dict[str, Any] = None) -> ReasoningResult:
        """
        Process large context through extended reasoning pipeline

        Args:
            content: Raw text content to analyze
            context_id: Unique identifier for this context
            metadata: Additional context metadata

        Returns:
            ReasoningResult with comprehensive analysis
        """
        start_time = time.time()

        if context_id is None:
            context_id = f"ctx_{int(time.time())}"

        try:
            # Initialize integration client if not already done
            if self.integration_client is None:
                self.integration_client = await create_integration_client(
                    compression_endpoint=self.compression_endpoint,
                    vector_db_endpoint=self.vector_db_endpoint
                )

            # Initialize reasoning context
            reasoning_context = await self._initialize_context(
                content, context_id, metadata or {}
            )

            # Execute 5-phase reasoning workflow
            await self._execute_reasoning_phases(reasoning_context)

            # Generate final synthesis
            final_result = await self._generate_final_result(
                reasoning_context, time.time() - start_time
            )

            # Cleanup
            await self._cleanup_context(context_id)

            return final_result

        except Exception as e:
            logger.error(f"Error processing context {context_id}: {str(e)}")
            await self._cleanup_context(context_id)
            raise

    async def _initialize_context(self, content: str, context_id: str,
                                metadata: Dict[str, Any]) -> ReasoningContext:
        """Initialize reasoning context with intelligent chunking"""
        logger.info(f"Initializing context {context_id}")

        # Estimate token count
        token_count = len(content.split()) * 1.3  # Rough estimation

        # Create context object
        context = ReasoningContext(
            context_id=context_id,
            raw_content=content,
            tokens=int(token_count),
            metadata=metadata
        )

        # Intelligent chunking for large contexts
        if token_count > self.chunk_size:
            if self.integration_client:
                context.chunks = await self.integration_client.intelligent_chunk(
                    content, self.chunk_size, self.overlap_ratio
                )
            else:
                context.chunks = await self._intelligent_chunking(content)
        else:
            context.chunks = [content]

        # Initialize knowledge graph
        context.knowledge_graph = nx.DiGraph()

        # Store in active contexts
        self.active_contexts[context_id] = context

        logger.info(f"Context {context_id} initialized with {len(context.chunks)} chunks")
        return context

    async def _intelligent_chunking(self, content: str) -> List[str]:
        """
        Intelligent content chunking with semantic preservation

        Args:
            content: Raw content to chunk

        Returns:
            List of semantically coherent chunks
        """
        try:
            if self.integration_client:
                return await self.integration_client.intelligent_chunk(
                    content, self.chunk_size, self.overlap_ratio
                )
            else:
                # Use compression service for intelligent chunking
                async with aiohttp.ClientSession() as session:
                    chunk_payload = {
                        'content': content,
                        'chunk_size': self.chunk_size,
                        'overlap_ratio': self.overlap_ratio,
                        'preserve_semantics': True
                    }

                    async with session.post(
                        f"{self.compression_endpoint}/chunk",
                        json=chunk_payload
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result.get('chunks', [])
                        else:
                            logger.warning("Compression service unavailable, using fallback chunking")
                            return self._fallback_chunking(content)

        except Exception as e:
            logger.warning(f"Error in intelligent chunking: {str(e)}, using fallback")
            return self._fallback_chunking(content)

    def _fallback_chunking(self, content: str) -> List[str]:
        """Fallback chunking strategy"""
        words = content.split()
        chunk_words = self.chunk_size // 1.3  # Rough word count
        overlap_words = int(chunk_words * self.overlap_ratio)

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + int(chunk_words), len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)

            if end >= len(words):
                break

            start = end - overlap_words

        return chunks

    async def _execute_reasoning_phases(self, context: ReasoningContext):
        """Execute all 5 reasoning phases"""
        phases = [
            ReasoningPhase.EXPLORATION,
            ReasoningPhase.DEEP_DIVING,
            ReasoningPhase.CROSS_VERIFICATION,
            ReasoningPhase.SYNTHESIS,
            ReasoningPhase.COUNTERFACTUAL
        ]

        for phase in phases:
            logger.info(f"Executing phase: {phase.value}")
            context.current_phase = phase

            phase_start = time.time()
            try:
                await self._execute_phase(context, phase)
                phase_time = time.time() - phase_start
                self.processing_metrics[f"{phase.value}_time"].append(phase_time)

            except Exception as e:
                logger.error(f"Error in phase {phase.value}: {str(e)}")
                context.uncertainty_factors.append(f"Phase {phase.value} incomplete: {str(e)}")

    async def _execute_phase(self, context: ReasoningContext, phase: ReasoningPhase):
        """Execute specific reasoning phase"""
        phase_config = self.phase_configs[phase]
        timeout = phase_config['timeout']

        try:
            if phase == ReasoningPhase.EXPLORATION:
                result = await asyncio.wait_for(
                    self._exploration_phase(context), timeout=timeout
                )
            elif phase == ReasoningPhase.DEEP_DIVING:
                result = await asyncio.wait_for(
                    self._deep_diving_phase(context), timeout=timeout
                )
            elif phase == ReasoningPhase.CROSS_VERIFICATION:
                result = await asyncio.wait_for(
                    self._cross_verification_phase(context), timeout=timeout
                )
            elif phase == ReasoningPhase.SYNTHESIS:
                result = await asyncio.wait_for(
                    self._synthesis_phase(context), timeout=timeout
                )
            elif phase == ReasoningPhase.COUNTERFACTUAL:
                result = await asyncio.wait_for(
                    self._counterfactual_phase(context), timeout=timeout
                )
            else:
                raise ValueError(f"Unknown phase: {phase}")

            context.phase_results[phase.value] = result

        except asyncio.TimeoutError:
            logger.warning(f"Phase {phase.value} timed out after {timeout}s")
            context.uncertainty_factors.append(f"Phase {phase.value} timeout")

    async def _exploration_phase(self, context: ReasoningContext) -> Dict[str, Any]:
        """
        Exploration phase: Initial analysis and information mapping
        """
        logger.info("Starting exploration phase")

        # Use workflow engine for initial exploration
        exploration_results = await self.workflow_engine.explore_context(
            context.chunks, context.metadata
        )

        # Extract key entities and relationships
        entities = exploration_results.get('entities', [])
        relationships = exploration_results.get('relationships', [])
        topics = exploration_results.get('topics', [])

        # Build initial knowledge graph
        for entity in entities:
            context.knowledge_graph.add_node(
                entity['name'],
                type=entity.get('type', 'unknown'),
                confidence=entity.get('confidence', 0.5),
                attributes=entity.get('attributes', {})
            )

        for rel in relationships:
            context.knowledge_graph.add_edge(
                rel['source'],
                rel['target'],
                relation=rel.get('relation', 'related'),
                confidence=rel.get('confidence', 0.5)
            )

        # Calculate exploration confidence
        exploration_confidence = self._calculate_exploration_confidence(
            entities, relationships, topics
        )
        context.confidence_scores['exploration'] = exploration_confidence

        return {
            'entities': entities,
            'relationships': relationships,
            'topics': topics,
            'confidence': exploration_confidence,
            'graph_metrics': {
                'nodes': context.knowledge_graph.number_of_nodes(),
                'edges': context.knowledge_graph.number_of_edges(),
                'density': nx.density(context.knowledge_graph)
            }
        }

    async def _deep_diving_phase(self, context: ReasoningContext) -> Dict[str, Any]:
        """
        Deep diving phase: Detailed analysis of key findings
        """
        logger.info("Starting deep diving phase")

        # Get exploration results
        exploration = context.phase_results.get('exploration', {})
        key_entities = exploration.get('entities', [])[:10]  # Focus on top entities

        # Deep analysis of key entities
        deep_analysis = await self.workflow_engine.deep_analyze(
            key_entities, context.chunks
        )

        # Enhanced knowledge graph with detailed attributes
        for entity_analysis in deep_analysis.get('detailed_entities', []):
            entity_name = entity_analysis['name']
            if context.knowledge_graph.has_node(entity_name):
                # Update node with detailed information
                context.knowledge_graph.nodes[entity_name].update({
                    'detailed_attributes': entity_analysis.get('attributes', {}),
                    'evidence': entity_analysis.get('evidence', []),
                    'significance': entity_analysis.get('significance', 0.5)
                })

        # Calculate deep diving confidence
        deep_confidence = self._calculate_deep_diving_confidence(deep_analysis)
        context.confidence_scores['deep_diving'] = deep_confidence

        return {
            'detailed_entities': deep_analysis.get('detailed_entities', []),
            'significant_patterns': deep_analysis.get('patterns', []),
            'evidence_strength': deep_analysis.get('evidence_strength', {}),
            'confidence': deep_confidence
        }

    async def _cross_verification_phase(self, context: ReasoningContext) -> Dict[str, Any]:
        """
        Cross-verification phase: Multi-source validation
        """
        logger.info("Starting cross-verification phase")

        # Get previous phase results
        exploration = context.phase_results.get('exploration', {})
        deep_diving = context.phase_results.get('deep_diving', {})

        # Cross-verify findings across chunks
        verification_results = await self.workflow_engine.cross_verify(
            exploration, deep_diving, context.chunks
        )

        # Update confidence scores based on verification
        verified_entities = verification_results.get('verified_entities', [])
        conflicting_info = verification_results.get('conflicts', [])

        # Calculate verification confidence
        verification_confidence = self._calculate_verification_confidence(
            verified_entities, conflicting_info
        )
        context.confidence_scores['cross_verification'] = verification_confidence

        # Add uncertainty factors for conflicts
        for conflict in conflicting_info:
            context.uncertainty_factors.append(f"Conflict: {conflict['description']}")

        return {
            'verified_entities': verified_entities,
            'conflicts': conflicting_info,
            'consistency_score': verification_results.get('consistency_score', 0.5),
            'confidence': verification_confidence
        }

    async def _synthesis_phase(self, context: ReasoningContext) -> Dict[str, Any]:
        """
        Synthesis phase: Integrate findings into coherent analysis
        """
        logger.info("Starting synthesis phase")

        # Use knowledge synthesizer
        synthesis_result = await self.knowledge_synthesizer.synthesize(
            context.knowledge_graph,
            context.phase_results,
            context.confidence_scores
        )

        # Calculate synthesis confidence
        synthesis_confidence = self._calculate_synthesis_confidence(synthesis_result)
        context.confidence_scores['synthesis'] = synthesis_confidence

        return synthesis_result

    async def _counterfactual_phase(self, context: ReasoningContext) -> Dict[str, Any]:
        """
        Counterfactual phase: Alternative hypothesis testing
        """
        logger.info("Starting counterfactual phase")

        # Use counterfactual analyzer
        counterfactual_result = await self.counterfactual_analyzer.analyze(
            context.knowledge_graph,
            context.phase_results,
            hypothesis_count=self.phase_configs[ReasoningPhase.COUNTERFACTUAL]['hypothesis_count']
        )

        # Calculate counterfactual confidence
        counterfactual_confidence = self._calculate_counterfactual_confidence(
            counterfactual_result
        )
        context.confidence_scores['counterfactual'] = counterfactual_confidence

        return counterfactual_result

    def _calculate_exploration_confidence(self, entities: List[Dict],
                                        relationships: List[Dict],
                                        topics: List[Dict]) -> float:
        """Calculate confidence score for exploration phase"""
        entity_confidence = np.mean([e.get('confidence', 0.5) for e in entities]) if entities else 0.3
        relationship_confidence = np.mean([r.get('confidence', 0.5) for r in relationships]) if relationships else 0.3
        topic_confidence = np.mean([t.get('confidence', 0.5) for t in topics]) if topics else 0.3

        # Weighted average with coverage factor
        coverage_factor = min(1.0, (len(entities) + len(relationships)) / 20)
        base_confidence = (entity_confidence * 0.4 + relationship_confidence * 0.4 + topic_confidence * 0.2)

        return base_confidence * coverage_factor

    def _calculate_deep_diving_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for deep diving phase"""
        detailed_entities = analysis.get('detailed_entities', [])
        if not detailed_entities:
            return 0.3

        evidence_scores = []
        for entity in detailed_entities:
            evidence = entity.get('evidence', [])
            significance = entity.get('significance', 0.5)
            evidence_strength = len(evidence) / 10  # Normalize to [0,1]
            evidence_scores.append(min(1.0, evidence_strength * significance))

        return np.mean(evidence_scores) if evidence_scores else 0.3

    def _calculate_verification_confidence(self, verified_entities: List[Dict],
                                         conflicts: List[Dict]) -> float:
        """Calculate confidence score for verification phase"""
        if not verified_entities:
            return 0.3

        verification_rate = len([e for e in verified_entities if e.get('verified', False)]) / len(verified_entities)
        conflict_penalty = min(0.3, len(conflicts) * 0.05)

        return max(0.2, verification_rate - conflict_penalty)

    def _calculate_synthesis_confidence(self, synthesis: Dict[str, Any]) -> float:
        """Calculate confidence score for synthesis phase"""
        integration_score = synthesis.get('integration_score', 0.5)
        coherence_score = synthesis.get('coherence_score', 0.5)
        completeness_score = synthesis.get('completeness_score', 0.5)

        return (integration_score * 0.4 + coherence_score * 0.4 + completeness_score * 0.2)

    def _calculate_counterfactual_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score for counterfactual phase"""
        hypotheses = analysis.get('alternative_hypotheses', [])
        if not hypotheses:
            return 0.3

        hypothesis_strengths = [h.get('strength', 0.5) for h in hypotheses]
        avg_strength = np.mean(hypothesis_strengths)

        # Higher average strength of alternatives reduces confidence in original
        return max(0.2, 0.8 - (avg_strength * 0.3))

    async def _generate_final_result(self, context: ReasoningContext,
                                   processing_time: float) -> ReasoningResult:
        """Generate final reasoning result"""
        logger.info("Generating final reasoning result")

        # Calculate overall confidence
        phase_confidences = list(context.confidence_scores.values())
        overall_confidence = np.mean(phase_confidences) if phase_confidences else 0.3

        # Get synthesis result
        synthesis = context.phase_results.get('synthesis', {})
        final_synthesis = synthesis.get('integrated_analysis', "Analysis incomplete")

        # Calculate token efficiency
        token_efficiency = self.max_tokens / max(context.tokens, 1)

        # Get verification and counterfactual results
        verification_results = context.phase_results.get('cross_verification', {})
        counterfactual_analysis = context.phase_results.get('counterfactual', {})

        # Generate recommendations
        recommendations = self._generate_recommendations(context, overall_confidence)

        # Store knowledge graph and results if integration client available
        if self.integration_client:
            try:
                # Store knowledge graph
                entities_for_storage = []
                for node in context.knowledge_graph.nodes():
                    node_data = context.knowledge_graph.nodes[node]
                    entities_for_storage.append({
                        'name': node,
                        'attributes': node_data
                    })

                relationships_for_storage = []
                for edge in context.knowledge_graph.edges():
                    edge_data = context.knowledge_graph.edges[edge]
                    relationships_for_storage.append({
                        'source': edge[0],
                        'target': edge[1],
                        'attributes': edge_data
                    })

                await self.integration_client.store_knowledge_graph(
                    entities_for_storage,
                    relationships_for_storage,
                    context.context_id
                )

                logger.info(f"Stored knowledge graph for context {context.context_id}")

            except Exception as e:
                logger.warning(f"Failed to store knowledge graph: {str(e)}")

        return ReasoningResult(
            context_id=context.context_id,
            final_synthesis=final_synthesis,
            confidence_score=overall_confidence,
            uncertainty_factors=context.uncertainty_factors,
            phase_outputs=context.phase_results,
            knowledge_graph=context.knowledge_graph,
            processing_time=processing_time,
            token_efficiency=token_efficiency,
            verification_results=verification_results,
            counterfactual_analysis=counterfactual_analysis,
            recommendations=recommendations
        )

    def _generate_recommendations(self, context: ReasoningContext,
                                confidence: float) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        if confidence < 0.6:
            recommendations.append("Additional verification required due to low confidence")

        if len(context.uncertainty_factors) > 3:
            recommendations.append("Multiple uncertainty factors detected - consider additional sources")

        if context.knowledge_graph.number_of_nodes() < 5:
            recommendations.append("Limited entity detection - may need broader context")

        if 'conflicts' in context.phase_results.get('cross_verification', {}):
            conflicts = context.phase_results['cross_verification']['conflicts']
            if len(conflicts) > 2:
                recommendations.append("Significant conflicts detected - manual review recommended")

        counterfactual = context.phase_results.get('counterfactual', {})
        if counterfactual.get('strong_alternatives', 0) > 2:
            recommendations.append("Strong alternative hypotheses exist - consider multiple scenarios")

        return recommendations

    async def _cleanup_context(self, context_id: str):
        """Clean up processing context"""
        if context_id in self.active_contexts:
            del self.active_contexts[context_id]
        logger.info(f"Context {context_id} cleaned up")

    async def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing performance metrics"""
        metrics = {}

        for metric_name, values in self.processing_metrics.items():
            if values:
                metrics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }

        metrics['active_contexts'] = len(self.active_contexts)

        return metrics

    async def health_check(self) -> Dict[str, Any]:
        """Health check for reasoning pipeline"""
        health = {
            'status': 'healthy',
            'active_contexts': len(self.active_contexts),
            'memory_usage': 'unknown',  # Would implement actual memory tracking
            'services': {}
        }

        # Use integration client for health checks if available
        if self.integration_client:
            try:
                service_health = await self.integration_client.check_service_health()
                health['services'].update(service_health)
            except Exception as e:
                logger.warning(f"Integration client health check failed: {str(e)}")
                health['services']['integration_client'] = False
        else:
            # Fallback to direct health checks
            # Check compression service
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.compression_endpoint}/health") as response:
                        health['services']['compression'] = response.status == 200
            except:
                health['services']['compression'] = False

            # Check vector database
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.vector_db_endpoint}/health") as response:
                        health['services']['vector_db'] = response.status == 200
            except:
                health['services']['vector_db'] = False

        if not all(health['services'].values()):
            health['status'] = 'degraded'

        return health