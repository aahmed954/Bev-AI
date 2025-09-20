"""
Test Suite for Extended Reasoning Pipeline
Validates the comprehensive reasoning system
"""

import asyncio
import pytest
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock

# Test the extended reasoning pipeline
from src.agents.extended_reasoning import (
    ExtendedReasoningPipeline,
    ReasoningPhase,
    ReasoningContext,
    ReasoningResult
)
from src.agents.research_workflow import ResearchWorkflowEngine
from src.agents.counterfactual_analyzer import CounterfactualAnalyzer
from src.agents.knowledge_synthesizer import KnowledgeSynthesizer

class TestExtendedReasoningPipeline:
    """Test suite for Extended Reasoning Pipeline"""

    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            'compression_endpoint': 'http://test-compression:8000',
            'vector_db_endpoint': 'http://test-vectordb:8000',
            'max_tokens': 50000,
            'chunk_size': 4000,
            'overlap_ratio': 0.1,
            'min_confidence': 0.6,
            'max_processing_time': 300,
            'entity_confidence_threshold': 0.6,
            'relationship_confidence_threshold': 0.5,
            'pattern_significance_threshold': 0.7
        }

    @pytest.fixture
    def pipeline(self, config):
        """Extended reasoning pipeline instance"""
        return ExtendedReasoningPipeline(config)

    @pytest.fixture
    def sample_content(self):
        """Sample content for testing"""
        return """
        John Smith works at Acme Corporation as the Chief Technology Officer.
        He previously worked at TechStart Inc. where he led the development team.
        Acme Corporation is located in San Francisco and specializes in AI technology.
        John received an email from jane.doe@competitor.com regarding a potential collaboration.
        The company has been involved in several acquisitions over the past year.
        Financial records show transfers of $500,000 to offshore accounts.
        Meeting logs indicate discussions about Project Aurora with external partners.
        IP address 192.168.1.100 accessed the company's internal systems multiple times.
        """

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, config):
        """Test pipeline initialization"""
        pipeline = ExtendedReasoningPipeline(config)

        assert pipeline.max_tokens == 50000
        assert pipeline.chunk_size == 4000
        assert pipeline.overlap_ratio == 0.1
        assert pipeline.min_confidence == 0.6

        # Check component initialization
        assert isinstance(pipeline.workflow_engine, ResearchWorkflowEngine)
        assert isinstance(pipeline.counterfactual_analyzer, CounterfactualAnalyzer)
        assert isinstance(pipeline.knowledge_synthesizer, KnowledgeSynthesizer)

    @pytest.mark.asyncio
    async def test_context_initialization(self, pipeline, sample_content):
        """Test context initialization"""
        context = await pipeline._initialize_context(
            sample_content, "test_ctx", {"source": "test"}
        )

        assert context.context_id == "test_ctx"
        assert context.raw_content == sample_content
        assert context.tokens > 0
        assert len(context.chunks) > 0
        assert context.current_phase == ReasoningPhase.EXPLORATION
        assert context.knowledge_graph is not None

    @pytest.mark.asyncio
    async def test_intelligent_chunking_fallback(self, pipeline, sample_content):
        """Test fallback chunking when compression service unavailable"""
        chunks = await pipeline._intelligent_chunking(sample_content)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    @pytest.mark.asyncio
    async def test_fallback_chunking(self, pipeline):
        """Test fallback chunking strategy"""
        # Test with long content
        long_content = "word " * 10000  # 10000 words
        chunks = pipeline._fallback_chunking(long_content)

        assert isinstance(chunks, list)
        assert len(chunks) > 1  # Should be split into multiple chunks

        # Test overlap
        if len(chunks) > 1:
            # Check for overlap between consecutive chunks
            first_chunk_words = chunks[0].split()
            second_chunk_words = chunks[1].split()
            overlap_size = int(len(first_chunk_words) * pipeline.overlap_ratio)
            assert overlap_size > 0

    @pytest.mark.asyncio
    async def test_exploration_phase(self, pipeline, sample_content):
        """Test exploration phase execution"""
        context = await pipeline._initialize_context(sample_content, "test_ctx", {})

        # Mock workflow engine
        mock_result = {
            'entities': [
                {'name': 'John Smith', 'type': 'person', 'confidence': 0.9},
                {'name': 'Acme Corporation', 'type': 'organization', 'confidence': 0.8}
            ],
            'relationships': [
                {'source': 'John Smith', 'target': 'Acme Corporation', 'relation': 'works_at', 'confidence': 0.85}
            ],
            'topics': [
                {'name': 'business', 'confidence': 0.7}
            ]
        }

        with patch.object(pipeline.workflow_engine, 'explore_context', return_value=mock_result):
            result = await pipeline._exploration_phase(context)

            assert 'entities' in result
            assert 'relationships' in result
            assert 'topics' in result
            assert 'confidence' in result
            assert len(result['entities']) == 2
            assert len(result['relationships']) == 1

    @pytest.mark.asyncio
    async def test_deep_diving_phase(self, pipeline, sample_content):
        """Test deep diving phase execution"""
        context = await pipeline._initialize_context(sample_content, "test_ctx", {})

        # Set up exploration results
        context.phase_results['exploration'] = {
            'entities': [
                {'name': 'John Smith', 'type': 'person', 'confidence': 0.9},
                {'name': 'Acme Corporation', 'type': 'organization', 'confidence': 0.8}
            ]
        }

        # Mock workflow engine
        mock_result = {
            'detailed_entities': [
                {
                    'name': 'John Smith',
                    'attributes': {'role': 'CTO'},
                    'evidence': ['Chief Technology Officer'],
                    'significance': 0.8
                }
            ],
            'patterns': [
                {'type': 'organizational', 'description': 'Leadership structure'}
            ],
            'evidence_strength': {'strong': 1, 'medium': 0, 'weak': 0}
        }

        with patch.object(pipeline.workflow_engine, 'deep_analyze', return_value=mock_result):
            result = await pipeline._deep_diving_phase(context)

            assert 'detailed_entities' in result
            assert 'significant_patterns' in result
            assert 'confidence' in result

    @pytest.mark.asyncio
    async def test_confidence_calculations(self, pipeline):
        """Test confidence score calculations"""
        # Test exploration confidence
        entities = [
            {'confidence': 0.9},
            {'confidence': 0.8},
            {'confidence': 0.7}
        ]
        relationships = [
            {'confidence': 0.85}
        ]
        topics = [
            {'confidence': 0.75}
        ]

        confidence = pipeline._calculate_exploration_confidence(entities, relationships, topics)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high with good inputs

    @pytest.mark.asyncio
    async def test_counterfactual_phase(self, pipeline, sample_content):
        """Test counterfactual analysis phase"""
        context = await pipeline._initialize_context(sample_content, "test_ctx", {})

        # Set up previous phase results
        context.phase_results = {
            'exploration': {'entities': [], 'relationships': []},
            'deep_diving': {'detailed_entities': []},
            'cross_verification': {'verified_entities': []}
        }

        # Mock counterfactual analyzer
        mock_result = {
            'alternative_hypotheses': [
                {
                    'hypothesis_id': 'test_hyp_1',
                    'description': 'Alternative scenario',
                    'strength': 0.7
                }
            ],
            'strong_alternatives': 1
        }

        with patch.object(pipeline.counterfactual_analyzer, 'analyze', return_value=mock_result):
            result = await pipeline._counterfactual_phase(context)

            assert 'alternative_hypotheses' in result
            assert len(result['alternative_hypotheses']) == 1

    @pytest.mark.asyncio
    async def test_synthesis_phase(self, pipeline, sample_content):
        """Test knowledge synthesis phase"""
        context = await pipeline._initialize_context(sample_content, "test_ctx", {})

        # Set up knowledge graph
        context.knowledge_graph.add_node('John Smith', type='person')
        context.knowledge_graph.add_node('Acme Corporation', type='organization')
        context.knowledge_graph.add_edge('John Smith', 'Acme Corporation', relation='works_at')

        # Set up previous phase results
        context.phase_results = {
            'exploration': {'entities': []},
            'deep_diving': {'detailed_entities': []},
            'cross_verification': {'verified_entities': []}
        }
        context.confidence_scores = {
            'exploration': 0.8,
            'deep_diving': 0.7
        }

        # Mock knowledge synthesizer
        mock_result = {
            'integrated_analysis': 'Test synthesis',
            'key_insights': [],
            'integration_score': 0.8,
            'coherence_score': 0.7,
            'completeness_score': 0.75
        }

        with patch.object(pipeline.knowledge_synthesizer, 'synthesize', return_value=mock_result):
            result = await pipeline._synthesis_phase(context)

            assert 'integrated_analysis' in result
            assert 'key_insights' in result
            assert 'integration_score' in result

    @pytest.mark.asyncio
    async def test_generate_recommendations(self, pipeline, sample_content):
        """Test recommendation generation"""
        context = await pipeline._initialize_context(sample_content, "test_ctx", {})

        # Test with low confidence
        recommendations = pipeline._generate_recommendations(context, 0.4)
        assert any("verification required" in rec.lower() for rec in recommendations)

        # Test with many uncertainty factors
        context.uncertainty_factors = ['factor1', 'factor2', 'factor3', 'factor4']
        recommendations = pipeline._generate_recommendations(context, 0.8)
        assert any("uncertainty factors" in rec.lower() for rec in recommendations)

    @pytest.mark.asyncio
    async def test_processing_metrics(self, pipeline):
        """Test processing metrics collection"""
        # Add some mock metrics
        pipeline.processing_metrics['exploration_time'].extend([1.2, 1.5, 1.1])
        pipeline.processing_metrics['deep_diving_time'].extend([2.1, 2.3])

        metrics = await pipeline.get_processing_metrics()

        assert 'exploration_time' in metrics
        assert 'deep_diving_time' in metrics
        assert 'active_contexts' in metrics

        # Check metric statistics
        exploration_metrics = metrics['exploration_time']
        assert 'mean' in exploration_metrics
        assert 'std' in exploration_metrics
        assert 'count' in exploration_metrics
        assert exploration_metrics['count'] == 3

    @pytest.mark.asyncio
    async def test_health_check(self, pipeline):
        """Test health check functionality"""
        # Mock successful health checks
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            health = await pipeline.health_check()

            assert health['status'] in ['healthy', 'degraded']
            assert 'active_contexts' in health
            assert 'services' in health

    @pytest.mark.asyncio
    async def test_context_cleanup(self, pipeline, sample_content):
        """Test context cleanup"""
        context_id = "test_cleanup_ctx"

        # Initialize context
        await pipeline._initialize_context(sample_content, context_id, {})
        assert context_id in pipeline.active_contexts

        # Cleanup
        await pipeline._cleanup_context(context_id)
        assert context_id not in pipeline.active_contexts

    @pytest.mark.asyncio
    @patch('src.agents.extended_reasoning.create_integration_client')
    async def test_full_processing_pipeline(self, mock_integration_client, pipeline, sample_content):
        """Test complete processing pipeline end-to-end"""
        # Mock integration client
        mock_client = AsyncMock()
        mock_client.intelligent_chunk.return_value = [sample_content]  # Single chunk
        mock_client.store_knowledge_graph.return_value = True
        mock_integration_client.return_value = mock_client

        # Mock all sub-components
        with patch.object(pipeline.workflow_engine, 'explore_context') as mock_explore, \
             patch.object(pipeline.workflow_engine, 'deep_analyze') as mock_deep, \
             patch.object(pipeline.workflow_engine, 'cross_verify') as mock_verify, \
             patch.object(pipeline.knowledge_synthesizer, 'synthesize') as mock_synthesize, \
             patch.object(pipeline.counterfactual_analyzer, 'analyze') as mock_counterfactual:

            # Set up mock returns
            mock_explore.return_value = {
                'entities': [{'name': 'John Smith', 'type': 'person', 'confidence': 0.9}],
                'relationships': [],
                'topics': [],
                'patterns': []
            }

            mock_deep.return_value = {
                'detailed_entities': [{'name': 'John Smith', 'significance': 0.8}],
                'patterns': [],
                'evidence_strength': {}
            }

            mock_verify.return_value = {
                'verified_entities': [{'entity_name': 'John Smith', 'verified': True}],
                'conflicts': [],
                'consistency_score': 0.8
            }

            mock_synthesize.return_value = {
                'integrated_analysis': 'Test synthesis',
                'key_insights': [],
                'integration_score': 0.8,
                'coherence_score': 0.7,
                'completeness_score': 0.75,
                'network_analysis': {},
                'knowledge_clusters': [],
                'causal_chains': []
            }

            mock_counterfactual.return_value = {
                'alternative_hypotheses': [],
                'strong_alternatives': 0
            }

            # Run full pipeline
            result = await pipeline.process_context(
                content=sample_content,
                context_id="test_full_pipeline",
                metadata={'test': True}
            )

            # Verify result structure
            assert isinstance(result, ReasoningResult)
            assert result.context_id == "test_full_pipeline"
            assert result.final_synthesis == 'Test synthesis'
            assert 0.0 <= result.confidence_score <= 1.0
            assert result.processing_time > 0
            assert isinstance(result.phase_outputs, dict)
            assert len(result.phase_outputs) == 5  # All 5 phases

class TestResearchWorkflowIntegration:
    """Test integration with Research Workflow Engine"""

    @pytest.fixture
    def workflow_config(self):
        return {
            'entity_confidence_threshold': 0.6,
            'relationship_confidence_threshold': 0.5,
            'pattern_significance_threshold': 0.7
        }

    @pytest.fixture
    def workflow_engine(self, workflow_config):
        return ResearchWorkflowEngine(workflow_config)

    @pytest.mark.asyncio
    async def test_workflow_health_check(self, workflow_engine):
        """Test workflow engine health check"""
        health = await workflow_engine.health_check()

        assert 'status' in health
        assert health['status'] == 'healthy'

class TestIntegrationErrorHandling:
    """Test error handling in integration scenarios"""

    @pytest.mark.asyncio
    async def test_compression_service_failure(self, sample_content):
        """Test handling of compression service failures"""
        config = {
            'compression_endpoint': 'http://invalid-endpoint:8000',
            'vector_db_endpoint': 'http://test-vectordb:8000',
            'max_tokens': 50000
        }

        pipeline = ExtendedReasoningPipeline(config)

        # Should fallback gracefully
        chunks = await pipeline._intelligent_chunking(sample_content)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_phase_timeout_handling(self):
        """Test handling of phase timeouts"""
        config = {
            'compression_endpoint': 'http://test-compression:8000',
            'vector_db_endpoint': 'http://test-vectordb:8000',
            'max_tokens': 50000
        }

        pipeline = ExtendedReasoningPipeline(config)

        # Set very short timeout for testing
        pipeline.phase_configs[ReasoningPhase.EXPLORATION]['timeout'] = 0.001

        context = await pipeline._initialize_context("test content", "test_ctx", {})

        # Should handle timeout gracefully
        await pipeline._execute_phase(context, ReasoningPhase.EXPLORATION)

        # Should have uncertainty factor about timeout
        assert any("timeout" in factor.lower() for factor in context.uncertainty_factors)

if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v"])