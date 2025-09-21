"""
Knowledge Synthesizer for BEV OSINT Framework
Implements graph-based reasoning and knowledge integration
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import json
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import heapq

logger = logging.getLogger(__name__)

class SynthesisStrategy(Enum):
    """Knowledge synthesis strategies"""
    HIERARCHICAL_INTEGRATION = "hierarchical_integration"
    GRAPH_CENTRALITY = "graph_centrality"
    EVIDENCE_CONVERGENCE = "evidence_convergence"
    PATTERN_REINFORCEMENT = "pattern_reinforcement"
    SEMANTIC_CLUSTERING = "semantic_clustering"

class InsightType(Enum):
    """Types of synthesized insights"""
    CENTRAL_ENTITY = "central_entity"
    KEY_RELATIONSHIP = "key_relationship"
    EMERGENT_PATTERN = "emergent_pattern"
    CAUSAL_CHAIN = "causal_chain"
    NETWORK_STRUCTURE = "network_structure"
    EVIDENCE_CLUSTER = "evidence_cluster"

@dataclass
class SynthesizedInsight:
    """Container for synthesized insights"""
    insight_id: str
    insight_type: InsightType
    description: str
    confidence: float
    supporting_evidence: List[str]
    contributing_phases: List[str]
    affected_entities: List[str]
    network_metrics: Dict[str, float]
    synthesis_path: List[str]  # How this insight was derived
    validation_score: float
    impact_assessment: Dict[str, float]
    recommendations: List[str]

@dataclass
class KnowledgeCluster:
    """Cluster of related knowledge elements"""
    cluster_id: str
    entities: List[str]
    relationships: List[Tuple[str, str, str]]
    patterns: List[str]
    evidence_strength: float
    internal_consistency: float
    cluster_significance: float
    representative_elements: List[str]

@dataclass
class CausalChain:
    """Causal reasoning chain"""
    chain_id: str
    chain_elements: List[str]
    causal_links: List[Tuple[str, str, float]]  # (cause, effect, strength)
    chain_confidence: float
    supporting_evidence: List[str]
    alternative_explanations: List[str]
    validation_requirements: List[str]

@dataclass
class SynthesisResult:
    """Complete synthesis result"""
    synthesis_id: str
    integrated_analysis: str
    key_insights: List[SynthesizedInsight]
    knowledge_clusters: List[KnowledgeCluster]
    causal_chains: List[CausalChain]
    network_analysis: Dict[str, Any]
    integration_score: float
    coherence_score: float
    completeness_score: float
    evidence_summary: List[str]
    confidence_assessment: Dict[str, float]
    synthesis_metrics: Dict[str, float]

class KnowledgeSynthesizer:
    """
    Advanced knowledge synthesis system with graph-based reasoning
    Integrates findings from all analysis phases into coherent insights
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_cluster_size = config.get('min_cluster_size', 3)
        self.max_clusters = config.get('max_clusters', 10)
        self.centrality_threshold = config.get('centrality_threshold', 0.1)
        self.evidence_convergence_threshold = config.get('evidence_convergence_threshold', 0.7)
        self.min_causal_strength = config.get('min_causal_strength', 0.6)

        # Synthesis strategies and their weights
        self.strategy_weights = {
            SynthesisStrategy.HIERARCHICAL_INTEGRATION: 0.25,
            SynthesisStrategy.GRAPH_CENTRALITY: 0.20,
            SynthesisStrategy.EVIDENCE_CONVERGENCE: 0.20,
            SynthesisStrategy.PATTERN_REINFORCEMENT: 0.20,
            SynthesisStrategy.SEMANTIC_CLUSTERING: 0.15
        }

        # Network analysis metrics
        self.network_metrics = [
            'betweenness_centrality',
            'eigenvector_centrality',
            'pagerank',
            'clustering_coefficient',
            'degree_centrality'
        ]

        # Evidence weighting factors
        self.evidence_weights = {
            'exploration': 1.0,
            'deep_diving': 1.2,
            'cross_verification': 1.5,
            'pattern_analysis': 1.1
        }

    async def synthesize(self, knowledge_graph: nx.DiGraph,
                        phase_results: Dict[str, Any],
                        confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform comprehensive knowledge synthesis

        Args:
            knowledge_graph: Knowledge graph from reasoning phases
            phase_results: Results from all analysis phases
            confidence_scores: Confidence scores from each phase

        Returns:
            Dictionary with synthesis results
        """
        start_time = time.time()
        logger.info("Starting knowledge synthesis")

        try:
            # Initialize synthesis context
            synthesis_context = await self._initialize_synthesis_context(
                knowledge_graph, phase_results, confidence_scores
            )

            # Apply synthesis strategies
            strategy_results = await self._apply_synthesis_strategies(synthesis_context)

            # Perform graph-based analysis
            network_analysis = await self._perform_network_analysis(knowledge_graph, phase_results)

            # Generate knowledge clusters
            knowledge_clusters = await self._generate_knowledge_clusters(
                knowledge_graph, phase_results, network_analysis
            )

            # Extract causal chains
            causal_chains = await self._extract_causal_chains(
                knowledge_graph, phase_results, network_analysis
            )

            # Synthesize key insights
            key_insights = await self._synthesize_key_insights(
                strategy_results, knowledge_clusters, causal_chains, network_analysis
            )

            # Generate integrated analysis
            integrated_analysis = await self._generate_integrated_analysis(
                key_insights, knowledge_clusters, causal_chains
            )

            # Calculate synthesis metrics
            synthesis_metrics = await self._calculate_synthesis_metrics(
                key_insights, knowledge_clusters, causal_chains, confidence_scores
            )

            # Create final synthesis result
            synthesis_result = SynthesisResult(
                synthesis_id=f"synthesis_{int(time.time())}",
                integrated_analysis=integrated_analysis,
                key_insights=key_insights,
                knowledge_clusters=knowledge_clusters,
                causal_chains=causal_chains,
                network_analysis=network_analysis,
                integration_score=synthesis_metrics['integration_score'],
                coherence_score=synthesis_metrics['coherence_score'],
                completeness_score=synthesis_metrics['completeness_score'],
                evidence_summary=await self._create_evidence_summary(phase_results),
                confidence_assessment=confidence_scores,
                synthesis_metrics=synthesis_metrics
            )

            processing_time = time.time() - start_time

            return self._synthesis_result_to_dict(synthesis_result, processing_time)

        except Exception as e:
            logger.error(f"Error in knowledge synthesis: {str(e)}")
            raise

    async def _initialize_synthesis_context(self, knowledge_graph: nx.DiGraph,
                                          phase_results: Dict[str, Any],
                                          confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Initialize synthesis context with weighted elements"""
        logger.info("Initializing synthesis context")

        context = {
            'knowledge_graph': knowledge_graph,
            'weighted_entities': await self._weight_entities(phase_results, confidence_scores),
            'weighted_relationships': await self._weight_relationships(phase_results, confidence_scores),
            'weighted_patterns': await self._weight_patterns(phase_results, confidence_scores),
            'evidence_map': await self._create_evidence_map(phase_results),
            'confidence_distribution': confidence_scores,
            'phase_contributions': await self._assess_phase_contributions(phase_results)
        }

        return context

    async def _weight_entities(self, phase_results: Dict[str, Any],
                             confidence_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Weight entities based on phase contributions and confidence"""
        weighted_entities = []

        # Get entities from exploration phase
        exploration_entities = phase_results.get('exploration', {}).get('entities', [])

        for entity in exploration_entities:
            entity_weight = 0.0

            # Base weight from entity confidence
            base_confidence = entity.get('confidence', 0.5)
            entity_weight += base_confidence * 0.4

            # Weight from exploration phase
            exploration_weight = confidence_scores.get('exploration', 0.5)
            entity_weight += exploration_weight * self.evidence_weights['exploration'] * 0.2

            # Additional weight from deep diving if entity was analyzed
            deep_diving = phase_results.get('deep_diving', {})
            detailed_entities = deep_diving.get('detailed_entities', [])

            for detailed_entity in detailed_entities:
                if detailed_entity.get('name') == entity.get('name'):
                    significance = detailed_entity.get('significance', 0.5)
                    deep_weight = confidence_scores.get('deep_diving', 0.5)
                    entity_weight += significance * deep_weight * self.evidence_weights['deep_diving'] * 0.3

            # Weight from verification
            verification = phase_results.get('cross_verification', {})
            verified_entities = verification.get('verified_entities', [])

            for verified_entity in verified_entities:
                if verified_entity.get('entity_name') == entity.get('name'):
                    if verified_entity.get('verified', False):
                        verification_weight = confidence_scores.get('cross_verification', 0.5)
                        entity_weight += verification_weight * self.evidence_weights['cross_verification'] * 0.1

            weighted_entity = {
                'name': entity.get('name'),
                'type': entity.get('type'),
                'original_confidence': base_confidence,
                'synthesis_weight': min(1.0, entity_weight),
                'evidence_sources': self._get_entity_evidence_sources(entity, phase_results),
                'attributes': entity.get('attributes', {}),
                'importance_factors': {
                    'base_confidence': base_confidence,
                    'exploration_contribution': exploration_weight * 0.2,
                    'deep_analysis_contribution': 0.0,  # Updated above if found
                    'verification_contribution': 0.0    # Updated above if found
                }
            }

            weighted_entities.append(weighted_entity)

        # Sort by synthesis weight
        weighted_entities.sort(key=lambda x: x['synthesis_weight'], reverse=True)

        return weighted_entities

    async def _weight_relationships(self, phase_results: Dict[str, Any],
                                  confidence_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Weight relationships based on phase contributions and confidence"""
        weighted_relationships = []

        exploration_relationships = phase_results.get('exploration', {}).get('relationships', [])

        for relationship in exploration_relationships:
            rel_weight = 0.0

            # Base weight from relationship confidence
            base_confidence = relationship.get('confidence', 0.5)
            rel_weight += base_confidence * 0.5

            # Weight from exploration phase
            exploration_weight = confidence_scores.get('exploration', 0.5)
            rel_weight += exploration_weight * self.evidence_weights['exploration'] * 0.3

            # Weight from verification
            verification = phase_results.get('cross_verification', {})
            verification_weight = confidence_scores.get('cross_verification', 0.5)
            rel_weight += verification_weight * self.evidence_weights['cross_verification'] * 0.2

            weighted_relationship = {
                'source': relationship.get('source'),
                'target': relationship.get('target'),
                'relation': relationship.get('relation'),
                'original_confidence': base_confidence,
                'synthesis_weight': min(1.0, rel_weight),
                'evidence': relationship.get('evidence', ''),
                'context': relationship.get('context', ''),
                'strength': relationship.get('strength', 0.5)
            }

            weighted_relationships.append(weighted_relationship)

        weighted_relationships.sort(key=lambda x: x['synthesis_weight'], reverse=True)

        return weighted_relationships

    async def _weight_patterns(self, phase_results: Dict[str, Any],
                             confidence_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Weight patterns based on significance and confidence"""
        weighted_patterns = []

        exploration_patterns = phase_results.get('exploration', {}).get('patterns', [])

        for pattern in exploration_patterns:
            pattern_weight = 0.0

            # Base weight from pattern significance
            significance = pattern.get('significance', 0.5)
            pattern_weight += significance * 0.6

            # Weight from pattern analysis confidence
            pattern_confidence = pattern.get('confidence', 0.5)
            pattern_weight += pattern_confidence * 0.4

            weighted_pattern = {
                'type': pattern.get('type'),
                'description': pattern.get('description'),
                'original_significance': significance,
                'original_confidence': pattern_confidence,
                'synthesis_weight': min(1.0, pattern_weight),
                'frequency': pattern.get('frequency', 1),
                'entities_involved': pattern.get('entities_involved', []),
                'evidence': pattern.get('evidence', [])
            }

            weighted_patterns.append(weighted_pattern)

        weighted_patterns.sort(key=lambda x: x['synthesis_weight'], reverse=True)

        return weighted_patterns

    def _get_entity_evidence_sources(self, entity: Dict[str, Any],
                                   phase_results: Dict[str, Any]) -> List[str]:
        """Get evidence sources for an entity across phases"""
        sources = ['exploration']

        entity_name = entity.get('name', '')

        # Check if entity appears in deep diving
        deep_diving = phase_results.get('deep_diving', {})
        detailed_entities = deep_diving.get('detailed_entities', [])

        for detailed_entity in detailed_entities:
            if detailed_entity.get('name') == entity_name:
                sources.append('deep_diving')
                break

        # Check if entity appears in verification
        verification = phase_results.get('cross_verification', {})
        verified_entities = verification.get('verified_entities', [])

        for verified_entity in verified_entities:
            if verified_entity.get('entity_name') == entity_name:
                sources.append('cross_verification')
                break

        return sources

    async def _create_evidence_map(self, phase_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create comprehensive evidence mapping"""
        evidence_map = defaultdict(list)

        # Map evidence from each phase
        for phase_name, phase_data in phase_results.items():
            if isinstance(phase_data, dict):
                # Entities evidence
                entities = phase_data.get('entities', [])
                for entity in entities:
                    entity_name = entity.get('name', '')
                    evidence = entity.get('evidence', [])
                    if evidence:
                        evidence_map[f"entity_{entity_name}"].extend(evidence)

                # Relationships evidence
                relationships = phase_data.get('relationships', [])
                for rel in relationships:
                    rel_key = f"relationship_{rel.get('source')}_{rel.get('target')}"
                    evidence = rel.get('evidence', '')
                    if evidence:
                        evidence_map[rel_key].append(evidence)

                # Patterns evidence
                patterns = phase_data.get('patterns', [])
                for pattern in patterns:
                    pattern_key = f"pattern_{pattern.get('type')}"
                    evidence = pattern.get('evidence', [])
                    if evidence:
                        evidence_map[pattern_key].extend(evidence)

        return dict(evidence_map)

    async def _assess_phase_contributions(self, phase_results: Dict[str, Any]) -> Dict[str, float]:
        """Assess the contribution of each phase to the overall analysis"""
        contributions = {}

        for phase_name, phase_data in phase_results.items():
            if isinstance(phase_data, dict):
                contribution_score = 0.0

                # Count elements contributed by phase
                entities_count = len(phase_data.get('entities', []))
                relationships_count = len(phase_data.get('relationships', []))
                patterns_count = len(phase_data.get('patterns', []))

                # Calculate contribution based on content richness
                contribution_score += entities_count * 0.3
                contribution_score += relationships_count * 0.4
                contribution_score += patterns_count * 0.3

                # Normalize contribution score
                contributions[phase_name] = min(1.0, contribution_score / 20)

        return contributions

    async def _apply_synthesis_strategies(self, synthesis_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple synthesis strategies and combine results"""
        logger.info("Applying synthesis strategies")

        strategy_results = {}

        for strategy, weight in self.strategy_weights.items():
            try:
                if strategy == SynthesisStrategy.HIERARCHICAL_INTEGRATION:
                    result = await self._hierarchical_integration(synthesis_context)
                elif strategy == SynthesisStrategy.GRAPH_CENTRALITY:
                    result = await self._graph_centrality_analysis(synthesis_context)
                elif strategy == SynthesisStrategy.EVIDENCE_CONVERGENCE:
                    result = await self._evidence_convergence_analysis(synthesis_context)
                elif strategy == SynthesisStrategy.PATTERN_REINFORCEMENT:
                    result = await self._pattern_reinforcement_analysis(synthesis_context)
                elif strategy == SynthesisStrategy.SEMANTIC_CLUSTERING:
                    result = await self._semantic_clustering_analysis(synthesis_context)
                else:
                    continue

                strategy_results[strategy.value] = {
                    'result': result,
                    'weight': weight,
                    'confidence': result.get('confidence', 0.5)
                }

            except Exception as e:
                logger.warning(f"Error in {strategy.value}: {str(e)}")
                strategy_results[strategy.value] = {
                    'result': {'insights': [], 'confidence': 0.0},
                    'weight': weight,
                    'confidence': 0.0
                }

        return strategy_results

    async def _hierarchical_integration(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Hierarchical integration of knowledge elements"""
        insights = []

        # Create hierarchy based on entity weights
        weighted_entities = context['weighted_entities']
        top_entities = weighted_entities[:5]  # Top 5 entities

        for entity in top_entities:
            insight = {
                'type': 'hierarchical_entity',
                'description': f"Key entity: {entity['name']} (weight: {entity['synthesis_weight']:.2f})",
                'confidence': entity['synthesis_weight'],
                'evidence': entity['evidence_sources'],
                'entity': entity['name']
            }
            insights.append(insight)

        # Hierarchical relationship analysis
        weighted_relationships = context['weighted_relationships']
        top_relationships = weighted_relationships[:5]

        for rel in top_relationships:
            insight = {
                'type': 'hierarchical_relationship',
                'description': f"Key relationship: {rel['source']} {rel['relation']} {rel['target']} (weight: {rel['synthesis_weight']:.2f})",
                'confidence': rel['synthesis_weight'],
                'evidence': [rel['evidence']],
                'entities': [rel['source'], rel['target']]
            }
            insights.append(insight)

        return {
            'insights': insights,
            'confidence': np.mean([i['confidence'] for i in insights]) if insights else 0.0
        }

    async def _graph_centrality_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Graph centrality-based analysis"""
        insights = []
        knowledge_graph = context['knowledge_graph']

        if knowledge_graph.number_of_nodes() == 0:
            return {'insights': [], 'confidence': 0.0}

        # Calculate centrality metrics
        try:
            betweenness = nx.betweenness_centrality(knowledge_graph)
            eigenvector = nx.eigenvector_centrality(knowledge_graph, max_iter=1000)
            pagerank = nx.pagerank(knowledge_graph)

            # Find highly central nodes
            central_nodes = []
            for node in knowledge_graph.nodes():
                centrality_score = (
                    betweenness.get(node, 0) * 0.4 +
                    eigenvector.get(node, 0) * 0.3 +
                    pagerank.get(node, 0) * 0.3
                )

                if centrality_score > self.centrality_threshold:
                    central_nodes.append((node, centrality_score))

            # Sort by centrality score
            central_nodes.sort(key=lambda x: x[1], reverse=True)

            # Create insights for central nodes
            for node, score in central_nodes[:5]:
                insight = {
                    'type': 'central_entity',
                    'description': f"Highly central entity: {node} (centrality: {score:.3f})",
                    'confidence': min(1.0, score * 2),  # Scale centrality to confidence
                    'evidence': [f"High network centrality across multiple metrics"],
                    'entity': node,
                    'centrality_metrics': {
                        'betweenness': betweenness.get(node, 0),
                        'eigenvector': eigenvector.get(node, 0),
                        'pagerank': pagerank.get(node, 0)
                    }
                }
                insights.append(insight)

        except Exception as e:
            logger.warning(f"Error in centrality analysis: {str(e)}")

        return {
            'insights': insights,
            'confidence': np.mean([i['confidence'] for i in insights]) if insights else 0.0
        }

    async def _evidence_convergence_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evidence convergence analysis"""
        insights = []
        evidence_map = context['evidence_map']

        # Find elements with converging evidence from multiple sources
        convergent_elements = []

        for element_key, evidence_list in evidence_map.items():
            if len(evidence_list) >= 2:  # At least 2 pieces of evidence
                convergence_score = min(1.0, len(evidence_list) / 5)  # Normalize
                if convergence_score >= self.evidence_convergence_threshold:
                    convergent_elements.append((element_key, evidence_list, convergence_score))

        # Sort by convergence score
        convergent_elements.sort(key=lambda x: x[2], reverse=True)

        # Create insights for convergent elements
        for element_key, evidence_list, score in convergent_elements[:5]:
            insight = {
                'type': 'evidence_convergence',
                'description': f"Strong evidence convergence for {element_key} ({len(evidence_list)} sources)",
                'confidence': score,
                'evidence': evidence_list,
                'convergence_score': score
            }

            # Extract entity/relationship from element key
            if element_key.startswith('entity_'):
                insight['entity'] = element_key.replace('entity_', '')
            elif element_key.startswith('relationship_'):
                rel_parts = element_key.replace('relationship_', '').split('_')
                if len(rel_parts) >= 2:
                    insight['entities'] = [rel_parts[0], rel_parts[1]]

            insights.append(insight)

        return {
            'insights': insights,
            'confidence': np.mean([i['confidence'] for i in insights]) if insights else 0.0
        }

    async def _pattern_reinforcement_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pattern reinforcement analysis"""
        insights = []
        weighted_patterns = context['weighted_patterns']

        # Find patterns that reinforce each other
        pattern_groups = defaultdict(list)

        for pattern in weighted_patterns:
            pattern_type = pattern['type']
            entities_involved = pattern.get('entities_involved', [])

            # Group patterns by overlapping entities
            for group_key, group_patterns in pattern_groups.items():
                if any(entity in entities_involved for p in group_patterns for entity in p.get('entities_involved', [])):
                    pattern_groups[group_key].append(pattern)
                    break
            else:
                # Create new group
                group_key = f"group_{len(pattern_groups)}"
                pattern_groups[group_key] = [pattern]

        # Analyze pattern reinforcement
        for group_key, patterns in pattern_groups.items():
            if len(patterns) >= 2:  # At least 2 reinforcing patterns
                total_weight = sum(p['synthesis_weight'] for p in patterns)
                avg_weight = total_weight / len(patterns)

                all_entities = set()
                for pattern in patterns:
                    all_entities.update(pattern.get('entities_involved', []))

                insight = {
                    'type': 'pattern_reinforcement',
                    'description': f"Reinforcing patterns involving {len(all_entities)} entities across {len(patterns)} pattern types",
                    'confidence': min(1.0, avg_weight),
                    'evidence': [p['description'] for p in patterns],
                    'entities': list(all_entities),
                    'pattern_types': [p['type'] for p in patterns],
                    'reinforcement_strength': total_weight
                }
                insights.append(insight)

        return {
            'insights': insights,
            'confidence': np.mean([i['confidence'] for i in insights]) if insights else 0.0
        }

    async def _semantic_clustering_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Semantic clustering analysis"""
        insights = []

        # This is a simplified semantic clustering
        # In practice, would use proper NLP embeddings and clustering

        weighted_entities = context['weighted_entities']
        entity_types = defaultdict(list)

        # Group entities by type
        for entity in weighted_entities:
            entity_type = entity.get('type', 'unknown')
            entity_types[entity_type].append(entity)

        # Create insights for significant entity clusters
        for entity_type, entities in entity_types.items():
            if len(entities) >= self.min_cluster_size:
                avg_weight = np.mean([e['synthesis_weight'] for e in entities])

                insight = {
                    'type': 'semantic_cluster',
                    'description': f"Semantic cluster: {len(entities)} entities of type '{entity_type}'",
                    'confidence': min(1.0, avg_weight),
                    'evidence': [f"Entity type clustering: {entity_type}"],
                    'entities': [e['name'] for e in entities],
                    'cluster_type': entity_type,
                    'cluster_size': len(entities)
                }
                insights.append(insight)

        return {
            'insights': insights,
            'confidence': np.mean([i['confidence'] for i in insights]) if insights else 0.0
        }

    async def _perform_network_analysis(self, knowledge_graph: nx.DiGraph,
                                      phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive network analysis"""
        logger.info("Performing network analysis")

        analysis = {
            'basic_metrics': {},
            'centrality_metrics': {},
            'structural_analysis': {},
            'community_detection': {},
            'path_analysis': {}
        }

        try:
            # Basic network metrics
            analysis['basic_metrics'] = {
                'nodes': knowledge_graph.number_of_nodes(),
                'edges': knowledge_graph.number_of_edges(),
                'density': nx.density(knowledge_graph),
                'is_connected': nx.is_weakly_connected(knowledge_graph),
                'number_of_components': nx.number_weakly_connected_components(knowledge_graph)
            }

            if knowledge_graph.number_of_nodes() > 0:
                # Centrality metrics
                analysis['centrality_metrics'] = {
                    'betweenness_centrality': dict(nx.betweenness_centrality(knowledge_graph)),
                    'degree_centrality': dict(nx.degree_centrality(knowledge_graph)),
                    'in_degree_centrality': dict(nx.in_degree_centrality(knowledge_graph)),
                    'out_degree_centrality': dict(nx.out_degree_centrality(knowledge_graph))
                }

                try:
                    analysis['centrality_metrics']['eigenvector_centrality'] = dict(
                        nx.eigenvector_centrality(knowledge_graph, max_iter=1000)
                    )
                    analysis['centrality_metrics']['pagerank'] = dict(
                        nx.pagerank(knowledge_graph)
                    )
                except:
                    logger.warning("Could not compute eigenvector centrality or pagerank")

                # Structural analysis
                analysis['structural_analysis'] = {
                    'average_clustering': nx.average_clustering(knowledge_graph.to_undirected()),
                    'transitivity': nx.transitivity(knowledge_graph.to_undirected())
                }

                # Community detection using simple connected components
                components = list(nx.weakly_connected_components(knowledge_graph))
                analysis['community_detection'] = {
                    'communities': [list(component) for component in components],
                    'number_of_communities': len(components),
                    'modularity': self._calculate_modularity(knowledge_graph, components)
                }

                # Path analysis
                analysis['path_analysis'] = await self._analyze_paths(knowledge_graph)

        except Exception as e:
            logger.warning(f"Error in network analysis: {str(e)}")

        return analysis

    def _calculate_modularity(self, graph: nx.DiGraph, communities: List[Set]) -> float:
        """Calculate network modularity"""
        try:
            # Simple modularity calculation
            total_edges = graph.number_of_edges()
            if total_edges == 0:
                return 0.0

            modularity = 0.0
            for community in communities:
                internal_edges = 0
                external_edges = 0

                for node in community:
                    for neighbor in graph.neighbors(node):
                        if neighbor in community:
                            internal_edges += 1
                        else:
                            external_edges += 1

                if internal_edges + external_edges > 0:
                    modularity += internal_edges / (internal_edges + external_edges)

            return modularity / len(communities) if communities else 0.0

        except Exception as e:
            logger.warning(f"Error calculating modularity: {str(e)}")
            return 0.0

    async def _analyze_paths(self, knowledge_graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze important paths in the network"""
        path_analysis = {
            'shortest_paths': {},
            'critical_paths': [],
            'path_diversity': 0.0
        }

        try:
            nodes = list(knowledge_graph.nodes())
            if len(nodes) < 2:
                return path_analysis

            # Sample node pairs for path analysis (to avoid exponential complexity)
            sample_pairs = []
            for i, source in enumerate(nodes[:10]):  # Limit to first 10 nodes
                for target in nodes[i+1:min(i+6, len(nodes))]:  # Max 5 targets per source
                    sample_pairs.append((source, target))

            # Find shortest paths
            for source, target in sample_pairs:
                try:
                    if nx.has_path(knowledge_graph, source, target):
                        path = nx.shortest_path(knowledge_graph, source, target)
                        path_analysis['shortest_paths'][f"{source}->{target}"] = path
                except:
                    continue

            # Identify critical paths (paths through high centrality nodes)
            centrality = nx.betweenness_centrality(knowledge_graph)
            high_centrality_nodes = [node for node, cent in centrality.items() if cent > 0.1]

            for path_key, path in path_analysis['shortest_paths'].items():
                critical_nodes = [node for node in path if node in high_centrality_nodes]
                if len(critical_nodes) >= 2:
                    path_analysis['critical_paths'].append({
                        'path': path,
                        'critical_nodes': critical_nodes,
                        'importance': len(critical_nodes) / len(path)
                    })

            # Calculate path diversity
            all_paths = list(path_analysis['shortest_paths'].values())
            if all_paths:
                unique_nodes = set()
                for path in all_paths:
                    unique_nodes.update(path)
                path_analysis['path_diversity'] = len(unique_nodes) / knowledge_graph.number_of_nodes()

        except Exception as e:
            logger.warning(f"Error in path analysis: {str(e)}")

        return path_analysis

    async def _generate_knowledge_clusters(self, knowledge_graph: nx.DiGraph,
                                         phase_results: Dict[str, Any],
                                         network_analysis: Dict[str, Any]) -> List[KnowledgeCluster]:
        """Generate knowledge clusters based on network structure and content"""
        logger.info("Generating knowledge clusters")

        clusters = []

        try:
            # Use community detection from network analysis
            communities = network_analysis.get('community_detection', {}).get('communities', [])

            for i, community in enumerate(communities):
                if len(community) >= self.min_cluster_size:
                    cluster = await self._create_knowledge_cluster(
                        f"cluster_{i}",
                        list(community),
                        knowledge_graph,
                        phase_results
                    )
                    clusters.append(cluster)

            # Additional clustering based on entity types
            entity_type_clusters = await self._cluster_by_entity_types(phase_results)
            clusters.extend(entity_type_clusters)

            # Sort clusters by significance
            clusters.sort(key=lambda x: x.cluster_significance, reverse=True)

            # Limit to max clusters
            clusters = clusters[:self.max_clusters]

        except Exception as e:
            logger.warning(f"Error generating knowledge clusters: {str(e)}")

        return clusters

    async def _create_knowledge_cluster(self, cluster_id: str, entities: List[str],
                                      knowledge_graph: nx.DiGraph,
                                      phase_results: Dict[str, Any]) -> KnowledgeCluster:
        """Create a knowledge cluster from entities"""
        # Extract relationships within cluster
        cluster_relationships = []
        for source in entities:
            for target in entities:
                if knowledge_graph.has_edge(source, target):
                    edge_data = knowledge_graph[source][target]
                    relation = edge_data.get('relation', 'related')
                    cluster_relationships.append((source, target, relation))

        # Find relevant patterns
        exploration_patterns = phase_results.get('exploration', {}).get('patterns', [])
        relevant_patterns = []

        for pattern in exploration_patterns:
            pattern_entities = pattern.get('entities_involved', [])
            if any(entity in entities for entity in pattern_entities):
                relevant_patterns.append(pattern.get('type', 'unknown'))

        # Calculate cluster metrics
        evidence_strength = await self._calculate_cluster_evidence_strength(entities, phase_results)
        internal_consistency = await self._calculate_cluster_consistency(
            entities, cluster_relationships, phase_results
        )
        cluster_significance = await self._calculate_cluster_significance(
            entities, cluster_relationships, relevant_patterns
        )

        # Identify representative elements
        representative_elements = entities[:3]  # Top 3 entities as representatives

        return KnowledgeCluster(
            cluster_id=cluster_id,
            entities=entities,
            relationships=cluster_relationships,
            patterns=relevant_patterns,
            evidence_strength=evidence_strength,
            internal_consistency=internal_consistency,
            cluster_significance=cluster_significance,
            representative_elements=representative_elements
        )

    async def _cluster_by_entity_types(self, phase_results: Dict[str, Any]) -> List[KnowledgeCluster]:
        """Create clusters based on entity types"""
        type_clusters = []

        entities = phase_results.get('exploration', {}).get('entities', [])
        entities_by_type = defaultdict(list)

        for entity in entities:
            entity_type = entity.get('type', 'unknown')
            entities_by_type[entity_type].append(entity)

        for entity_type, type_entities in entities_by_type.items():
            if len(type_entities) >= self.min_cluster_size:
                entity_names = [e['name'] for e in type_entities]

                cluster = KnowledgeCluster(
                    cluster_id=f"type_cluster_{entity_type}",
                    entities=entity_names,
                    relationships=[],  # Would need to extract from graph
                    patterns=[],
                    evidence_strength=np.mean([e.get('confidence', 0.5) for e in type_entities]),
                    internal_consistency=0.8,  # High for type-based clustering
                    cluster_significance=len(type_entities) / 10,  # Normalize by size
                    representative_elements=entity_names[:3]
                )

                type_clusters.append(cluster)

        return type_clusters

    async def _calculate_cluster_evidence_strength(self, entities: List[str],
                                                 phase_results: Dict[str, Any]) -> float:
        """Calculate evidence strength for cluster"""
        total_strength = 0.0
        entity_count = 0

        exploration_entities = phase_results.get('exploration', {}).get('entities', [])

        for entity_name in entities:
            for entity in exploration_entities:
                if entity.get('name') == entity_name:
                    strength = entity.get('confidence', 0.5)
                    total_strength += strength
                    entity_count += 1
                    break

        return total_strength / max(entity_count, 1)

    async def _calculate_cluster_consistency(self, entities: List[str],
                                           relationships: List[Tuple[str, str, str]],
                                           phase_results: Dict[str, Any]) -> float:
        """Calculate internal consistency of cluster"""
        # Simple consistency based on relationship density
        if len(entities) <= 1:
            return 1.0

        max_possible_relationships = len(entities) * (len(entities) - 1)
        actual_relationships = len(relationships)

        relationship_density = actual_relationships / max(max_possible_relationships, 1)

        # Higher density indicates higher consistency
        return min(1.0, relationship_density * 2)

    async def _calculate_cluster_significance(self, entities: List[str],
                                            relationships: List[Tuple[str, str, str]],
                                            patterns: List[str]) -> float:
        """Calculate cluster significance"""
        # Combine multiple factors
        size_factor = min(1.0, len(entities) / 10)
        relationship_factor = min(1.0, len(relationships) / 5)
        pattern_factor = min(1.0, len(patterns) / 3)

        significance = (size_factor * 0.4 + relationship_factor * 0.4 + pattern_factor * 0.2)

        return significance

    async def _extract_causal_chains(self, knowledge_graph: nx.DiGraph,
                                   phase_results: Dict[str, Any],
                                   network_analysis: Dict[str, Any]) -> List[CausalChain]:
        """Extract causal reasoning chains"""
        logger.info("Extracting causal chains")

        causal_chains = []

        try:
            # Look for causal relationship patterns
            causal_relations = ['caused', 'led_to', 'resulted_in', 'triggered', 'owns', 'controls']

            # Find potential causal paths
            potential_chains = []

            for node in knowledge_graph.nodes():
                # Find outgoing causal relationships
                for neighbor in knowledge_graph.neighbors(node):
                    edge_data = knowledge_graph[node][neighbor]
                    relation = edge_data.get('relation', '')

                    if any(causal_rel in relation.lower() for causal_rel in causal_relations):
                        # Try to extend the chain
                        chain = await self._build_causal_chain(
                            node, neighbor, knowledge_graph, causal_relations
                        )
                        if len(chain) >= 2:
                            potential_chains.append(chain)

            # Filter and validate chains
            for i, chain in enumerate(potential_chains):
                if len(chain) >= 2:
                    causal_chain = await self._create_causal_chain(
                        f"chain_{i}",
                        chain,
                        knowledge_graph,
                        phase_results
                    )

                    if causal_chain.chain_confidence >= self.min_causal_strength:
                        causal_chains.append(causal_chain)

            # Sort by confidence
            causal_chains.sort(key=lambda x: x.chain_confidence, reverse=True)

        except Exception as e:
            logger.warning(f"Error extracting causal chains: {str(e)}")

        return causal_chains

    async def _build_causal_chain(self, start_node: str, current_node: str,
                                knowledge_graph: nx.DiGraph,
                                causal_relations: List[str],
                                visited: Set[str] = None,
                                max_depth: int = 5) -> List[str]:
        """Build causal chain starting from a node"""
        if visited is None:
            visited = set()

        if current_node in visited or len(visited) >= max_depth:
            return [start_node, current_node] if start_node != current_node else [start_node]

        visited.add(current_node)
        chain = [start_node] if start_node not in visited else []
        chain.append(current_node)

        # Try to extend the chain
        for neighbor in knowledge_graph.neighbors(current_node):
            if neighbor not in visited:
                edge_data = knowledge_graph[current_node][neighbor]
                relation = edge_data.get('relation', '')

                if any(causal_rel in relation.lower() for causal_rel in causal_relations):
                    extended_chain = await self._build_causal_chain(
                        start_node, neighbor, knowledge_graph, causal_relations, visited.copy(), max_depth
                    )
                    if len(extended_chain) > len(chain):
                        return extended_chain

        return chain

    async def _create_causal_chain(self, chain_id: str, chain_elements: List[str],
                                 knowledge_graph: nx.DiGraph,
                                 phase_results: Dict[str, Any]) -> CausalChain:
        """Create a causal chain object"""
        causal_links = []
        supporting_evidence = []

        # Extract causal links
        for i in range(len(chain_elements) - 1):
            source = chain_elements[i]
            target = chain_elements[i + 1]

            if knowledge_graph.has_edge(source, target):
                edge_data = knowledge_graph[source][target]
                confidence = edge_data.get('confidence', 0.5)
                evidence = edge_data.get('evidence', '')

                causal_links.append((source, target, confidence))
                if evidence:
                    supporting_evidence.append(evidence)

        # Calculate chain confidence
        if causal_links:
            link_confidences = [link[2] for link in causal_links]
            chain_confidence = np.mean(link_confidences)
        else:
            chain_confidence = 0.0

        # Generate alternative explanations (simplified)
        alternative_explanations = [
            "Alternative causal paths may exist",
            "Correlation does not imply causation",
            "Missing intermediate factors may influence the chain"
        ]

        # Validation requirements
        validation_requirements = [
            "Temporal sequence verification",
            "Mechanism identification",
            "Alternative explanation elimination"
        ]

        return CausalChain(
            chain_id=chain_id,
            chain_elements=chain_elements,
            causal_links=causal_links,
            chain_confidence=chain_confidence,
            supporting_evidence=supporting_evidence,
            alternative_explanations=alternative_explanations,
            validation_requirements=validation_requirements
        )

    async def _synthesize_key_insights(self, strategy_results: Dict[str, Any],
                                     knowledge_clusters: List[KnowledgeCluster],
                                     causal_chains: List[CausalChain],
                                     network_analysis: Dict[str, Any]) -> List[SynthesizedInsight]:
        """Synthesize key insights from all analysis components"""
        logger.info("Synthesizing key insights")

        key_insights = []

        # Extract insights from strategy results
        for strategy_name, strategy_data in strategy_results.items():
            strategy_insights = strategy_data.get('result', {}).get('insights', [])

            for insight_data in strategy_insights:
                insight = await self._create_synthesized_insight(
                    insight_data, strategy_name, strategy_data['weight']
                )
                key_insights.append(insight)

        # Extract insights from knowledge clusters
        for cluster in knowledge_clusters:
            if cluster.cluster_significance > 0.7:
                insight = SynthesizedInsight(
                    insight_id=f"cluster_insight_{cluster.cluster_id}",
                    insight_type=InsightType.EVIDENCE_CLUSTER,
                    description=f"Significant knowledge cluster with {len(cluster.entities)} entities",
                    confidence=cluster.cluster_significance,
                    supporting_evidence=[f"Cluster consistency: {cluster.internal_consistency:.2f}"],
                    contributing_phases=['synthesis'],
                    affected_entities=cluster.entities,
                    network_metrics={'cluster_size': len(cluster.entities)},
                    synthesis_path=[f"Knowledge clustering -> {cluster.cluster_id}"],
                    validation_score=cluster.internal_consistency,
                    impact_assessment={'network_understanding': cluster.cluster_significance},
                    recommendations=[f"Focus analysis on cluster entities: {', '.join(cluster.representative_elements)}"]
                )
                key_insights.append(insight)

        # Extract insights from causal chains
        for chain in causal_chains:
            if chain.chain_confidence > 0.7:
                insight = SynthesizedInsight(
                    insight_id=f"causal_insight_{chain.chain_id}",
                    insight_type=InsightType.CAUSAL_CHAIN,
                    description=f"Strong causal chain: {' -> '.join(chain.chain_elements)}",
                    confidence=chain.chain_confidence,
                    supporting_evidence=chain.supporting_evidence,
                    contributing_phases=['synthesis'],
                    affected_entities=chain.chain_elements,
                    network_metrics={'chain_length': len(chain.chain_elements)},
                    synthesis_path=[f"Causal analysis -> {chain.chain_id}"],
                    validation_score=chain.chain_confidence,
                    impact_assessment={'causal_understanding': chain.chain_confidence},
                    recommendations=chain.validation_requirements
                )
                key_insights.append(insight)

        # Network structure insights
        basic_metrics = network_analysis.get('basic_metrics', {})
        if basic_metrics.get('density', 0) > 0.3:
            insight = SynthesizedInsight(
                insight_id="network_density_insight",
                insight_type=InsightType.NETWORK_STRUCTURE,
                description=f"Dense network structure (density: {basic_metrics['density']:.3f})",
                confidence=0.8,
                supporting_evidence=[f"Network density: {basic_metrics['density']:.3f}"],
                contributing_phases=['synthesis'],
                affected_entities=[],
                network_metrics=basic_metrics,
                synthesis_path=["Network analysis -> density calculation"],
                validation_score=0.8,
                impact_assessment={'network_complexity': basic_metrics['density']},
                recommendations=["Consider network reduction techniques for clarity"]
            )
            key_insights.append(insight)

        # Sort insights by confidence
        key_insights.sort(key=lambda x: x.confidence, reverse=True)

        return key_insights

    async def _create_synthesized_insight(self, insight_data: Dict[str, Any],
                                        strategy_name: str,
                                        strategy_weight: float) -> SynthesizedInsight:
        """Create a synthesized insight from strategy result"""
        insight_type_mapping = {
            'hierarchical_entity': InsightType.CENTRAL_ENTITY,
            'hierarchical_relationship': InsightType.KEY_RELATIONSHIP,
            'central_entity': InsightType.CENTRAL_ENTITY,
            'evidence_convergence': InsightType.EVIDENCE_CLUSTER,
            'pattern_reinforcement': InsightType.EMERGENT_PATTERN,
            'semantic_cluster': InsightType.EVIDENCE_CLUSTER
        }

        insight_type = insight_type_mapping.get(insight_data.get('type'), InsightType.CENTRAL_ENTITY)

        # Adjust confidence by strategy weight
        adjusted_confidence = insight_data.get('confidence', 0.5) * strategy_weight

        return SynthesizedInsight(
            insight_id=f"{strategy_name}_{insight_data.get('type', 'unknown')}_{int(time.time())}",
            insight_type=insight_type,
            description=insight_data.get('description', 'No description'),
            confidence=adjusted_confidence,
            supporting_evidence=insight_data.get('evidence', []),
            contributing_phases=[strategy_name],
            affected_entities=insight_data.get('entities', [insight_data.get('entity', '')]),
            network_metrics=insight_data.get('centrality_metrics', {}),
            synthesis_path=[f"{strategy_name} -> {insight_data.get('type')}"],
            validation_score=adjusted_confidence,
            impact_assessment={strategy_name: adjusted_confidence},
            recommendations=[f"Validate through {strategy_name} methodology"]
        )

    async def _generate_integrated_analysis(self, key_insights: List[SynthesizedInsight],
                                          knowledge_clusters: List[KnowledgeCluster],
                                          causal_chains: List[CausalChain]) -> str:
        """Generate integrated analysis narrative"""
        logger.info("Generating integrated analysis")

        analysis_parts = []

        # Executive summary
        analysis_parts.append("## INTEGRATED ANALYSIS SUMMARY")
        analysis_parts.append(f"Analysis synthesized {len(key_insights)} key insights across multiple reasoning strategies.")

        # Key entities section
        central_entities = [insight for insight in key_insights if insight.insight_type == InsightType.CENTRAL_ENTITY]
        if central_entities:
            analysis_parts.append("\n### CENTRAL ENTITIES")
            for insight in central_entities[:3]:  # Top 3
                analysis_parts.append(f"- {insight.description} (confidence: {insight.confidence:.2f})")

        # Key relationships section
        key_relationships = [insight for insight in key_insights if insight.insight_type == InsightType.KEY_RELATIONSHIP]
        if key_relationships:
            analysis_parts.append("\n### KEY RELATIONSHIPS")
            for insight in key_relationships[:3]:
                analysis_parts.append(f"- {insight.description} (confidence: {insight.confidence:.2f})")

        # Pattern analysis
        patterns = [insight for insight in key_insights if insight.insight_type == InsightType.EMERGENT_PATTERN]
        if patterns:
            analysis_parts.append("\n### EMERGENT PATTERNS")
            for insight in patterns[:2]:
                analysis_parts.append(f"- {insight.description}")

        # Causal reasoning
        if causal_chains:
            analysis_parts.append("\n### CAUSAL ANALYSIS")
            for chain in causal_chains[:2]:
                confidence_desc = "high" if chain.chain_confidence > 0.8 else "medium" if chain.chain_confidence > 0.6 else "low"
                analysis_parts.append(f"- Causal chain detected: {' -> '.join(chain.chain_elements)} ({confidence_desc} confidence)")

        # Knowledge clusters
        if knowledge_clusters:
            analysis_parts.append("\n### KNOWLEDGE CLUSTERS")
            significant_clusters = [c for c in knowledge_clusters if c.cluster_significance > 0.6]
            for cluster in significant_clusters[:3]:
                analysis_parts.append(f"- Cluster '{cluster.cluster_id}': {len(cluster.entities)} entities, significance: {cluster.cluster_significance:.2f}")

        # Overall assessment
        avg_confidence = np.mean([insight.confidence for insight in key_insights]) if key_insights else 0.0
        confidence_level = "high" if avg_confidence > 0.7 else "medium" if avg_confidence > 0.5 else "low"

        analysis_parts.append(f"\n### CONFIDENCE ASSESSMENT")
        analysis_parts.append(f"Overall synthesis confidence: {confidence_level} ({avg_confidence:.2f})")

        return "\n".join(analysis_parts)

    async def _calculate_synthesis_metrics(self, key_insights: List[SynthesizedInsight],
                                         knowledge_clusters: List[KnowledgeCluster],
                                         causal_chains: List[CausalChain],
                                         confidence_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive synthesis metrics"""
        metrics = {}

        # Integration score based on insight diversity and confidence
        if key_insights:
            insight_types = set([insight.insight_type for insight in key_insights])
            type_diversity = len(insight_types) / len(InsightType)
            avg_confidence = np.mean([insight.confidence for insight in key_insights])
            metrics['integration_score'] = (type_diversity * 0.4 + avg_confidence * 0.6)
        else:
            metrics['integration_score'] = 0.0

        # Coherence score based on cluster consistency and insight agreement
        if knowledge_clusters:
            cluster_consistencies = [cluster.internal_consistency for cluster in knowledge_clusters]
            avg_consistency = np.mean(cluster_consistencies)
            metrics['coherence_score'] = avg_consistency
        else:
            metrics['coherence_score'] = 0.5

        # Completeness score based on phase coverage and insight coverage
        phase_coverage = len(confidence_scores) / 5  # 5 expected phases
        insight_coverage = len(key_insights) / 10  # Normalize to expected insights
        metrics['completeness_score'] = min(1.0, (phase_coverage * 0.6 + insight_coverage * 0.4))

        # Causal reasoning score
        if causal_chains:
            causal_confidences = [chain.chain_confidence for chain in causal_chains]
            metrics['causal_reasoning_score'] = np.mean(causal_confidences)
        else:
            metrics['causal_reasoning_score'] = 0.0

        # Evidence strength score
        phase_confidences = list(confidence_scores.values())
        if phase_confidences:
            metrics['evidence_strength_score'] = np.mean(phase_confidences)
        else:
            metrics['evidence_strength_score'] = 0.0

        return metrics

    async def _create_evidence_summary(self, phase_results: Dict[str, Any]) -> List[str]:
        """Create summary of evidence across all phases"""
        evidence_summary = []

        for phase_name, phase_data in phase_results.items():
            if isinstance(phase_data, dict):
                phase_summary = f"Phase '{phase_name}': "

                # Count evidence pieces
                entities_count = len(phase_data.get('entities', []))
                relationships_count = len(phase_data.get('relationships', []))
                patterns_count = len(phase_data.get('patterns', []))

                phase_summary += f"{entities_count} entities, {relationships_count} relationships, {patterns_count} patterns"
                evidence_summary.append(phase_summary)

        return evidence_summary

    def _synthesis_result_to_dict(self, synthesis_result: SynthesisResult,
                                processing_time: float) -> Dict[str, Any]:
        """Convert synthesis result to dictionary"""
        return {
            'synthesis_id': synthesis_result.synthesis_id,
            'integrated_analysis': synthesis_result.integrated_analysis,
            'key_insights': [self._insight_to_dict(insight) for insight in synthesis_result.key_insights],
            'knowledge_clusters': [self._cluster_to_dict(cluster) for cluster in synthesis_result.knowledge_clusters],
            'causal_chains': [self._causal_chain_to_dict(chain) for chain in synthesis_result.causal_chains],
            'network_analysis': synthesis_result.network_analysis,
            'integration_score': synthesis_result.integration_score,
            'coherence_score': synthesis_result.coherence_score,
            'completeness_score': synthesis_result.completeness_score,
            'evidence_summary': synthesis_result.evidence_summary,
            'confidence_assessment': synthesis_result.confidence_assessment,
            'synthesis_metrics': synthesis_result.synthesis_metrics,
            'processing_time': processing_time
        }

    def _insight_to_dict(self, insight: SynthesizedInsight) -> Dict[str, Any]:
        """Convert SynthesizedInsight to dictionary"""
        return {
            'insight_id': insight.insight_id,
            'type': insight.insight_type.value,
            'description': insight.description,
            'confidence': insight.confidence,
            'supporting_evidence': insight.supporting_evidence,
            'contributing_phases': insight.contributing_phases,
            'affected_entities': insight.affected_entities,
            'network_metrics': insight.network_metrics,
            'synthesis_path': insight.synthesis_path,
            'validation_score': insight.validation_score,
            'impact_assessment': insight.impact_assessment,
            'recommendations': insight.recommendations
        }

    def _cluster_to_dict(self, cluster: KnowledgeCluster) -> Dict[str, Any]:
        """Convert KnowledgeCluster to dictionary"""
        return {
            'cluster_id': cluster.cluster_id,
            'entities': cluster.entities,
            'relationships': cluster.relationships,
            'patterns': cluster.patterns,
            'evidence_strength': cluster.evidence_strength,
            'internal_consistency': cluster.internal_consistency,
            'cluster_significance': cluster.cluster_significance,
            'representative_elements': cluster.representative_elements
        }

    def _causal_chain_to_dict(self, chain: CausalChain) -> Dict[str, Any]:
        """Convert CausalChain to dictionary"""
        return {
            'chain_id': chain.chain_id,
            'chain_elements': chain.chain_elements,
            'causal_links': chain.causal_links,
            'chain_confidence': chain.chain_confidence,
            'supporting_evidence': chain.supporting_evidence,
            'alternative_explanations': chain.alternative_explanations,
            'validation_requirements': chain.validation_requirements
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for knowledge synthesizer"""
        return {
            'status': 'healthy',
            'synthesis_strategies': len(self.strategy_weights),
            'network_metrics': len(self.network_metrics),
            'min_cluster_size': self.min_cluster_size,
            'max_clusters': self.max_clusters,
            'centrality_threshold': self.centrality_threshold,
            'evidence_convergence_threshold': self.evidence_convergence_threshold,
            'min_causal_strength': self.min_causal_strength
        }