"""
Counterfactual Analyzer for BEV OSINT Framework
Implements hypothesis testing and alternative scenario analysis
"""

import asyncio
import logging
import time
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import json
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
from itertools import combinations
import heapq

logger = logging.getLogger(__name__)

class HypothesisType(Enum):
    """Types of counterfactual hypotheses"""
    ALTERNATIVE_ATTRIBUTION = "alternative_attribution"
    MISSING_ENTITY = "missing_entity"
    ALTERED_RELATIONSHIP = "altered_relationship"
    TEMPORAL_VARIATION = "temporal_variation"
    CAUSAL_INVERSION = "causal_inversion"
    SCENARIO_NEGATION = "scenario_negation"

class ConfidenceLevel(Enum):
    """Confidence levels for hypotheses"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9

@dataclass
class AlternativeHypothesis:
    """Container for alternative hypothesis"""
    hypothesis_id: str
    hypothesis_type: HypothesisType
    description: str
    strength: float  # How plausible this alternative is
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    affected_entities: List[str]
    altered_relationships: List[Tuple[str, str, str]]  # (source, target, new_relation)
    probability_estimate: float
    impact_assessment: Dict[str, float]
    reasoning_chain: List[str]
    validation_tests: List[Dict[str, Any]]

@dataclass
class CounterfactualScenario:
    """Complete counterfactual scenario"""
    scenario_id: str
    base_hypothesis: str
    alternative_hypotheses: List[AlternativeHypothesis]
    scenario_strength: float
    consistency_score: float
    evidence_requirements: List[str]
    testable_predictions: List[str]
    risk_assessment: Dict[str, float]

@dataclass
class HypothesisTest:
    """Hypothesis testing framework"""
    test_id: str
    hypothesis_id: str
    test_type: str
    test_description: str
    expected_outcome: str
    confidence_threshold: float
    validation_criteria: List[str]
    test_results: Optional[Dict[str, Any]] = None

class CounterfactualAnalyzer:
    """
    Advanced counterfactual analysis system for OSINT investigation
    Generates and tests alternative hypotheses
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_hypotheses = config.get('max_hypotheses', 10)
        self.min_hypothesis_strength = config.get('min_hypothesis_strength', 0.3)
        self.evidence_weight_threshold = config.get('evidence_weight_threshold', 0.6)
        self.relationship_variation_factor = config.get('relationship_variation_factor', 0.8)

        # Hypothesis generation strategies
        self.generation_strategies = {
            HypothesisType.ALTERNATIVE_ATTRIBUTION: self._generate_attribution_alternatives,
            HypothesisType.MISSING_ENTITY: self._generate_missing_entity_hypotheses,
            HypothesisType.ALTERED_RELATIONSHIP: self._generate_relationship_alternatives,
            HypothesisType.TEMPORAL_VARIATION: self._generate_temporal_variations,
            HypothesisType.CAUSAL_INVERSION: self._generate_causal_inversions,
            HypothesisType.SCENARIO_NEGATION: self._generate_scenario_negations
        }

        # Testing frameworks
        self.test_frameworks = {
            'consistency_test': self._test_internal_consistency,
            'evidence_test': self._test_evidence_support,
            'logical_test': self._test_logical_coherence,
            'plausibility_test': self._test_scenario_plausibility,
            'impact_test': self._test_impact_assessment
        }

        # Validation metrics
        self.validation_metrics = defaultdict(list)

    async def analyze(self, knowledge_graph: nx.DiGraph, phase_results: Dict[str, Any],
                     hypothesis_count: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive counterfactual analysis

        Args:
            knowledge_graph: Knowledge graph from synthesis phase
            phase_results: Results from all previous phases
            hypothesis_count: Number of hypotheses to generate

        Returns:
            Dictionary with counterfactual analysis results
        """
        start_time = time.time()
        logger.info("Starting counterfactual analysis")

        try:
            # Extract base scenario from phase results
            base_scenario = await self._extract_base_scenario(knowledge_graph, phase_results)

            # Generate alternative hypotheses
            alternative_hypotheses = await self._generate_hypotheses(
                knowledge_graph, phase_results, hypothesis_count
            )

            # Test hypotheses
            tested_hypotheses = await self._test_hypotheses(
                alternative_hypotheses, knowledge_graph, phase_results
            )

            # Create counterfactual scenarios
            scenarios = await self._create_scenarios(tested_hypotheses, base_scenario)

            # Assess scenario strengths and risks
            scenario_assessment = await self._assess_scenarios(scenarios, knowledge_graph)

            # Generate validation framework
            validation_framework = await self._create_validation_framework(scenarios)

            processing_time = time.time() - start_time

            return {
                'base_scenario': base_scenario,
                'alternative_hypotheses': [self._hypothesis_to_dict(h) for h in tested_hypotheses],
                'counterfactual_scenarios': [self._scenario_to_dict(s) for s in scenarios],
                'scenario_assessment': scenario_assessment,
                'validation_framework': validation_framework,
                'strong_alternatives': len([h for h in tested_hypotheses if h.strength > 0.7]),
                'processing_time': processing_time,
                'generation_stats': self._get_generation_stats(tested_hypotheses)
            }

        except Exception as e:
            logger.error(f"Error in counterfactual analysis: {str(e)}")
            raise

    async def _extract_base_scenario(self, knowledge_graph: nx.DiGraph,
                                   phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract base scenario from analysis results"""
        logger.info("Extracting base scenario")

        # Get synthesis results
        synthesis = phase_results.get('synthesis', {})
        entities = phase_results.get('exploration', {}).get('entities', [])
        relationships = phase_results.get('exploration', {}).get('relationships', [])

        # Key assertions from the analysis
        key_assertions = []

        # Entity assertions
        high_confidence_entities = [e for e in entities if e.get('confidence', 0) > 0.7]
        for entity in high_confidence_entities:
            key_assertions.append(f"Entity '{entity['name']}' exists with type {entity.get('type', 'unknown')}")

        # Relationship assertions
        high_confidence_rels = [r for r in relationships if r.get('confidence', 0) > 0.7]
        for rel in high_confidence_rels:
            key_assertions.append(f"Relationship: {rel['source']} {rel['relation']} {rel['target']}")

        # Pattern assertions
        patterns = phase_results.get('exploration', {}).get('patterns', [])
        significant_patterns = [p for p in patterns if p.get('significance', 0) > 0.7]
        for pattern in significant_patterns:
            key_assertions.append(f"Pattern detected: {pattern['type']} - {pattern['description']}")

        base_scenario = {
            'key_assertions': key_assertions,
            'primary_entities': [e['name'] for e in high_confidence_entities],
            'primary_relationships': [(r['source'], r['target'], r['relation']) for r in high_confidence_rels],
            'supporting_evidence': synthesis.get('evidence_summary', []),
            'confidence_level': synthesis.get('overall_confidence', 0.6)
        }

        return base_scenario

    async def _generate_hypotheses(self, knowledge_graph: nx.DiGraph,
                                 phase_results: Dict[str, Any],
                                 target_count: int) -> List[AlternativeHypothesis]:
        """Generate alternative hypotheses using multiple strategies"""
        logger.info(f"Generating {target_count} alternative hypotheses")

        all_hypotheses = []

        # Generate hypotheses for each type
        for hypothesis_type, generator_func in self.generation_strategies.items():
            try:
                hypotheses = await generator_func(knowledge_graph, phase_results)
                all_hypotheses.extend(hypotheses)
            except Exception as e:
                logger.warning(f"Error generating {hypothesis_type.value} hypotheses: {str(e)}")

        # Filter and rank hypotheses
        filtered_hypotheses = [h for h in all_hypotheses if h.strength >= self.min_hypothesis_strength]

        # Sort by strength and select top hypotheses
        filtered_hypotheses.sort(key=lambda x: x.strength, reverse=True)

        return filtered_hypotheses[:target_count]

    async def _generate_attribution_alternatives(self, knowledge_graph: nx.DiGraph,
                                               phase_results: Dict[str, Any]) -> List[AlternativeHypothesis]:
        """Generate alternative attribution hypotheses"""
        hypotheses = []
        entities = phase_results.get('exploration', {}).get('entities', [])
        relationships = phase_results.get('exploration', {}).get('relationships', [])

        # Find high-confidence attributions that could be questioned
        for rel in relationships:
            if rel.get('confidence', 0) > 0.6 and rel.get('relation') in ['owns', 'created', 'founded']:
                # Generate alternative attribution
                alternative_sources = [e['name'] for e in entities
                                     if e['name'] != rel['source'] and e.get('type') == 'person']

                for alt_source in alternative_sources[:3]:  # Limit alternatives
                    hypothesis_id = f"alt_attr_{rel['source']}_{alt_source}_{int(time.time())}"

                    hypothesis = AlternativeHypothesis(
                        hypothesis_id=hypothesis_id,
                        hypothesis_type=HypothesisType.ALTERNATIVE_ATTRIBUTION,
                        description=f"Alternative: {alt_source} (not {rel['source']}) {rel['relation']} {rel['target']}",
                        strength=self._calculate_attribution_strength(alt_source, rel, entities),
                        supporting_evidence=[],
                        contradicting_evidence=[rel.get('evidence', '')],
                        affected_entities=[rel['source'], alt_source, rel['target']],
                        altered_relationships=[(alt_source, rel['target'], rel['relation'])],
                        probability_estimate=0.3,  # Default alternative probability
                        impact_assessment={'attribution_change': 0.8, 'network_impact': 0.5},
                        reasoning_chain=[
                            f"Original attribution: {rel['source']} {rel['relation']} {rel['target']}",
                            f"Alternative attribution: {alt_source} {rel['relation']} {rel['target']}",
                            f"This would change the primary responsibility/ownership"
                        ],
                        validation_tests=[]
                    )

                    hypotheses.append(hypothesis)

        return hypotheses

    def _calculate_attribution_strength(self, alternative_source: str,
                                      original_relationship: Dict[str, Any],
                                      entities: List[Dict[str, Any]]) -> float:
        """Calculate strength of alternative attribution"""
        # Find the alternative entity
        alt_entity = next((e for e in entities if e['name'] == alternative_source), None)
        if not alt_entity:
            return 0.2

        # Base strength on entity importance and type compatibility
        entity_importance = alt_entity.get('importance_score', 0.5)
        type_compatibility = 0.8 if alt_entity.get('type') in ['person', 'organization'] else 0.4

        # Reduce strength based on original relationship confidence
        original_confidence = original_relationship.get('confidence', 0.5)
        confidence_factor = 1.0 - original_confidence

        return min(0.9, entity_importance * type_compatibility * confidence_factor)

    async def _generate_missing_entity_hypotheses(self, knowledge_graph: nx.DiGraph,
                                                phase_results: Dict[str, Any]) -> List[AlternativeHypothesis]:
        """Generate hypotheses about missing entities"""
        hypotheses = []
        entities = phase_results.get('exploration', {}).get('entities', [])
        relationships = phase_results.get('exploration', {}).get('relationships', [])

        # Analyze relationship patterns to infer missing entities
        entity_names = {e['name'] for e in entities}

        # Look for incomplete relationship chains
        for rel in relationships:
            source, target = rel['source'], rel['target']

            # Check for missing intermediary entities
            if rel.get('relation') in ['transferred_to', 'communicated_with', 'received_from']:
                hypothesis_id = f"missing_intermediary_{source}_{target}_{int(time.time())}"

                hypothesis = AlternativeHypothesis(
                    hypothesis_id=hypothesis_id,
                    hypothesis_type=HypothesisType.MISSING_ENTITY,
                    description=f"Missing intermediary entity between {source} and {target}",
                    strength=self._calculate_missing_entity_strength(rel, entities),
                    supporting_evidence=[f"Direct relationship: {source} -> {target} may indicate missing steps"],
                    contradicting_evidence=[],
                    affected_entities=[source, target],
                    altered_relationships=[],
                    probability_estimate=0.4,
                    impact_assessment={'investigation_scope': 0.7, 'evidence_gap': 0.8},
                    reasoning_chain=[
                        f"Observed direct relationship: {source} {rel['relation']} {target}",
                        "In complex scenarios, direct relationships often involve intermediaries",
                        "Missing entity could change the nature and significance of the connection"
                    ],
                    validation_tests=[]
                )

                hypotheses.append(hypothesis)

        # Look for missing entities based on network holes
        if knowledge_graph.number_of_nodes() > 3:
            network_hypotheses = await self._analyze_network_holes(knowledge_graph, entities)
            hypotheses.extend(network_hypotheses)

        return hypotheses

    def _calculate_missing_entity_strength(self, relationship: Dict[str, Any],
                                         entities: List[Dict[str, Any]]) -> float:
        """Calculate likelihood of missing entity"""
        # Higher strength for complex relationship types
        complex_relations = ['transferred_to', 'received_from', 'communicated_with']
        relation_complexity = 0.8 if relationship.get('relation') in complex_relations else 0.4

        # Consider network density
        source_connections = sum(1 for e in entities if e['name'] == relationship['source'])
        target_connections = sum(1 for e in entities if e['name'] == relationship['target'])

        network_sparsity = 1.0 - min(1.0, (source_connections + target_connections) / 10)

        return min(0.8, relation_complexity * network_sparsity * 0.7)

    async def _analyze_network_holes(self, knowledge_graph: nx.DiGraph,
                                   entities: List[Dict[str, Any]]) -> List[AlternativeHypothesis]:
        """Analyze network structure for potential missing entities"""
        hypotheses = []

        try:
            # Find nodes with unexpectedly low connectivity
            degrees = dict(knowledge_graph.degree())
            avg_degree = np.mean(list(degrees.values())) if degrees else 0

            for node, degree in degrees.items():
                if degree < avg_degree * 0.5 and avg_degree > 2:  # Isolated node
                    hypothesis_id = f"missing_connections_{node}_{int(time.time())}"

                    hypothesis = AlternativeHypothesis(
                        hypothesis_id=hypothesis_id,
                        hypothesis_type=HypothesisType.MISSING_ENTITY,
                        description=f"Missing entities connected to {node} (isolated node)",
                        strength=0.6,
                        supporting_evidence=[f"Node {node} has unusually low connectivity"],
                        contradicting_evidence=[],
                        affected_entities=[node],
                        altered_relationships=[],
                        probability_estimate=0.5,
                        impact_assessment={'network_completeness': 0.6},
                        reasoning_chain=[
                            f"Node {node} has degree {degree}, average is {avg_degree:.2f}",
                            "Low connectivity may indicate missing relationships or entities"
                        ],
                        validation_tests=[]
                    )

                    hypotheses.append(hypothesis)

        except Exception as e:
            logger.warning(f"Error in network hole analysis: {str(e)}")

        return hypotheses

    async def _generate_relationship_alternatives(self, knowledge_graph: nx.DiGraph,
                                                phase_results: Dict[str, Any]) -> List[AlternativeHypothesis]:
        """Generate alternative relationship interpretations"""
        hypotheses = []
        relationships = phase_results.get('exploration', {}).get('relationships', [])

        # Define alternative relationship mappings
        relationship_alternatives = {
            'owns': ['controls', 'manages', 'influences'],
            'employed_by': ['contracted_to', 'consulting_for', 'affiliated_with'],
            'communicated_with': ['negotiated_with', 'confronted', 'coordinated_with'],
            'transferred_to': ['sold_to', 'gave_to', 'invested_in'],
            'located_in': ['operates_from', 'registered_in', 'hiding_in']
        }

        for rel in relationships:
            if rel.get('confidence', 0) < 0.8:  # Only question lower confidence relationships
                relation_type = rel.get('relation', '')
                alternatives = relationship_alternatives.get(relation_type, [])

                for alt_relation in alternatives:
                    hypothesis_id = f"alt_rel_{rel['source']}_{alt_relation}_{int(time.time())}"

                    hypothesis = AlternativeHypothesis(
                        hypothesis_id=hypothesis_id,
                        hypothesis_type=HypothesisType.ALTERED_RELATIONSHIP,
                        description=f"Alternative relationship: {rel['source']} {alt_relation} {rel['target']} (not {relation_type})",
                        strength=self._calculate_relationship_alternative_strength(rel, alt_relation),
                        supporting_evidence=[],
                        contradicting_evidence=[rel.get('evidence', '')],
                        affected_entities=[rel['source'], rel['target']],
                        altered_relationships=[(rel['source'], rel['target'], alt_relation)],
                        probability_estimate=0.3,
                        impact_assessment={'relationship_meaning': 0.7, 'causal_chain': 0.5},
                        reasoning_chain=[
                            f"Original relationship: {rel['source']} {relation_type} {rel['target']}",
                            f"Alternative: {rel['source']} {alt_relation} {rel['target']}",
                            f"This changes the nature and implications of the connection"
                        ],
                        validation_tests=[]
                    )

                    hypotheses.append(hypothesis)

        return hypotheses

    def _calculate_relationship_alternative_strength(self, original_rel: Dict[str, Any],
                                                   alternative_relation: str) -> float:
        """Calculate strength of alternative relationship interpretation"""
        original_confidence = original_rel.get('confidence', 0.5)

        # Lower original confidence makes alternatives stronger
        confidence_factor = 1.0 - original_confidence

        # Some alternatives are more plausible than others
        plausibility_map = {
            'controls': 0.8, 'manages': 0.9, 'influences': 0.7,
            'contracted_to': 0.8, 'consulting_for': 0.7, 'affiliated_with': 0.6,
            'negotiated_with': 0.7, 'confronted': 0.5, 'coordinated_with': 0.8,
            'sold_to': 0.8, 'gave_to': 0.6, 'invested_in': 0.7,
            'operates_from': 0.9, 'registered_in': 0.8, 'hiding_in': 0.4
        }

        alternative_plausibility = plausibility_map.get(alternative_relation, 0.5)

        return min(0.9, confidence_factor * alternative_plausibility * 0.8)

    async def _generate_temporal_variations(self, knowledge_graph: nx.DiGraph,
                                          phase_results: Dict[str, Any]) -> List[AlternativeHypothesis]:
        """Generate temporal variation hypotheses"""
        hypotheses = []

        # Look for time-sensitive patterns
        temporal_indicators = ['before', 'after', 'during', 'when', 'then']

        # This is a simplified implementation - in practice, would need sophisticated
        # temporal reasoning and timeline reconstruction

        hypothesis_id = f"temporal_sequence_{int(time.time())}"

        hypothesis = AlternativeHypothesis(
            hypothesis_id=hypothesis_id,
            hypothesis_type=HypothesisType.TEMPORAL_VARIATION,
            description="Alternative temporal sequence of events",
            strength=0.4,
            supporting_evidence=["Temporal indicators present in text"],
            contradicting_evidence=[],
            affected_entities=[],
            altered_relationships=[],
            probability_estimate=0.3,
            impact_assessment={'timeline_accuracy': 0.8},
            reasoning_chain=[
                "Events may have occurred in different temporal order",
                "Timeline reconstruction affects causal interpretation"
            ],
            validation_tests=[]
        )

        hypotheses.append(hypothesis)

        return hypotheses

    async def _generate_causal_inversions(self, knowledge_graph: nx.DiGraph,
                                        phase_results: Dict[str, Any]) -> List[AlternativeHypothesis]:
        """Generate causal inversion hypotheses"""
        hypotheses = []
        relationships = phase_results.get('exploration', {}).get('relationships', [])

        # Look for potentially bidirectional relationships
        causal_relations = ['owns', 'controls', 'influences', 'caused']

        for rel in relationships:
            if any(causal_rel in rel.get('relation', '') for causal_rel in causal_relations):
                hypothesis_id = f"causal_inversion_{rel['source']}_{rel['target']}_{int(time.time())}"

                hypothesis = AlternativeHypothesis(
                    hypothesis_id=hypothesis_id,
                    hypothesis_type=HypothesisType.CAUSAL_INVERSION,
                    description=f"Causal inversion: {rel['target']} influences {rel['source']} (not vice versa)",
                    strength=0.4,
                    supporting_evidence=["Causal relationships can be misattributed"],
                    contradicting_evidence=[rel.get('evidence', '')],
                    affected_entities=[rel['source'], rel['target']],
                    altered_relationships=[(rel['target'], rel['source'], rel.get('relation', ''))],
                    probability_estimate=0.25,
                    impact_assessment={'causal_understanding': 0.9},
                    reasoning_chain=[
                        f"Original: {rel['source']} -> {rel['target']}",
                        f"Inversion: {rel['target']} -> {rel['source']}",
                        "Causal direction significantly affects interpretation"
                    ],
                    validation_tests=[]
                )

                hypotheses.append(hypothesis)

        return hypotheses

    async def _generate_scenario_negations(self, knowledge_graph: nx.DiGraph,
                                         phase_results: Dict[str, Any]) -> List[AlternativeHypothesis]:
        """Generate scenario negation hypotheses"""
        hypotheses = []

        # Generate negation of main scenario conclusions
        synthesis = phase_results.get('synthesis', {})
        main_conclusions = synthesis.get('key_findings', [])

        for i, conclusion in enumerate(main_conclusions[:3]):  # Limit to top 3
            hypothesis_id = f"scenario_negation_{i}_{int(time.time())}"

            hypothesis = AlternativeHypothesis(
                hypothesis_id=hypothesis_id,
                hypothesis_type=HypothesisType.SCENARIO_NEGATION,
                description=f"Negation of conclusion: {conclusion}",
                strength=0.3,
                supporting_evidence=["Alternative interpretations possible"],
                contradicting_evidence=[conclusion],
                affected_entities=[],
                altered_relationships=[],
                probability_estimate=0.2,
                impact_assessment={'scenario_validity': 1.0},
                reasoning_chain=[
                    f"Main conclusion: {conclusion}",
                    "Consider: What if this conclusion is incorrect?",
                    "Negation forces reconsideration of evidence interpretation"
                ],
                validation_tests=[]
            )

            hypotheses.append(hypothesis)

        return hypotheses

    async def _test_hypotheses(self, hypotheses: List[AlternativeHypothesis],
                              knowledge_graph: nx.DiGraph,
                              phase_results: Dict[str, Any]) -> List[AlternativeHypothesis]:
        """Test hypotheses using multiple validation frameworks"""
        logger.info(f"Testing {len(hypotheses)} hypotheses")

        tested_hypotheses = []

        for hypothesis in hypotheses:
            # Run all test frameworks
            test_results = {}

            for test_name, test_func in self.test_frameworks.items():
                try:
                    result = await test_func(hypothesis, knowledge_graph, phase_results)
                    test_results[test_name] = result
                except Exception as e:
                    logger.warning(f"Error in {test_name} for hypothesis {hypothesis.hypothesis_id}: {str(e)}")
                    test_results[test_name] = {'score': 0.0, 'notes': f"Test failed: {str(e)}"}

            # Update hypothesis strength based on test results
            updated_hypothesis = await self._update_hypothesis_strength(hypothesis, test_results)
            tested_hypotheses.append(updated_hypothesis)

        # Sort by final strength
        tested_hypotheses.sort(key=lambda x: x.strength, reverse=True)

        return tested_hypotheses

    async def _test_internal_consistency(self, hypothesis: AlternativeHypothesis,
                                       knowledge_graph: nx.DiGraph,
                                       phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test internal consistency of hypothesis"""
        consistency_score = 0.8  # Default high consistency

        # Check for logical contradictions
        contradictions = []

        # Check if altered relationships create cycles or conflicts
        if hypothesis.altered_relationships:
            for source, target, relation in hypothesis.altered_relationships:
                # Check if this creates a logical conflict
                if knowledge_graph.has_edge(target, source):
                    existing_relation = knowledge_graph[target][source].get('relation', '')
                    if existing_relation in ['owns', 'controls'] and relation in ['owns', 'controls']:
                        contradictions.append(f"Circular ownership: {source} <-> {target}")
                        consistency_score -= 0.3

        return {
            'score': max(0.0, consistency_score),
            'contradictions': contradictions,
            'notes': f"Found {len(contradictions)} logical contradictions"
        }

    async def _test_evidence_support(self, hypothesis: AlternativeHypothesis,
                                   knowledge_graph: nx.DiGraph,
                                   phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test evidence support for hypothesis"""
        supporting_strength = len(hypothesis.supporting_evidence) * 0.2
        contradicting_penalty = len(hypothesis.contradicting_evidence) * 0.3

        evidence_score = max(0.0, 0.5 + supporting_strength - contradicting_penalty)

        return {
            'score': min(1.0, evidence_score),
            'supporting_count': len(hypothesis.supporting_evidence),
            'contradicting_count': len(hypothesis.contradicting_evidence),
            'notes': f"Evidence balance: +{len(hypothesis.supporting_evidence)} / -{len(hypothesis.contradicting_evidence)}"
        }

    async def _test_logical_coherence(self, hypothesis: AlternativeHypothesis,
                                    knowledge_graph: nx.DiGraph,
                                    phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test logical coherence of hypothesis"""
        coherence_score = 0.7  # Default moderate coherence

        # Evaluate reasoning chain
        reasoning_quality = len(hypothesis.reasoning_chain) * 0.15
        coherence_score = min(1.0, coherence_score + reasoning_quality)

        return {
            'score': coherence_score,
            'reasoning_steps': len(hypothesis.reasoning_chain),
            'notes': f"Reasoning chain has {len(hypothesis.reasoning_chain)} steps"
        }

    async def _test_scenario_plausibility(self, hypothesis: AlternativeHypothesis,
                                        knowledge_graph: nx.DiGraph,
                                        phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test overall scenario plausibility"""
        # Base plausibility on hypothesis type
        type_plausibilities = {
            HypothesisType.ALTERNATIVE_ATTRIBUTION: 0.7,
            HypothesisType.MISSING_ENTITY: 0.6,
            HypothesisType.ALTERED_RELATIONSHIP: 0.5,
            HypothesisType.TEMPORAL_VARIATION: 0.4,
            HypothesisType.CAUSAL_INVERSION: 0.3,
            HypothesisType.SCENARIO_NEGATION: 0.2
        }

        base_plausibility = type_plausibilities.get(hypothesis.hypothesis_type, 0.5)

        # Adjust based on probability estimate
        adjusted_plausibility = (base_plausibility + hypothesis.probability_estimate) / 2

        return {
            'score': adjusted_plausibility,
            'base_plausibility': base_plausibility,
            'probability_estimate': hypothesis.probability_estimate,
            'notes': f"Type: {hypothesis.hypothesis_type.value}, Base: {base_plausibility}"
        }

    async def _test_impact_assessment(self, hypothesis: AlternativeHypothesis,
                                    knowledge_graph: nx.DiGraph,
                                    phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test impact assessment of hypothesis"""
        total_impact = sum(hypothesis.impact_assessment.values())
        avg_impact = total_impact / max(len(hypothesis.impact_assessment), 1)

        # High impact hypotheses are more significant
        impact_score = min(1.0, avg_impact)

        return {
            'score': impact_score,
            'total_impact': total_impact,
            'average_impact': avg_impact,
            'impact_areas': len(hypothesis.impact_assessment),
            'notes': f"Average impact: {avg_impact:.2f} across {len(hypothesis.impact_assessment)} areas"
        }

    async def _update_hypothesis_strength(self, hypothesis: AlternativeHypothesis,
                                        test_results: Dict[str, Any]) -> AlternativeHypothesis:
        """Update hypothesis strength based on test results"""
        # Weight different test types
        test_weights = {
            'consistency_test': 0.25,
            'evidence_test': 0.30,
            'logical_test': 0.20,
            'plausibility_test': 0.15,
            'impact_test': 0.10
        }

        weighted_score = 0.0
        total_weight = 0.0

        for test_name, weight in test_weights.items():
            if test_name in test_results:
                score = test_results[test_name].get('score', 0.0)
                weighted_score += score * weight
                total_weight += weight

        if total_weight > 0:
            final_strength = weighted_score / total_weight
            # Combine with original strength (weighted average)
            hypothesis.strength = (hypothesis.strength * 0.3 + final_strength * 0.7)

        # Add validation tests to hypothesis
        for test_name, result in test_results.items():
            test = HypothesisTest(
                test_id=f"{hypothesis.hypothesis_id}_{test_name}",
                hypothesis_id=hypothesis.hypothesis_id,
                test_type=test_name,
                test_description=f"{test_name.replace('_', ' ').title()} validation",
                expected_outcome="High score indicates strong hypothesis",
                confidence_threshold=0.6,
                validation_criteria=[result.get('notes', 'No notes')],
                test_results=result
            )
            hypothesis.validation_tests.append(self._test_to_dict(test))

        return hypothesis

    async def _create_scenarios(self, hypotheses: List[AlternativeHypothesis],
                              base_scenario: Dict[str, Any]) -> List[CounterfactualScenario]:
        """Create complete counterfactual scenarios"""
        logger.info("Creating counterfactual scenarios")

        scenarios = []

        # Group hypotheses by theme/impact
        hypothesis_groups = self._group_hypotheses_by_theme(hypotheses)

        for theme, theme_hypotheses in hypothesis_groups.items():
            if len(theme_hypotheses) >= 1:  # Minimum hypotheses for scenario
                scenario_id = f"scenario_{theme}_{int(time.time())}"

                scenario = CounterfactualScenario(
                    scenario_id=scenario_id,
                    base_hypothesis=f"Alternative {theme} scenario",
                    alternative_hypotheses=theme_hypotheses[:3],  # Top 3 hypotheses
                    scenario_strength=np.mean([h.strength for h in theme_hypotheses[:3]]),
                    consistency_score=await self._calculate_scenario_consistency(theme_hypotheses[:3]),
                    evidence_requirements=self._identify_evidence_requirements(theme_hypotheses[:3]),
                    testable_predictions=self._generate_testable_predictions(theme_hypotheses[:3]),
                    risk_assessment=self._assess_scenario_risks(theme_hypotheses[:3])
                )

                scenarios.append(scenario)

        return scenarios

    def _group_hypotheses_by_theme(self, hypotheses: List[AlternativeHypothesis]) -> Dict[str, List[AlternativeHypothesis]]:
        """Group hypotheses by thematic similarity"""
        groups = defaultdict(list)

        for hypothesis in hypotheses:
            # Simple grouping by hypothesis type
            theme = hypothesis.hypothesis_type.value
            groups[theme].append(hypothesis)

        return dict(groups)

    async def _calculate_scenario_consistency(self, hypotheses: List[AlternativeHypothesis]) -> float:
        """Calculate consistency across hypotheses in scenario"""
        if len(hypotheses) <= 1:
            return 1.0

        # Check for conflicts between hypotheses
        conflicts = 0
        total_pairs = 0

        for i, h1 in enumerate(hypotheses):
            for h2 in hypotheses[i+1:]:
                total_pairs += 1

                # Check for entity conflicts
                if set(h1.affected_entities) & set(h2.affected_entities):
                    # Same entities affected - check for contradictions
                    if self._hypotheses_conflict(h1, h2):
                        conflicts += 1

        if total_pairs == 0:
            return 1.0

        consistency = 1.0 - (conflicts / total_pairs)
        return max(0.0, consistency)

    def _hypotheses_conflict(self, h1: AlternativeHypothesis, h2: AlternativeHypothesis) -> bool:
        """Check if two hypotheses conflict"""
        # Simple conflict detection - could be much more sophisticated

        # Check for conflicting relationship alterations
        h1_rels = set(h1.altered_relationships)
        h2_rels = set(h2.altered_relationships)

        # If they alter the same relationship differently, they conflict
        h1_pairs = {(r[0], r[1]) for r in h1_rels}
        h2_pairs = {(r[0], r[1]) for r in h2_rels}

        common_pairs = h1_pairs & h2_pairs
        if common_pairs:
            # Check if the relationships are different
            for pair in common_pairs:
                h1_rel = next((r[2] for r in h1_rels if (r[0], r[1]) == pair), None)
                h2_rel = next((r[2] for r in h2_rels if (r[0], r[1]) == pair), None)
                if h1_rel != h2_rel:
                    return True

        return False

    def _identify_evidence_requirements(self, hypotheses: List[AlternativeHypothesis]) -> List[str]:
        """Identify evidence requirements for scenario validation"""
        requirements = set()

        for hypothesis in hypotheses:
            if hypothesis.hypothesis_type == HypothesisType.ALTERNATIVE_ATTRIBUTION:
                requirements.add("Independent verification of attribution claims")
            elif hypothesis.hypothesis_type == HypothesisType.MISSING_ENTITY:
                requirements.add("Additional data sources to identify missing entities")
            elif hypothesis.hypothesis_type == HypothesisType.ALTERED_RELATIONSHIP:
                requirements.add("Relationship type verification from multiple sources")
            elif hypothesis.hypothesis_type == HypothesisType.TEMPORAL_VARIATION:
                requirements.add("Temporal sequence verification and timeline reconstruction")

        return list(requirements)

    def _generate_testable_predictions(self, hypotheses: List[AlternativeHypothesis]) -> List[str]:
        """Generate testable predictions from hypotheses"""
        predictions = []

        for hypothesis in hypotheses:
            if hypothesis.altered_relationships:
                for source, target, relation in hypothesis.altered_relationships:
                    predictions.append(f"If true, should find evidence of {source} {relation} {target}")

            if hypothesis.affected_entities:
                predictions.append(f"Should find additional evidence involving {', '.join(hypothesis.affected_entities)}")

        return predictions

    def _assess_scenario_risks(self, hypotheses: List[AlternativeHypothesis]) -> Dict[str, float]:
        """Assess risks associated with scenario"""
        risks = {
            'investigation_misdirection': 0.0,
            'evidence_misinterpretation': 0.0,
            'conclusion_invalidity': 0.0
        }

        for hypothesis in hypotheses:
            # Higher strength hypotheses pose higher risks if wrong
            risk_factor = hypothesis.strength

            if hypothesis.hypothesis_type in [HypothesisType.ALTERNATIVE_ATTRIBUTION, HypothesisType.CAUSAL_INVERSION]:
                risks['investigation_misdirection'] = max(risks['investigation_misdirection'], risk_factor)

            if hypothesis.hypothesis_type == HypothesisType.ALTERED_RELATIONSHIP:
                risks['evidence_misinterpretation'] = max(risks['evidence_misinterpretation'], risk_factor)

            if hypothesis.hypothesis_type == HypothesisType.SCENARIO_NEGATION:
                risks['conclusion_invalidity'] = max(risks['conclusion_invalidity'], risk_factor)

        return risks

    async def _assess_scenarios(self, scenarios: List[CounterfactualScenario],
                              knowledge_graph: nx.DiGraph) -> Dict[str, Any]:
        """Assess overall scenario strengths and risks"""
        logger.info("Assessing counterfactual scenarios")

        assessment = {
            'total_scenarios': len(scenarios),
            'high_strength_scenarios': len([s for s in scenarios if s.scenario_strength > 0.7]),
            'risk_summary': {},
            'recommendations': []
        }

        # Aggregate risk assessment
        all_risks = defaultdict(list)
        for scenario in scenarios:
            for risk_type, risk_value in scenario.risk_assessment.items():
                all_risks[risk_type].append(risk_value)

        assessment['risk_summary'] = {
            risk_type: {
                'max': max(values),
                'avg': np.mean(values),
                'scenarios_affected': len(values)
            }
            for risk_type, values in all_risks.items()
        }

        # Generate recommendations
        high_risk_threshold = 0.7
        for risk_type, risk_data in assessment['risk_summary'].items():
            if risk_data['max'] > high_risk_threshold:
                assessment['recommendations'].append(
                    f"High {risk_type} risk detected - additional verification recommended"
                )

        return assessment

    async def _create_validation_framework(self, scenarios: List[CounterfactualScenario]) -> Dict[str, Any]:
        """Create validation framework for counterfactual scenarios"""
        framework = {
            'validation_strategies': [],
            'evidence_collection_plan': [],
            'verification_checkpoints': [],
            'success_criteria': []
        }

        # Validation strategies based on scenario types
        scenario_types = set()
        for scenario in scenarios:
            for hypothesis in scenario.alternative_hypotheses:
                scenario_types.add(hypothesis.hypothesis_type)

        for hypothesis_type in scenario_types:
            if hypothesis_type == HypothesisType.ALTERNATIVE_ATTRIBUTION:
                framework['validation_strategies'].append("Cross-reference attribution claims with independent sources")
            elif hypothesis_type == HypothesisType.MISSING_ENTITY:
                framework['validation_strategies'].append("Expand data collection to identify potential missing entities")

        # Evidence collection plan
        all_requirements = set()
        for scenario in scenarios:
            all_requirements.update(scenario.evidence_requirements)

        framework['evidence_collection_plan'] = list(all_requirements)

        # Verification checkpoints
        framework['verification_checkpoints'] = [
            "Initial hypothesis validation",
            "Evidence collection review",
            "Cross-scenario consistency check",
            "Final scenario assessment"
        ]

        # Success criteria
        framework['success_criteria'] = [
            "At least 80% of testable predictions verified",
            "No major logical contradictions in validated scenarios",
            "Evidence supports scenario strength assessments"
        ]

        return framework

    def _get_generation_stats(self, hypotheses: List[AlternativeHypothesis]) -> Dict[str, Any]:
        """Get statistics on hypothesis generation"""
        type_counts = Counter([h.hypothesis_type for h in hypotheses])

        return {
            'total_generated': len(hypotheses),
            'by_type': {t.value: count for t, count in type_counts.items()},
            'average_strength': np.mean([h.strength for h in hypotheses]) if hypotheses else 0.0,
            'strength_distribution': {
                'high': len([h for h in hypotheses if h.strength > 0.7]),
                'medium': len([h for h in hypotheses if 0.4 <= h.strength <= 0.7]),
                'low': len([h for h in hypotheses if h.strength < 0.4])
            }
        }

    # Utility methods for data conversion
    def _hypothesis_to_dict(self, hypothesis: AlternativeHypothesis) -> Dict[str, Any]:
        """Convert AlternativeHypothesis to dictionary"""
        return {
            'hypothesis_id': hypothesis.hypothesis_id,
            'type': hypothesis.hypothesis_type.value,
            'description': hypothesis.description,
            'strength': hypothesis.strength,
            'supporting_evidence': hypothesis.supporting_evidence,
            'contradicting_evidence': hypothesis.contradicting_evidence,
            'affected_entities': hypothesis.affected_entities,
            'altered_relationships': hypothesis.altered_relationships,
            'probability_estimate': hypothesis.probability_estimate,
            'impact_assessment': hypothesis.impact_assessment,
            'reasoning_chain': hypothesis.reasoning_chain,
            'validation_tests': hypothesis.validation_tests
        }

    def _scenario_to_dict(self, scenario: CounterfactualScenario) -> Dict[str, Any]:
        """Convert CounterfactualScenario to dictionary"""
        return {
            'scenario_id': scenario.scenario_id,
            'base_hypothesis': scenario.base_hypothesis,
            'alternative_hypotheses': [self._hypothesis_to_dict(h) for h in scenario.alternative_hypotheses],
            'scenario_strength': scenario.scenario_strength,
            'consistency_score': scenario.consistency_score,
            'evidence_requirements': scenario.evidence_requirements,
            'testable_predictions': scenario.testable_predictions,
            'risk_assessment': scenario.risk_assessment
        }

    def _test_to_dict(self, test: HypothesisTest) -> Dict[str, Any]:
        """Convert HypothesisTest to dictionary"""
        return {
            'test_id': test.test_id,
            'hypothesis_id': test.hypothesis_id,
            'test_type': test.test_type,
            'test_description': test.test_description,
            'expected_outcome': test.expected_outcome,
            'confidence_threshold': test.confidence_threshold,
            'validation_criteria': test.validation_criteria,
            'test_results': test.test_results
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for counterfactual analyzer"""
        return {
            'status': 'healthy',
            'max_hypotheses': self.max_hypotheses,
            'min_hypothesis_strength': self.min_hypothesis_strength,
            'generation_strategies': len(self.generation_strategies),
            'test_frameworks': len(self.test_frameworks),
            'validation_metrics_tracked': len(self.validation_metrics)
        }