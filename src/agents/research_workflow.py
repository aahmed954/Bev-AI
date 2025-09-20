"""
Research Workflow Engine for BEV OSINT Framework
Implements 5-phase research workflow with automated progression
"""

import asyncio
import logging
import time
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import json
import aiohttp
import numpy as np
from collections import defaultdict, Counter
import spacy
from textblob import TextBlob
import networkx as nx

logger = logging.getLogger(__name__)

class WorkflowPhase(Enum):
    """Research workflow phases"""
    INFORMATION_GATHERING = "information_gathering"
    ENTITY_EXTRACTION = "entity_extraction"
    RELATIONSHIP_MAPPING = "relationship_mapping"
    PATTERN_ANALYSIS = "pattern_analysis"
    INSIGHT_GENERATION = "insight_generation"

@dataclass
class EntityExtraction:
    """Container for extracted entities"""
    name: str
    entity_type: str
    confidence: float
    context: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    frequency: int = 1
    importance_score: float = 0.0

@dataclass
class RelationshipExtraction:
    """Container for extracted relationships"""
    source: str
    target: str
    relation_type: str
    confidence: float
    evidence: str
    context: str
    strength: float = 0.0

@dataclass
class PatternDiscovery:
    """Container for discovered patterns"""
    pattern_type: str
    description: str
    confidence: float
    frequency: int
    entities_involved: List[str]
    significance: float
    evidence: List[str] = field(default_factory=list)

@dataclass
class WorkflowState:
    """State tracking for workflow execution"""
    current_phase: WorkflowPhase
    completed_phases: Set[WorkflowPhase] = field(default_factory=set)
    phase_results: Dict[str, Any] = field(default_factory=dict)
    entities: List[EntityExtraction] = field(default_factory=list)
    relationships: List[RelationshipExtraction] = field(default_factory=list)
    patterns: List[PatternDiscovery] = field(default_factory=list)
    processing_metrics: Dict[str, float] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)

class ResearchWorkflowEngine:
    """
    Research workflow engine implementing automated 5-phase analysis
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_db_endpoint = config.get('vector_db_endpoint', 'http://172.30.0.44:8000')
        self.entity_confidence_threshold = config.get('entity_confidence_threshold', 0.6)
        self.relationship_confidence_threshold = config.get('relationship_confidence_threshold', 0.5)
        self.pattern_significance_threshold = config.get('pattern_significance_threshold', 0.7)

        # Initialize NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using fallback entity extraction")
            self.nlp = None

        # Entity type mappings
        self.entity_types = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',
            'DATE': 'date',
            'TIME': 'time',
            'MONEY': 'monetary',
            'PERCENT': 'percentage',
            'FACILITY': 'facility',
            'PRODUCT': 'product',
            'EVENT': 'event',
            'WORK_OF_ART': 'work_of_art',
            'LAW': 'law',
            'LANGUAGE': 'language'
        }

        # Relationship patterns
        self.relationship_patterns = [
            (r'(\w+)\s+(?:works at|employed by|member of)\s+(\w+)', 'employed_by'),
            (r'(\w+)\s+(?:owns|founded|created)\s+(\w+)', 'owns'),
            (r'(\w+)\s+(?:located in|based in|from)\s+(\w+)', 'located_in'),
            (r'(\w+)\s+(?:connected to|related to|associated with)\s+(\w+)', 'associated_with'),
            (r'(\w+)\s+(?:communicated with|contacted|met)\s+(\w+)', 'communicated_with'),
            (r'(\w+)\s+(?:transferred|sent|gave)\s+.+?\s+to\s+(\w+)', 'transferred_to'),
            (r'(\w+)\s+(?:received|got|obtained)\s+.+?\s+from\s+(\w+)', 'received_from'),
        ]

        # Pattern detection rules
        self.pattern_rules = {
            'financial_flow': {
                'keywords': ['payment', 'transfer', 'money', 'fund', 'investment', 'transaction'],
                'entities': ['MONEY', 'ORG', 'PERSON'],
                'min_frequency': 3
            },
            'communication_network': {
                'keywords': ['call', 'email', 'message', 'contact', 'communication', 'meeting'],
                'entities': ['PERSON', 'ORG'],
                'min_frequency': 2
            },
            'location_cluster': {
                'keywords': ['location', 'address', 'place', 'building', 'office'],
                'entities': ['GPE', 'FACILITY'],
                'min_frequency': 2
            },
            'temporal_sequence': {
                'keywords': ['before', 'after', 'during', 'when', 'then', 'next'],
                'entities': ['DATE', 'TIME', 'EVENT'],
                'min_frequency': 3
            },
            'organizational_hierarchy': {
                'keywords': ['boss', 'manager', 'director', 'ceo', 'executive', 'reports to'],
                'entities': ['PERSON', 'ORG'],
                'min_frequency': 2
            }
        }

    async def explore_context(self, chunks: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initial exploration of context chunks

        Args:
            chunks: List of text chunks to analyze
            metadata: Additional context metadata

        Returns:
            Dictionary with exploration results
        """
        logger.info("Starting context exploration")

        workflow_state = WorkflowState(current_phase=WorkflowPhase.INFORMATION_GATHERING)

        try:
            # Execute workflow phases
            await self._execute_information_gathering(chunks, workflow_state)
            await self._execute_entity_extraction(chunks, workflow_state)
            await self._execute_relationship_mapping(chunks, workflow_state)
            await self._execute_pattern_analysis(chunks, workflow_state)
            await self._execute_insight_generation(workflow_state)

            # Compile results
            return {
                'entities': [self._entity_to_dict(e) for e in workflow_state.entities],
                'relationships': [self._relationship_to_dict(r) for r in workflow_state.relationships],
                'topics': self._extract_topics(chunks),
                'patterns': [self._pattern_to_dict(p) for p in workflow_state.patterns],
                'processing_metrics': workflow_state.processing_metrics,
                'phase_results': workflow_state.phase_results
            }

        except Exception as e:
            logger.error(f"Error in context exploration: {str(e)}")
            workflow_state.error_log.append(f"Exploration error: {str(e)}")
            raise

    async def _execute_information_gathering(self, chunks: List[str], state: WorkflowState):
        """Phase 1: Information gathering and preprocessing"""
        start_time = time.time()
        logger.info("Executing information gathering phase")

        # Basic text statistics
        total_chars = sum(len(chunk) for chunk in chunks)
        total_words = sum(len(chunk.split()) for chunk in chunks)
        unique_words = set()
        for chunk in chunks:
            unique_words.update(chunk.lower().split())

        # Language detection and sentiment analysis
        language_scores = []
        sentiment_scores = []

        for chunk in chunks[:5]:  # Sample first 5 chunks for efficiency
            try:
                blob = TextBlob(chunk)
                language_scores.append(blob.detect_language())
                sentiment_scores.append(blob.sentiment.polarity)
            except:
                pass

        # Information density calculation
        info_density = len(unique_words) / max(total_words, 1)

        state.phase_results['information_gathering'] = {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'total_words': total_words,
            'unique_words': len(unique_words),
            'information_density': info_density,
            'detected_languages': list(set(language_scores)),
            'average_sentiment': np.mean(sentiment_scores) if sentiment_scores else 0.0,
            'processing_time': time.time() - start_time
        }

        state.completed_phases.add(WorkflowPhase.INFORMATION_GATHERING)
        state.processing_metrics['information_gathering_time'] = time.time() - start_time

    async def _execute_entity_extraction(self, chunks: List[str], state: WorkflowState):
        """Phase 2: Advanced entity extraction"""
        start_time = time.time()
        logger.info("Executing entity extraction phase")

        entity_counter = Counter()
        entity_contexts = defaultdict(list)

        for chunk_idx, chunk in enumerate(chunks):
            # Extract entities using spaCy if available
            if self.nlp:
                doc = self.nlp(chunk)
                for ent in doc.ents:
                    if ent.label_ in self.entity_types:
                        entity_name = ent.text.strip()
                        entity_type = self.entity_types[ent.label_]

                        entity_counter[entity_name] += 1
                        entity_contexts[entity_name].append({
                            'chunk': chunk_idx,
                            'context': chunk[max(0, ent.start_char-50):ent.end_char+50],
                            'confidence': 0.8  # spaCy confidence placeholder
                        })

            # Fallback regex-based extraction
            else:
                await self._regex_entity_extraction(chunk, chunk_idx, entity_counter, entity_contexts)

        # Create EntityExtraction objects
        for entity_name, frequency in entity_counter.items():
            if frequency >= 1:  # Minimum frequency threshold
                contexts = entity_contexts[entity_name]
                avg_confidence = np.mean([c['confidence'] for c in contexts])

                if avg_confidence >= self.entity_confidence_threshold:
                    entity = EntityExtraction(
                        name=entity_name,
                        entity_type=self._infer_entity_type(entity_name),
                        confidence=avg_confidence,
                        context=contexts[0]['context'],
                        frequency=frequency,
                        importance_score=self._calculate_entity_importance(entity_name, frequency, contexts)
                    )

                    # Add evidence
                    entity.evidence = [c['context'] for c in contexts[:3]]  # Top 3 contexts

                    state.entities.append(entity)

        # Sort entities by importance
        state.entities.sort(key=lambda x: x.importance_score, reverse=True)

        state.phase_results['entity_extraction'] = {
            'total_entities_found': len(entity_counter),
            'high_confidence_entities': len(state.entities),
            'entity_types': Counter([e.entity_type for e in state.entities]),
            'processing_time': time.time() - start_time
        }

        state.completed_phases.add(WorkflowPhase.ENTITY_EXTRACTION)
        state.processing_metrics['entity_extraction_time'] = time.time() - start_time

    async def _regex_entity_extraction(self, chunk: str, chunk_idx: int,
                                     entity_counter: Counter, entity_contexts: Dict):
        """Fallback regex-based entity extraction"""
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, chunk)
        for email in emails:
            entity_counter[email] += 1
            entity_contexts[email].append({
                'chunk': chunk_idx,
                'context': chunk,
                'confidence': 0.9
            })

        # Phone numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, chunk)
        for phone in phones:
            entity_counter[phone] += 1
            entity_contexts[phone].append({
                'chunk': chunk_idx,
                'context': chunk,
                'confidence': 0.85
            })

        # URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, chunk)
        for url in urls:
            entity_counter[url] += 1
            entity_contexts[url].append({
                'chunk': chunk_idx,
                'context': chunk,
                'confidence': 0.95
            })

        # IP addresses
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.findall(ip_pattern, chunk)
        for ip in ips:
            entity_counter[ip] += 1
            entity_contexts[ip].append({
                'chunk': chunk_idx,
                'context': chunk,
                'confidence': 0.9
            })

    def _infer_entity_type(self, entity_name: str) -> str:
        """Infer entity type from name"""
        entity_lower = entity_name.lower()

        if '@' in entity_name:
            return 'email'
        elif re.match(r'\d{3}[-.]?\d{3}[-.]?\d{4}', entity_name):
            return 'phone'
        elif entity_name.startswith('http'):
            return 'url'
        elif re.match(r'(?:[0-9]{1,3}\.){3}[0-9]{1,3}', entity_name):
            return 'ip_address'
        elif entity_name.istitle() and len(entity_name.split()) <= 3:
            return 'person'
        elif any(keyword in entity_lower for keyword in ['corp', 'inc', 'llc', 'ltd', 'company']):
            return 'organization'
        else:
            return 'unknown'

    def _calculate_entity_importance(self, entity_name: str, frequency: int,
                                   contexts: List[Dict]) -> float:
        """Calculate entity importance score"""
        # Base score from frequency
        frequency_score = min(1.0, frequency / 10)

        # Context diversity score
        unique_chunks = len(set([c['chunk'] for c in contexts]))
        diversity_score = min(1.0, unique_chunks / 5)

        # Type-based importance
        entity_type = self._infer_entity_type(entity_name)
        type_weights = {
            'person': 0.9,
            'organization': 0.9,
            'email': 0.8,
            'phone': 0.8,
            'url': 0.7,
            'ip_address': 0.8,
            'unknown': 0.5
        }
        type_score = type_weights.get(entity_type, 0.5)

        return (frequency_score * 0.4 + diversity_score * 0.3 + type_score * 0.3)

    async def _execute_relationship_mapping(self, chunks: List[str], state: WorkflowState):
        """Phase 3: Relationship mapping between entities"""
        start_time = time.time()
        logger.info("Executing relationship mapping phase")

        entity_names = [e.name for e in state.entities]
        relationships_found = []

        for chunk in chunks:
            # Pattern-based relationship extraction
            for pattern, relation_type in self.relationship_patterns:
                matches = re.finditer(pattern, chunk, re.IGNORECASE)
                for match in matches:
                    source = match.group(1)
                    target = match.group(2)

                    # Check if entities exist in our entity list
                    if any(source.lower() in e.name.lower() for e in state.entities) and \
                       any(target.lower() in e.name.lower() for e in state.entities):

                        relationship = RelationshipExtraction(
                            source=source,
                            target=target,
                            relation_type=relation_type,
                            confidence=0.7,
                            evidence=match.group(0),
                            context=chunk[max(0, match.start()-100):match.end()+100],
                            strength=0.8
                        )
                        relationships_found.append(relationship)

            # Co-occurrence based relationships
            await self._extract_cooccurrence_relationships(chunk, entity_names, relationships_found)

        # Filter and deduplicate relationships
        filtered_relationships = self._filter_relationships(relationships_found)
        state.relationships.extend(filtered_relationships)

        state.phase_results['relationship_mapping'] = {
            'total_relationships_found': len(relationships_found),
            'filtered_relationships': len(filtered_relationships),
            'relationship_types': Counter([r.relation_type for r in filtered_relationships]),
            'processing_time': time.time() - start_time
        }

        state.completed_phases.add(WorkflowPhase.RELATIONSHIP_MAPPING)
        state.processing_metrics['relationship_mapping_time'] = time.time() - start_time

    async def _extract_cooccurrence_relationships(self, chunk: str, entity_names: List[str],
                                                relationships_found: List[RelationshipExtraction]):
        """Extract relationships based on entity co-occurrence"""
        chunk_lower = chunk.lower()
        present_entities = []

        for entity_name in entity_names:
            if entity_name.lower() in chunk_lower:
                present_entities.append(entity_name)

        # Create co-occurrence relationships for entities in same chunk
        if len(present_entities) >= 2:
            for i, entity1 in enumerate(present_entities):
                for entity2 in present_entities[i+1:]:
                    # Calculate distance between entities in text
                    pos1 = chunk_lower.find(entity1.lower())
                    pos2 = chunk_lower.find(entity2.lower())
                    distance = abs(pos1 - pos2)

                    # Close proximity suggests relationship
                    if distance < 200:  # Within 200 characters
                        confidence = max(0.3, 1.0 - (distance / 500))

                        relationship = RelationshipExtraction(
                            source=entity1,
                            target=entity2,
                            relation_type='co_mentioned',
                            confidence=confidence,
                            evidence=f"Co-mentioned within {distance} characters",
                            context=chunk[max(0, min(pos1, pos2)-50):max(pos1, pos2)+50],
                            strength=confidence
                        )
                        relationships_found.append(relationship)

    def _filter_relationships(self, relationships: List[RelationshipExtraction]) -> List[RelationshipExtraction]:
        """Filter and deduplicate relationships"""
        filtered = []
        seen_pairs = set()

        for rel in relationships:
            # Create canonical pair representation
            pair = tuple(sorted([rel.source.lower(), rel.target.lower()]))
            pair_key = (pair, rel.relation_type)

            if pair_key not in seen_pairs and rel.confidence >= self.relationship_confidence_threshold:
                seen_pairs.add(pair_key)
                filtered.append(rel)

        return filtered

    async def _execute_pattern_analysis(self, chunks: List[str], state: WorkflowState):
        """Phase 4: Pattern analysis and discovery"""
        start_time = time.time()
        logger.info("Executing pattern analysis phase")

        patterns_found = []

        for pattern_name, rule in self.pattern_rules.items():
            pattern_result = await self._analyze_pattern(pattern_name, rule, chunks, state)
            if pattern_result:
                patterns_found.append(pattern_result)

        # Add temporal patterns
        temporal_patterns = await self._analyze_temporal_patterns(chunks, state)
        patterns_found.extend(temporal_patterns)

        # Add communication patterns
        communication_patterns = await self._analyze_communication_patterns(chunks, state)
        patterns_found.extend(communication_patterns)

        # Filter significant patterns
        significant_patterns = [p for p in patterns_found
                              if p.significance >= self.pattern_significance_threshold]

        state.patterns.extend(significant_patterns)

        state.phase_results['pattern_analysis'] = {
            'total_patterns_found': len(patterns_found),
            'significant_patterns': len(significant_patterns),
            'pattern_types': Counter([p.pattern_type for p in significant_patterns]),
            'processing_time': time.time() - start_time
        }

        state.completed_phases.add(WorkflowPhase.PATTERN_ANALYSIS)
        state.processing_metrics['pattern_analysis_time'] = time.time() - start_time

    async def _analyze_pattern(self, pattern_name: str, rule: Dict[str, Any],
                             chunks: List[str], state: WorkflowState) -> Optional[PatternDiscovery]:
        """Analyze specific pattern type"""
        keywords = rule['keywords']
        required_entities = rule['entities']
        min_frequency = rule['min_frequency']

        # Count keyword occurrences
        keyword_count = 0
        relevant_entities = []
        evidence_chunks = []

        for chunk in chunks:
            chunk_lower = chunk.lower()
            chunk_keywords = sum(1 for keyword in keywords if keyword in chunk_lower)

            if chunk_keywords > 0:
                keyword_count += chunk_keywords
                evidence_chunks.append(chunk)

                # Find relevant entities in this chunk
                for entity in state.entities:
                    if entity.entity_type in required_entities and entity.name.lower() in chunk_lower:
                        relevant_entities.append(entity.name)

        if keyword_count >= min_frequency and len(set(relevant_entities)) >= 2:
            confidence = min(0.95, keyword_count / (min_frequency * 3))
            significance = self._calculate_pattern_significance(
                keyword_count, len(set(relevant_entities)), len(evidence_chunks)
            )

            return PatternDiscovery(
                pattern_type=pattern_name,
                description=f"Pattern involving {len(set(relevant_entities))} entities with {keyword_count} relevant mentions",
                confidence=confidence,
                frequency=keyword_count,
                entities_involved=list(set(relevant_entities)),
                significance=significance,
                evidence=evidence_chunks[:3]  # Top 3 evidence chunks
            )

        return None

    async def _analyze_temporal_patterns(self, chunks: List[str],
                                       state: WorkflowState) -> List[PatternDiscovery]:
        """Analyze temporal patterns in the data"""
        temporal_indicators = ['before', 'after', 'during', 'when', 'then', 'next', 'previous', 'later']
        patterns = []

        sequence_count = 0
        for chunk in chunks:
            chunk_lower = chunk.lower()
            temporal_words = sum(1 for word in temporal_indicators if word in chunk_lower)
            sequence_count += temporal_words

        if sequence_count >= 3:
            patterns.append(PatternDiscovery(
                pattern_type='temporal_sequence',
                description=f"Temporal sequence pattern with {sequence_count} temporal indicators",
                confidence=min(0.9, sequence_count / 10),
                frequency=sequence_count,
                entities_involved=[],
                significance=self._calculate_pattern_significance(sequence_count, 0, len(chunks))
            ))

        return patterns

    async def _analyze_communication_patterns(self, chunks: List[str],
                                            state: WorkflowState) -> List[PatternDiscovery]:
        """Analyze communication patterns"""
        communication_verbs = ['called', 'emailed', 'texted', 'messaged', 'contacted', 'spoke', 'communicated']
        patterns = []

        communication_count = 0
        involved_entities = set()

        for chunk in chunks:
            chunk_lower = chunk.lower()
            comm_words = sum(1 for word in communication_verbs if word in chunk_lower)
            communication_count += comm_words

            if comm_words > 0:
                # Find entities involved in communication
                for entity in state.entities:
                    if entity.entity_type in ['person', 'organization'] and entity.name.lower() in chunk_lower:
                        involved_entities.add(entity.name)

        if communication_count >= 2 and len(involved_entities) >= 2:
            patterns.append(PatternDiscovery(
                pattern_type='communication_network',
                description=f"Communication network involving {len(involved_entities)} entities",
                confidence=min(0.85, communication_count / 5),
                frequency=communication_count,
                entities_involved=list(involved_entities),
                significance=self._calculate_pattern_significance(
                    communication_count, len(involved_entities), len(chunks)
                )
            ))

        return patterns

    def _calculate_pattern_significance(self, frequency: int, entity_count: int, chunk_count: int) -> float:
        """Calculate pattern significance score"""
        frequency_score = min(1.0, frequency / 10)
        entity_score = min(1.0, entity_count / 5)
        coverage_score = min(1.0, chunk_count / 10)

        return (frequency_score * 0.5 + entity_score * 0.3 + coverage_score * 0.2)

    async def _execute_insight_generation(self, state: WorkflowState):
        """Phase 5: Insight generation and synthesis"""
        start_time = time.time()
        logger.info("Executing insight generation phase")

        insights = []

        # Entity-based insights
        if state.entities:
            top_entities = state.entities[:5]
            insights.append(f"Top entities: {', '.join([e.name for e in top_entities])}")

        # Relationship insights
        if state.relationships:
            relationship_types = Counter([r.relation_type for r in state.relationships])
            most_common_rel = relationship_types.most_common(1)[0]
            insights.append(f"Most common relationship type: {most_common_rel[0]} ({most_common_rel[1]} instances)")

        # Pattern insights
        if state.patterns:
            pattern_types = Counter([p.pattern_type for p in state.patterns])
            insights.append(f"Discovered patterns: {', '.join(pattern_types.keys())}")

        # Network analysis insights
        if len(state.entities) >= 3 and len(state.relationships) >= 2:
            insights.append("Complex network structure detected with multiple interconnected entities")

        state.phase_results['insight_generation'] = {
            'insights': insights,
            'insight_count': len(insights),
            'processing_time': time.time() - start_time
        }

        state.completed_phases.add(WorkflowPhase.INSIGHT_GENERATION)
        state.processing_metrics['insight_generation_time'] = time.time() - start_time

    async def deep_analyze(self, key_entities: List[Dict[str, Any]], chunks: List[str]) -> Dict[str, Any]:
        """
        Deep analysis of key entities

        Args:
            key_entities: List of key entities to analyze in depth
            chunks: Text chunks for analysis

        Returns:
            Dictionary with detailed analysis results
        """
        logger.info("Starting deep analysis")

        detailed_entities = []

        for entity in key_entities:
            entity_name = entity.get('name', '')
            detailed_analysis = await self._deep_analyze_entity(entity_name, chunks)
            detailed_entities.append(detailed_analysis)

        # Pattern analysis for key entities
        patterns = await self._analyze_entity_patterns(detailed_entities, chunks)

        # Evidence strength assessment
        evidence_strength = self._assess_evidence_strength(detailed_entities)

        return {
            'detailed_entities': detailed_entities,
            'patterns': patterns,
            'evidence_strength': evidence_strength
        }

    async def _deep_analyze_entity(self, entity_name: str, chunks: List[str]) -> Dict[str, Any]:
        """Deep analysis of a single entity"""
        entity_data = {
            'name': entity_name,
            'attributes': {},
            'evidence': [],
            'significance': 0.0,
            'context_analysis': {}
        }

        relevant_chunks = []
        total_mentions = 0

        for chunk in chunks:
            if entity_name.lower() in chunk.lower():
                relevant_chunks.append(chunk)
                total_mentions += chunk.lower().count(entity_name.lower())

        # Extract attributes from context
        attributes = await self._extract_entity_attributes(entity_name, relevant_chunks)
        entity_data['attributes'] = attributes

        # Collect evidence
        entity_data['evidence'] = relevant_chunks[:5]  # Top 5 relevant chunks

        # Calculate significance
        entity_data['significance'] = self._calculate_entity_significance(
            total_mentions, len(relevant_chunks), len(chunks)
        )

        # Context analysis
        entity_data['context_analysis'] = {
            'total_mentions': total_mentions,
            'relevant_chunks': len(relevant_chunks),
            'context_diversity': len(set(relevant_chunks)) / max(len(relevant_chunks), 1)
        }

        return entity_data

    async def _extract_entity_attributes(self, entity_name: str,
                                       contexts: List[str]) -> Dict[str, Any]:
        """Extract attributes for an entity from its contexts"""
        attributes = {}

        for context in contexts:
            # Simple attribute extraction patterns
            context_lower = context.lower()
            entity_lower = entity_name.lower()

            # Look for descriptive patterns
            if 'ceo' in context_lower and entity_lower in context_lower:
                attributes['role'] = 'CEO'
            elif 'director' in context_lower and entity_lower in context_lower:
                attributes['role'] = 'Director'
            elif 'manager' in context_lower and entity_lower in context_lower:
                attributes['role'] = 'Manager'

            # Location attributes
            location_pattern = rf'{re.escape(entity_lower)}.{{0,50}}(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
            location_match = re.search(location_pattern, context, re.IGNORECASE)
            if location_match:
                attributes['location'] = location_match.group(1)

        return attributes

    def _calculate_entity_significance(self, mentions: int, relevant_chunks: int, total_chunks: int) -> float:
        """Calculate entity significance score"""
        mention_score = min(1.0, mentions / 10)
        coverage_score = relevant_chunks / max(total_chunks, 1)

        return (mention_score * 0.6 + coverage_score * 0.4)

    async def _analyze_entity_patterns(self, entities: List[Dict[str, Any]],
                                     chunks: List[str]) -> List[Dict[str, Any]]:
        """Analyze patterns involving the key entities"""
        patterns = []

        # Interaction patterns
        entity_names = [e['name'] for e in entities]
        interaction_count = 0

        for chunk in chunks:
            chunk_entities = [name for name in entity_names if name.lower() in chunk.lower()]
            if len(chunk_entities) >= 2:
                interaction_count += 1

        if interaction_count >= 2:
            patterns.append({
                'type': 'entity_interaction',
                'description': f'{len(entity_names)} key entities frequently co-occur',
                'frequency': interaction_count,
                'confidence': min(0.9, interaction_count / 5)
            })

        return patterns

    def _assess_evidence_strength(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall evidence strength"""
        total_evidence = sum(len(e.get('evidence', [])) for e in entities)
        avg_significance = np.mean([e.get('significance', 0) for e in entities])

        return {
            'total_evidence_pieces': total_evidence,
            'average_significance': avg_significance,
            'evidence_quality': 'high' if avg_significance > 0.7 else 'medium' if avg_significance > 0.4 else 'low'
        }

    async def cross_verify(self, exploration: Dict[str, Any], deep_diving: Dict[str, Any],
                          chunks: List[str]) -> Dict[str, Any]:
        """
        Cross-verify findings across different analysis phases

        Args:
            exploration: Results from exploration phase
            deep_diving: Results from deep diving phase
            chunks: Original text chunks

        Returns:
            Dictionary with verification results
        """
        logger.info("Starting cross-verification")

        # Verify entities across phases
        exploration_entities = {e['name']: e for e in exploration.get('entities', [])}
        detailed_entities = {e['name']: e for e in deep_diving.get('detailed_entities', [])}

        verified_entities = []
        conflicts = []

        for entity_name in exploration_entities:
            if entity_name in detailed_entities:
                # Cross-verify attributes
                exploration_entity = exploration_entities[entity_name]
                detailed_entity = detailed_entities[entity_name]

                verification_result = await self._verify_entity_consistency(
                    exploration_entity, detailed_entity, chunks
                )
                verified_entities.append(verification_result)

                # Check for conflicts
                if verification_result.get('conflicts'):
                    conflicts.extend(verification_result['conflicts'])

        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(verified_entities, conflicts)

        return {
            'verified_entities': verified_entities,
            'conflicts': conflicts,
            'consistency_score': consistency_score
        }

    async def _verify_entity_consistency(self, exploration_entity: Dict[str, Any],
                                       detailed_entity: Dict[str, Any],
                                       chunks: List[str]) -> Dict[str, Any]:
        """Verify consistency between exploration and detailed analysis"""
        entity_name = exploration_entity['name']
        conflicts = []

        # Check confidence consistency
        exploration_confidence = exploration_entity.get('confidence', 0)
        detailed_significance = detailed_entity.get('significance', 0)

        if abs(exploration_confidence - detailed_significance) > 0.3:
            conflicts.append({
                'type': 'confidence_mismatch',
                'description': f'Confidence mismatch for {entity_name}',
                'exploration_value': exploration_confidence,
                'detailed_value': detailed_significance
            })

        # Check attribute consistency
        exploration_attrs = exploration_entity.get('attributes', {})
        detailed_attrs = detailed_entity.get('attributes', {})

        for attr_name, attr_value in exploration_attrs.items():
            if attr_name in detailed_attrs and detailed_attrs[attr_name] != attr_value:
                conflicts.append({
                    'type': 'attribute_conflict',
                    'description': f'Attribute conflict for {entity_name}.{attr_name}',
                    'exploration_value': attr_value,
                    'detailed_value': detailed_attrs[attr_name]
                })

        return {
            'entity_name': entity_name,
            'verified': len(conflicts) == 0,
            'conflicts': conflicts,
            'final_confidence': (exploration_confidence + detailed_significance) / 2
        }

    def _calculate_consistency_score(self, verified_entities: List[Dict[str, Any]],
                                   conflicts: List[Dict[str, Any]]) -> float:
        """Calculate overall consistency score"""
        if not verified_entities:
            return 0.5

        verified_count = sum(1 for e in verified_entities if e['verified'])
        verification_rate = verified_count / len(verified_entities)

        conflict_penalty = min(0.4, len(conflicts) * 0.1)

        return max(0.1, verification_rate - conflict_penalty)

    def _extract_topics(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """Extract main topics from chunks"""
        # Simple keyword-based topic extraction
        topics = []

        # Combine all chunks
        combined_text = ' '.join(chunks).lower()

        # Define topic keywords
        topic_keywords = {
            'financial': ['money', 'payment', 'fund', 'investment', 'bank', 'transaction', 'financial'],
            'communication': ['call', 'email', 'message', 'contact', 'phone', 'communication'],
            'business': ['company', 'corporation', 'business', 'organization', 'enterprise'],
            'legal': ['law', 'legal', 'court', 'judge', 'lawyer', 'attorney', 'case'],
            'technology': ['computer', 'software', 'technology', 'digital', 'internet', 'data'],
            'location': ['address', 'location', 'place', 'building', 'office', 'city', 'country']
        }

        for topic_name, keywords in topic_keywords.items():
            keyword_count = sum(combined_text.count(keyword) for keyword in keywords)
            if keyword_count >= 3:
                confidence = min(0.95, keyword_count / 20)
                topics.append({
                    'name': topic_name,
                    'confidence': confidence,
                    'frequency': keyword_count,
                    'keywords': keywords
                })

        return sorted(topics, key=lambda x: x['confidence'], reverse=True)

    # Utility methods for data conversion
    def _entity_to_dict(self, entity: EntityExtraction) -> Dict[str, Any]:
        """Convert EntityExtraction to dictionary"""
        return {
            'name': entity.name,
            'type': entity.entity_type,
            'confidence': entity.confidence,
            'context': entity.context,
            'attributes': entity.attributes,
            'evidence': entity.evidence,
            'frequency': entity.frequency,
            'importance_score': entity.importance_score
        }

    def _relationship_to_dict(self, relationship: RelationshipExtraction) -> Dict[str, Any]:
        """Convert RelationshipExtraction to dictionary"""
        return {
            'source': relationship.source,
            'target': relationship.target,
            'relation': relationship.relation_type,
            'confidence': relationship.confidence,
            'evidence': relationship.evidence,
            'context': relationship.context,
            'strength': relationship.strength
        }

    def _pattern_to_dict(self, pattern: PatternDiscovery) -> Dict[str, Any]:
        """Convert PatternDiscovery to dictionary"""
        return {
            'type': pattern.pattern_type,
            'description': pattern.description,
            'confidence': pattern.confidence,
            'frequency': pattern.frequency,
            'entities_involved': pattern.entities_involved,
            'significance': pattern.significance,
            'evidence': pattern.evidence
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for workflow engine"""
        return {
            'status': 'healthy',
            'nlp_model': 'loaded' if self.nlp else 'fallback_mode',
            'entity_confidence_threshold': self.entity_confidence_threshold,
            'relationship_confidence_threshold': self.relationship_confidence_threshold,
            'pattern_significance_threshold': self.pattern_significance_threshold
        }