#!/usr/bin/env python3
"""
Knowledge Worker for ORACLE1
Knowledge synthesis, graph reasoning, and semantic analysis
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import structlog
from celery import Task
from celery_app import app
from gensim import corpora, models
from neo4j import GraphDatabase
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import redis
import spacy

# Configure structured logging
logger = structlog.get_logger("knowledge_worker")

class KnowledgeTask(BaseModel):
    """Knowledge processing task model"""
    task_id: str
    task_type: str
    data: Dict[str, Any]
    parameters: Dict[str, Any] = {}
    output_format: str = "json"
    cache_duration: int = 3600

class KnowledgeNode(BaseModel):
    """Knowledge graph node"""
    id: str
    type: str
    properties: Dict[str, Any]
    embeddings: Optional[List[float]] = None

class KnowledgeRelation(BaseModel):
    """Knowledge graph relationship"""
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any] = {}
    confidence: float = 1.0

class KnowledgeGraph(BaseModel):
    """Knowledge graph structure"""
    nodes: List[KnowledgeNode]
    relations: List[KnowledgeRelation]
    metadata: Dict[str, Any] = {}

class TopicModel(BaseModel):
    """Topic modeling result"""
    topics: List[Dict[str, Any]]
    document_topics: List[Dict[str, Any]]
    topic_words: Dict[int, List[str]]
    coherence_score: float

class KnowledgeSynthesizer:
    """Knowledge synthesis and reasoning engine"""

    def __init__(self):
        self.redis_client = redis.Redis(host='redis', port=6379, db=3)
        self.neo4j_driver = None
        self.sentence_transformer = None
        self.nlp = None
        self.setup_models()
        self.setup_neo4j()

    def setup_models(self):
        """Initialize NLP models and transformers"""
        try:
            logger.info("Loading knowledge processing models")

            # Load sentence transformer for embeddings
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

            # Load spaCy model
            self.nlp = spacy.load('en_core_web_sm')

            logger.info("Knowledge processing models loaded")

        except Exception as e:
            logger.error("Failed to load models", error=str(e))
            raise

    def setup_neo4j(self):
        """Setup Neo4j connection for knowledge storage"""
        try:
            self.neo4j_driver = GraphDatabase.driver(
                "bolt://neo4j:7687",
                auth=("neo4j", "admin123")
            )

            # Test connection
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")

            logger.info("Neo4j connection established")

        except Exception as e:
            logger.error("Failed to connect to Neo4j", error=str(e))
            # Continue without Neo4j for now
            self.neo4j_driver = None

    def synthesize_knowledge(self, task: KnowledgeTask) -> Dict[str, Any]:
        """Synthesize knowledge from multiple sources"""
        try:
            logger.info("Starting knowledge synthesis", task_id=task.task_id)
            start_time = time.time()

            sources = task.data.get('sources', [])
            synthesis_type = task.parameters.get('synthesis_type', 'comprehensive')

            # Process each source
            processed_sources = []
            for source in sources:
                processed = self._process_knowledge_source(source)
                processed_sources.append(processed)

            # Synthesize knowledge
            if synthesis_type == 'comprehensive':
                result = self._comprehensive_synthesis(processed_sources, task.parameters)
            elif synthesis_type == 'topic_based':
                result = self._topic_based_synthesis(processed_sources, task.parameters)
            elif synthesis_type == 'graph_based':
                result = self._graph_based_synthesis(processed_sources, task.parameters)
            else:
                result = self._comprehensive_synthesis(processed_sources, task.parameters)

            # Add metadata
            result['metadata'] = {
                'task_id': task.task_id,
                'synthesis_type': synthesis_type,
                'sources_count': len(sources),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

            # Cache result
            self._cache_result(task.task_id, result, task.cache_duration)

            return result

        except Exception as e:
            logger.error("Knowledge synthesis failed", task_id=task.task_id, error=str(e))
            raise

    def _process_knowledge_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual knowledge source"""
        try:
            source_type = source.get('type', 'text')
            content = source.get('content', '')

            if source_type == 'text':
                return self._process_text_source(content, source)
            elif source_type == 'document':
                return self._process_document_source(source)
            elif source_type == 'database':
                return self._process_database_source(source)
            else:
                return self._process_text_source(content, source)

        except Exception as e:
            logger.error("Source processing failed", source_id=source.get('id'), error=str(e))
            return {'entities': [], 'concepts': [], 'relations': [], 'embeddings': []}

    def _process_text_source(self, text: str, source: Dict[str, Any]) -> Dict[str, Any]:
        """Process text source for knowledge extraction"""
        try:
            # Extract entities and concepts
            doc = self.nlp(text)

            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })

            # Extract noun phrases as concepts
            concepts = []
            for chunk in doc.noun_chunks:
                concepts.append({
                    'text': chunk.text,
                    'root': chunk.root.text,
                    'pos': chunk.root.pos_
                })

            # Generate embeddings
            embeddings = self.sentence_transformer.encode([text])[0].tolist()

            # Extract relations (simple dependency parsing)
            relations = []
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ in ['nsubj', 'dobj'] and token.head.pos_ == 'VERB':
                        relations.append({
                            'subject': token.text,
                            'predicate': token.head.text,
                            'object': None,
                            'sentence': sent.text
                        })

            return {
                'source_id': source.get('id'),
                'entities': entities,
                'concepts': concepts,
                'relations': relations,
                'embeddings': embeddings,
                'text_length': len(text),
                'metadata': source.get('metadata', {})
            }

        except Exception as e:
            logger.error("Text processing failed", error=str(e))
            return {'entities': [], 'concepts': [], 'relations': [], 'embeddings': []}

    def _process_document_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Process document source (from database or file)"""
        # Placeholder for document processing
        # In real implementation, would fetch and process document
        return self._process_text_source(source.get('content', ''), source)

    def _process_database_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Process database source for knowledge extraction"""
        # Placeholder for database processing
        # In real implementation, would query database and extract knowledge
        return {'entities': [], 'concepts': [], 'relations': [], 'embeddings': []}

    def _comprehensive_synthesis(self, sources: List[Dict], parameters: Dict) -> Dict[str, Any]:
        """Perform comprehensive knowledge synthesis"""
        try:
            # Combine all entities
            all_entities = []
            all_concepts = []
            all_relations = []
            all_embeddings = []

            for source in sources:
                all_entities.extend(source.get('entities', []))
                all_concepts.extend(source.get('concepts', []))
                all_relations.extend(source.get('relations', []))
                if source.get('embeddings'):
                    all_embeddings.append(source['embeddings'])

            # Deduplicate and cluster entities
            unique_entities = self._deduplicate_entities(all_entities)

            # Cluster concepts
            clustered_concepts = self._cluster_concepts(all_concepts)

            # Build knowledge graph
            knowledge_graph = self._build_knowledge_graph(
                unique_entities,
                clustered_concepts,
                all_relations
            )

            # Generate insights
            insights = self._generate_insights(knowledge_graph, all_embeddings)

            return {
                'synthesis_type': 'comprehensive',
                'entities': unique_entities,
                'concepts': clustered_concepts,
                'relations': all_relations,
                'knowledge_graph': knowledge_graph,
                'insights': insights,
                'statistics': {
                    'total_entities': len(unique_entities),
                    'total_concepts': len(clustered_concepts),
                    'total_relations': len(all_relations)
                }
            }

        except Exception as e:
            logger.error("Comprehensive synthesis failed", error=str(e))
            raise

    def _topic_based_synthesis(self, sources: List[Dict], parameters: Dict) -> Dict[str, Any]:
        """Perform topic-based knowledge synthesis"""
        try:
            # Collect all text content
            texts = []
            for source in sources:
                # Combine entities and concepts as text
                text_content = []
                for entity in source.get('entities', []):
                    text_content.append(entity.get('text', ''))
                for concept in source.get('concepts', []):
                    text_content.append(concept.get('text', ''))

                if text_content:
                    texts.append(' '.join(text_content))

            if not texts:
                return {'topics': [], 'document_topics': [], 'topic_words': {}}

            # Perform topic modeling
            topic_model = self._perform_topic_modeling(texts, parameters)

            # Map topics back to sources
            topic_mapping = self._map_topics_to_sources(topic_model, sources)

            return {
                'synthesis_type': 'topic_based',
                'topic_model': topic_model.dict() if hasattr(topic_model, 'dict') else topic_model,
                'topic_mapping': topic_mapping,
                'sources_analyzed': len(sources)
            }

        except Exception as e:
            logger.error("Topic-based synthesis failed", error=str(e))
            raise

    def _graph_based_synthesis(self, sources: List[Dict], parameters: Dict) -> Dict[str, Any]:
        """Perform graph-based knowledge synthesis"""
        try:
            # Build comprehensive knowledge graph
            graph = nx.Graph()

            # Add nodes from all sources
            node_id = 0
            for source in sources:
                for entity in source.get('entities', []):
                    graph.add_node(node_id,
                                 type='entity',
                                 text=entity.get('text'),
                                 label=entity.get('label'),
                                 source_id=source.get('source_id'))
                    node_id += 1

                for concept in source.get('concepts', []):
                    graph.add_node(node_id,
                                 type='concept',
                                 text=concept.get('text'),
                                 source_id=source.get('source_id'))
                    node_id += 1

            # Add edges from relations
            # Simplified edge creation based on text similarity
            nodes = list(graph.nodes(data=True))
            for i, (node1_id, node1_data) in enumerate(nodes):
                for j, (node2_id, node2_data) in enumerate(nodes[i+1:], i+1):
                    similarity = self._calculate_text_similarity(
                        node1_data.get('text', ''),
                        node2_data.get('text', '')
                    )
                    if similarity > 0.7:  # Threshold for connection
                        graph.add_edge(node1_id, node2_id, weight=similarity)

            # Analyze graph structure
            graph_analysis = self._analyze_graph_structure(graph)

            # Store in Neo4j if available
            if self.neo4j_driver:
                self._store_knowledge_graph(graph)

            return {
                'synthesis_type': 'graph_based',
                'graph_statistics': graph_analysis,
                'node_count': graph.number_of_nodes(),
                'edge_count': graph.number_of_edges(),
                'components': nx.number_connected_components(graph)
            }

        except Exception as e:
            logger.error("Graph-based synthesis failed", error=str(e))
            raise

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities using text similarity"""
        try:
            if not entities:
                return []

            unique_entities = []
            seen_texts = set()

            for entity in entities:
                text = entity.get('text', '').lower().strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    unique_entities.append(entity)

            return unique_entities

        except Exception as e:
            logger.error("Entity deduplication failed", error=str(e))
            return entities

    def _cluster_concepts(self, concepts: List[Dict]) -> List[Dict]:
        """Cluster similar concepts together"""
        try:
            if not concepts:
                return []

            # Extract concept texts
            concept_texts = [concept.get('text', '') for concept in concepts]

            if len(concept_texts) < 2:
                return concepts

            # Generate embeddings
            embeddings = self.sentence_transformer.encode(concept_texts)

            # Perform clustering
            n_clusters = min(10, len(concepts) // 2 + 1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Group concepts by cluster
            clustered_concepts = []
            for i, concept in enumerate(concepts):
                concept['cluster'] = int(cluster_labels[i])
                clustered_concepts.append(concept)

            return clustered_concepts

        except Exception as e:
            logger.error("Concept clustering failed", error=str(e))
            return concepts

    def _build_knowledge_graph(self, entities: List[Dict], concepts: List[Dict], relations: List[Dict]) -> Dict[str, Any]:
        """Build knowledge graph from extracted elements"""
        try:
            nodes = []
            edges = []

            # Create nodes from entities
            for i, entity in enumerate(entities):
                nodes.append({
                    'id': f"entity_{i}",
                    'type': 'entity',
                    'label': entity.get('label', 'UNKNOWN'),
                    'text': entity.get('text', ''),
                    'properties': entity
                })

            # Create nodes from concepts
            for i, concept in enumerate(concepts):
                nodes.append({
                    'id': f"concept_{i}",
                    'type': 'concept',
                    'text': concept.get('text', ''),
                    'cluster': concept.get('cluster', 0),
                    'properties': concept
                })

            # Create edges from relations
            for i, relation in enumerate(relations):
                if relation.get('subject') and relation.get('predicate'):
                    edges.append({
                        'id': f"relation_{i}",
                        'source': relation.get('subject'),
                        'target': relation.get('object', ''),
                        'relation': relation.get('predicate'),
                        'properties': relation
                    })

            return {
                'nodes': nodes,
                'edges': edges,
                'node_count': len(nodes),
                'edge_count': len(edges)
            }

        except Exception as e:
            logger.error("Knowledge graph building failed", error=str(e))
            return {'nodes': [], 'edges': [], 'node_count': 0, 'edge_count': 0}

    def _generate_insights(self, knowledge_graph: Dict, embeddings: List) -> List[Dict]:
        """Generate insights from knowledge graph"""
        try:
            insights = []

            # Node distribution insight
            node_types = {}
            for node in knowledge_graph.get('nodes', []):
                node_type = node.get('type', 'unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1

            insights.append({
                'type': 'distribution',
                'title': 'Node Type Distribution',
                'data': node_types
            })

            # Connectivity insight
            edge_count = knowledge_graph.get('edge_count', 0)
            node_count = knowledge_graph.get('node_count', 0)
            if node_count > 0:
                connectivity = edge_count / node_count
                insights.append({
                    'type': 'metric',
                    'title': 'Graph Connectivity',
                    'value': connectivity,
                    'description': f'Average {connectivity:.2f} connections per node'
                })

            # Semantic similarity insight
            if len(embeddings) > 1:
                embeddings_array = np.array(embeddings)
                similarity_matrix = cosine_similarity(embeddings_array)
                avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])

                insights.append({
                    'type': 'metric',
                    'title': 'Semantic Similarity',
                    'value': float(avg_similarity),
                    'description': f'Average semantic similarity: {avg_similarity:.3f}'
                })

            return insights

        except Exception as e:
            logger.error("Insight generation failed", error=str(e))
            return []

    def _perform_topic_modeling(self, texts: List[str], parameters: Dict) -> Dict[str, Any]:
        """Perform topic modeling on text collection"""
        try:
            if not texts:
                return {}

            # Prepare documents
            processed_texts = []
            for text in texts:
                doc = self.nlp(text)
                tokens = [token.lemma_.lower() for token in doc
                         if not token.is_stop and not token.is_punct and token.is_alpha]
                processed_texts.append(tokens)

            # Create dictionary and corpus
            dictionary = corpora.Dictionary(processed_texts)
            corpus = [dictionary.doc2bow(text) for text in processed_texts]

            # Train LDA model
            num_topics = parameters.get('num_topics', 5)
            lda_model = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                per_word_topics=True
            )

            # Extract topics
            topics = []
            for idx, topic in lda_model.print_topics():
                topics.append({
                    'id': idx,
                    'words': topic,
                    'weight': 1.0 / num_topics
                })

            # Get document-topic distributions
            document_topics = []
            for i, doc in enumerate(corpus):
                doc_topics = lda_model.get_document_topics(doc)
                document_topics.append({
                    'document_id': i,
                    'topics': [(topic_id, prob) for topic_id, prob in doc_topics]
                })

            return {
                'topics': topics,
                'document_topics': document_topics,
                'num_topics': num_topics,
                'coherence_score': 0.5  # Placeholder - would calculate actual coherence
            }

        except Exception as e:
            logger.error("Topic modeling failed", error=str(e))
            return {}

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            if not text1 or not text2:
                return 0.0

            embeddings = self.sentence_transformer.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)

        except Exception as e:
            logger.error("Similarity calculation failed", error=str(e))
            return 0.0

    def _analyze_graph_structure(self, graph: nx.Graph) -> Dict[str, Any]:
        """Analyze graph structure and properties"""
        try:
            analysis = {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'components': nx.number_connected_components(graph)
            }

            if graph.number_of_nodes() > 0:
                # Calculate centrality measures
                degree_centrality = nx.degree_centrality(graph)
                analysis['max_degree_centrality'] = max(degree_centrality.values())
                analysis['avg_degree_centrality'] = sum(degree_centrality.values()) / len(degree_centrality)

                # Find most central nodes
                top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                analysis['top_central_nodes'] = [{'node': node, 'centrality': centrality}
                                               for node, centrality in top_central]

            return analysis

        except Exception as e:
            logger.error("Graph analysis failed", error=str(e))
            return {}

    def _store_knowledge_graph(self, graph: nx.Graph):
        """Store knowledge graph in Neo4j"""
        try:
            if not self.neo4j_driver:
                return

            with self.neo4j_driver.session() as session:
                # Clear existing graph
                session.run("MATCH (n) DETACH DELETE n")

                # Add nodes
                for node_id, node_data in graph.nodes(data=True):
                    session.run(
                        """
                        CREATE (n:KnowledgeNode {
                            id: $node_id,
                            type: $type,
                            text: $text
                        })
                        """,
                        node_id=str(node_id),
                        type=node_data.get('type', 'unknown'),
                        text=node_data.get('text', '')
                    )

                # Add edges
                for source, target, edge_data in graph.edges(data=True):
                    session.run(
                        """
                        MATCH (a:KnowledgeNode {id: $source})
                        MATCH (b:KnowledgeNode {id: $target})
                        CREATE (a)-[:RELATED {weight: $weight}]->(b)
                        """,
                        source=str(source),
                        target=str(target),
                        weight=edge_data.get('weight', 1.0)
                    )

            logger.info("Knowledge graph stored in Neo4j")

        except Exception as e:
            logger.error("Failed to store knowledge graph", error=str(e))

    def _map_topics_to_sources(self, topic_model: Dict, sources: List[Dict]) -> Dict[str, Any]:
        """Map topics back to original sources"""
        try:
            topic_mapping = {}

            document_topics = topic_model.get('document_topics', [])
            for i, doc_topics in enumerate(document_topics):
                if i < len(sources):
                    source_id = sources[i].get('source_id', f'source_{i}')
                    topic_mapping[source_id] = doc_topics.get('topics', [])

            return topic_mapping

        except Exception as e:
            logger.error("Topic mapping failed", error=str(e))
            return {}

    def _cache_result(self, task_id: str, result: Dict[str, Any], duration: int):
        """Cache processing result"""
        try:
            self.redis_client.setex(
                f"knowledge_result:{task_id}",
                duration,
                json.dumps(result, default=str)
            )
            logger.info("Result cached", task_id=task_id)

        except Exception as e:
            logger.warning("Failed to cache result", task_id=task_id, error=str(e))

    def get_cached_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        try:
            cached_data = self.redis_client.get(f"knowledge_result:{task_id}")
            if cached_data:
                return json.loads(cached_data)
            return None

        except Exception as e:
            logger.warning("Failed to get cached result", task_id=task_id, error=str(e))
            return None

# Initialize knowledge synthesizer
knowledge_synthesizer = KnowledgeSynthesizer()

class KnowledgeSynthesisTask(Task):
    """Custom Celery task for knowledge synthesis"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("Knowledge synthesis task failed",
                    task_id=task_id,
                    exception=str(exc),
                    traceback=str(einfo))

@app.task(bind=True, base=KnowledgeSynthesisTask, queue='knowledge_synthesis')
def synthesize_knowledge(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize knowledge from multiple sources"""
    try:
        logger.info("Processing knowledge synthesis task", task_id=self.request.id)

        task = KnowledgeTask(**task_data)

        # Check cache first
        cached_result = knowledge_synthesizer.get_cached_result(task.task_id)
        if cached_result:
            logger.info("Returning cached result", task_id=task.task_id)
            return cached_result

        # Run synthesis
        result = knowledge_synthesizer.synthesize_knowledge(task)
        return result

    except Exception as e:
        logger.error("Knowledge synthesis failed", task_id=self.request.id, error=str(e))
        raise self.retry(exc=e, countdown=60, max_retries=3)

@app.task(bind=True, base=KnowledgeSynthesisTask, queue='knowledge_synthesis')
def extract_knowledge_graph(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract knowledge graph from sources"""
    try:
        logger.info("Processing knowledge graph extraction", task_id=self.request.id)

        task_data = {
            'task_id': self.request.id,
            'task_type': 'graph_extraction',
            'data': {'sources': sources},
            'parameters': {'synthesis_type': 'graph_based'}
        }

        task = KnowledgeTask(**task_data)
        result = knowledge_synthesizer.synthesize_knowledge(task)

        return result

    except Exception as e:
        logger.error("Knowledge graph extraction failed", task_id=self.request.id, error=str(e))
        raise self.retry(exc=e, countdown=60, max_retries=3)

@app.task(bind=True, base=KnowledgeSynthesisTask, queue='knowledge_synthesis')
def perform_topic_analysis(self, documents: List[str], num_topics: int = 5) -> Dict[str, Any]:
    """Perform topic analysis on document collection"""
    try:
        logger.info("Processing topic analysis", task_id=self.request.id)

        task_data = {
            'task_id': self.request.id,
            'task_type': 'topic_analysis',
            'data': {'sources': [{'type': 'text', 'content': doc, 'id': f'doc_{i}'}
                               for i, doc in enumerate(documents)]},
            'parameters': {'synthesis_type': 'topic_based', 'num_topics': num_topics}
        }

        task = KnowledgeTask(**task_data)
        result = knowledge_synthesizer.synthesize_knowledge(task)

        return result

    except Exception as e:
        logger.error("Topic analysis failed", task_id=self.request.id, error=str(e))
        raise self.retry(exc=e, countdown=60, max_retries=3)

@app.task(bind=True, base=KnowledgeSynthesisTask, queue='knowledge_synthesis')
def semantic_similarity_analysis(self, texts: List[str]) -> Dict[str, Any]:
    """Analyze semantic similarity between texts"""
    try:
        logger.info("Processing semantic similarity analysis", task_id=self.request.id)

        if not texts or len(texts) < 2:
            return {'error': 'Need at least 2 texts for similarity analysis'}

        # Generate embeddings
        embeddings = knowledge_synthesizer.sentence_transformer.encode(texts)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Find most similar pairs
        similar_pairs = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = float(similarity_matrix[i][j])
                similar_pairs.append({
                    'text1_index': i,
                    'text2_index': j,
                    'text1': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                    'text2': texts[j][:100] + '...' if len(texts[j]) > 100 else texts[j],
                    'similarity': similarity
                })

        # Sort by similarity
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)

        return {
            'task_id': self.request.id,
            'similarity_matrix': similarity_matrix.tolist(),
            'most_similar_pairs': similar_pairs[:10],
            'average_similarity': float(np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])),
            'text_count': len(texts)
        }

    except Exception as e:
        logger.error("Semantic similarity analysis failed", task_id=self.request.id, error=str(e))
        raise

if __name__ == "__main__":
    # Test knowledge synthesis functionality
    test_sources = [
        {
            'id': 'source_1',
            'type': 'text',
            'content': 'Artificial intelligence is transforming the healthcare industry.',
            'metadata': {'domain': 'healthcare'}
        },
        {
            'id': 'source_2',
            'type': 'text',
            'content': 'Machine learning algorithms can predict patient outcomes.',
            'metadata': {'domain': 'healthcare'}
        }
    ]

    print("Knowledge synthesis worker ready for knowledge processing")