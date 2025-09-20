

class TextAnalyzer:
    """Advanced text analysis and NLP processing"""
    
    def __init__(self):
        self.nlp = None
        self.sentiment_analyzer = None
        self.ner_model = None
        self.load_models()
    
    def load_models(self):
        """Load NLP models"""
        try:
            # Load spaCy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found, downloading...")
            # SECURITY: Replace with subprocess.run() - os.system("python -m spacy download en_core_web_sm")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                logger.error("Failed to load spaCy model")
        
        # Load sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
        except:
            logger.warning("Sentiment model not available")
        
        # Load NER model
        try:
            self.ner_model = pipeline("ner", aggregation_strategy="simple")
        except:
            logger.warning("NER model not available")
    
    def analyze(self, text: str) -> Dict:
        """Comprehensive text analysis"""
        result = {
            'statistics': self.get_statistics(text),
            'readability': self.get_readability(text),
            'entities': self.extract_entities(text),
            'keywords': self.extract_keywords(text),
            'sentiment': self.analyze_sentiment(text),
            'topics': self.extract_topics(text),
            'summary': self.generate_summary(text)
        }
        
        return result
    
    def get_statistics(self, text: str) -> Dict:
        """Get text statistics"""
        words = text.split()
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / len(words) if words else 0
        }
    
    def get_readability(self, text: str) -> Dict:
        """Calculate readability scores"""
        try:
            return {
                'flesch_reading_ease': flesch_reading_ease(text),
                'flesch_kincaid_grade': flesch_kincaid_grade(text),
                'interpretation': self._interpret_readability(flesch_reading_ease(text))
            }
        except:
            return {'error': 'Could not calculate readability'}
    
    def _interpret_readability(self, score: float) -> str:
        """Interpret Flesch reading ease score"""
        if score >= 90:
            return "Very Easy (5th grade)"
        elif score >= 80:
            return "Easy (6th grade)"
        elif score >= 70:
            return "Fairly Easy (7th grade)"
        elif score >= 60:
            return "Standard (8-9th grade)"
        elif score >= 50:
            return "Fairly Difficult (10-12th grade)"
        elif score >= 30:
            return "Difficult (College)"
        else:
            return "Very Difficult (Graduate)"
    
    def extract_entities(self, text: str) -> Dict:
        """Extract named entities"""
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'money': [],
            'misc': []
        }
        
        if self.nlp:
            doc = self.nlp(text[:1000000])  # Limit for performance
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities['persons'].append(ent.text)
                elif ent.label_ == "ORG":
                    entities['organizations'].append(ent.text)
                elif ent.label_ in ["GPE", "LOC"]:
                    entities['locations'].append(ent.text)
                elif ent.label_ == "DATE":
                    entities['dates'].append(ent.text)
                elif ent.label_ == "MONEY":
                    entities['money'].append(ent.text)
                else:
                    entities['misc'].append((ent.text, ent.label_))
        
        # Remove duplicates
        for key in entities:
            if key != 'misc':
                entities[key] = list(set(entities[key]))
        
        return entities
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF"""
        try:
            # Tokenize and clean
            words = text.lower().split()
            words = [w for w in words if len(w) > 3 and w.isalnum()]
            
            if not words:
                return []
            
            # Calculate TF-IDF
            vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([' '.join(words)])
            
            # Get feature names and scores
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Sort by score
            keywords = [(feature_names[i], scores[i]) 
                       for i in np.argsort(scores)[::-1][:num_keywords]]
            
            return keywords
        except:
            return []
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        if not self.sentiment_analyzer:
            return {'error': 'Sentiment analyzer not available'}
        
        try:
            # Split into chunks for long texts
            max_length = 512
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            sentiments = []
            for chunk in chunks[:10]:  # Limit to first 10 chunks
                result = self.sentiment_analyzer(chunk)
                if result:
                    sentiments.append(result[0])
            
            # Aggregate results
            if sentiments:
                pos_score = np.mean([s['score'] for s in sentiments if s['label'] == 'POSITIVE'])
                neg_score = np.mean([s['score'] for s in sentiments if s['label'] == 'NEGATIVE'])
                
                return {
                    'overall': 'POSITIVE' if pos_score > neg_score else 'NEGATIVE',
                    'positive_score': float(pos_score) if not np.isnan(pos_score) else 0,
                    'negative_score': float(neg_score) if not np.isnan(neg_score) else 0,
                    'chunks_analyzed': len(sentiments)
                }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
        
        return {'error': 'Analysis failed'}
    
    def extract_topics(self, text: str, num_topics: int = 5) -> List[Dict]:
        """Extract topics using LDA"""
        try:
            # Prepare documents
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) < num_topics:
                return []
            
            # Vectorize
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            doc_term_matrix = vectorizer.fit_transform(sentences)
            
            # LDA
            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda.fit(doc_term_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words[:5],
                    'weight': float(topic[top_words_idx].mean())
                })
            
            return topics
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            return []
    
    def generate_summary(self, text: str, ratio: float = 0.2) -> str:
        """Generate text summary"""
        try:
            sentences = text.split('.')
            sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 20]
            
            if len(sentences) < 3:
                return text
            
            # Simple extractive summarization
            # Score sentences by position and keyword density
            scores = []
            keywords = set(word.lower() for word, _ in self.extract_keywords(text))
            
            for i, sent in enumerate(sentences):
                # Position score (first and last sentences are important)
                position_score = 1.0 if i == 0 or i == len(sentences) - 1 else 0.5
                
                # Keyword score
                words = sent.lower().split()
                keyword_score = sum(1 for w in words if w in keywords) / len(words) if words else 0
                
                # Combined score
                score = position_score + keyword_score
                scores.append((score, sent))
            
            # Sort by score and select top sentences
            scores.sort(reverse=True)
            num_sentences = max(1, int(len(sentences) * ratio))
            summary_sentences = [sent for _, sent in scores[:num_sentences]]
            
            # Reorder by original position
            summary_sentences = [s for s in sentences if s in summary_sentences]
            
            return ' '.join(summary_sentences)
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return text[:500] + "..." if len(text) > 500 else text


class DocumentIntelligence:
    """Advanced document intelligence extraction"""
    
    def __init__(self):
        self.parser = DocumentParser()
        self.analyzer = TextAnalyzer()
        
    def process_document(self, file_path: str) -> Dict:
        """Complete document processing pipeline"""
        logger.info(f"Processing document: {file_path}")
        
        # Parse document
        parsed = self.parser.parse(file_path)
        
        if 'error' in parsed:
            return parsed
        
        # Analyze content
        content = parsed.get('content', '')
        if content:
            analysis = self.analyzer.analyze(content)
        else:
            analysis = {}
        
        # Combine results
        result = {
            'file': file_path,
            'type': parsed.get('type'),
            'metadata': parsed.get('metadata', {}),
            'structure': self._extract_structure(parsed),
            'content_analysis': analysis,
            'intelligence': self._extract_intelligence(parsed, analysis)
        }
        
        return result
    
    def _extract_structure(self, parsed: Dict) -> Dict:
        """Extract document structure"""
        structure = {
            'sections': [],
            'tables': len(parsed.get('tables', [])),
            'images': len(parsed.get('images', [])),
            'links': len(parsed.get('links', []))
        }
        
        # Extract section hierarchy
        if 'headers' in parsed:
            structure['sections'] = parsed['headers']
        elif 'pages' in parsed:
            structure['pages'] = len(parsed['pages'])
        
        return structure
    
    def _extract_intelligence(self, parsed: Dict, analysis: Dict) -> Dict:
        """Extract actionable intelligence from document"""
        intelligence = {
            'key_points': [],
            'action_items': [],
            'risks': [],
            'opportunities': [],
            'relationships': []
        }
        
        content = parsed.get('content', '')
        
        # Extract key points from summary
        if 'summary' in analysis:
            intelligence['key_points'] = analysis['summary'].split('.')[:3]
        
        # Find action items
        action_patterns = [
            r'(?:TODO|FIXME|ACTION|REQUIRED|MUST|SHALL):\s*([^\.]+)',
            r'(?:need to|have to|must|should)\s+([^\.]+)',
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            intelligence['action_items'].extend(matches[:5])
        
        # Find risks
        risk_keywords = ['risk', 'threat', 'vulnerability', 'danger', 'warning']
        risk_sentences = []
        
        for sentence in content.split('.'):
            if any(keyword in sentence.lower() for keyword in risk_keywords):
                risk_sentences.append(sentence.strip())
        
        intelligence['risks'] = risk_sentences[:5]
        
        # Extract relationships from entities
        if 'entities' in analysis:
            entities = analysis['entities']
            
            # Create relationship graph
            if entities.get('persons') and entities.get('organizations'):
                for person in entities['persons'][:3]:
                    for org in entities['organizations'][:3]:
                        intelligence['relationships'].append({
                            'entity1': person,
                            'entity2': org,
                            'type': 'person-organization'
                        })
        
        return intelligence
    
    def compare_documents(self, file1: str, file2: str) -> Dict:
        """Compare two documents"""
        doc1 = self.process_document(file1)
        doc2 = self.process_document(file2)
        
        comparison = {
            'similarity': self._calculate_similarity(
                doc1.get('content_analysis', {}).get('keywords', []),
                doc2.get('content_analysis', {}).get('keywords', [])
            ),
            'common_entities': self._find_common_entities(
                doc1.get('content_analysis', {}).get('entities', {}),
                doc2.get('content_analysis', {}).get('entities', {})
            ),
            'differences': self._find_differences(doc1, doc2)
        }
        
        return comparison
    
    def _calculate_similarity(self, keywords1: List, keywords2: List) -> float:
        """Calculate keyword similarity"""
        if not keywords1 or not keywords2:
            return 0.0
        
        words1 = set(k[0] for k in keywords1)
        words2 = set(k[0] for k in keywords2)
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union else 0.0
    
    def _find_common_entities(self, entities1: Dict, entities2: Dict) -> Dict:
        """Find common entities between documents"""
        common = {}
        
        for entity_type in entities1:
            if entity_type in entities2:
                set1 = set(entities1[entity_type]) if isinstance(entities1[entity_type], list) else set()
                set2 = set(entities2[entity_type]) if isinstance(entities2[entity_type], list) else set()
                
                common_entities = set1 & set2
                if common_entities:
                    common[entity_type] = list(common_entities)
        
        return common
    
    def _find_differences(self, doc1: Dict, doc2: Dict) -> Dict:
        """Find key differences between documents"""
        differences = {
            'sentiment': {
                'doc1': doc1.get('content_analysis', {}).get('sentiment', {}).get('overall'),
                'doc2': doc2.get('content_analysis', {}).get('sentiment', {}).get('overall')
            },
            'length': {
                'doc1': doc1.get('content_analysis', {}).get('statistics', {}).get('word_count', 0),
                'doc2': doc2.get('content_analysis', {}).get('statistics', {}).get('word_count', 0)
            },
            'readability': {
                'doc1': doc1.get('content_analysis', {}).get('readability', {}).get('flesch_reading_ease', 0),
                'doc2': doc2.get('content_analysis', {}).get('readability', {}).get('flesch_reading_ease', 0)
            }
        }
        
        return differences
    
    def batch_analyze(self, directory: str, pattern: str = '*') -> List[Dict]:
        """Batch analyze documents"""
        import glob
        
        files = glob.glob(os.path.join(directory, pattern))
        results = []
        
        for file_path in files:
            try:
                result = self.process_document(file_path)
                results.append(result)
                logger.info(f"Processed {file_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append({'file': file_path, 'error': str(e)})
        
        return results


if __name__ == "__main__":
    # Initialize
    intelligence = DocumentIntelligence()
    
    # Example usage
    # result = intelligence.process_document("document.pdf")
    # print(json.dumps(result, indent=2))
    
    # Compare documents
    # comparison = intelligence.compare_documents("doc1.pdf", "doc2.pdf")
    # print(f"Similarity: {comparison['similarity']:.2%}")
    
    # Batch processing
    # results = intelligence.batch_analyze("./documents/", "*.pdf")
    # print(f"Analyzed {len(results)} documents")
    
    print("Document Intelligence System initialized - Extracting knowledge from everything!")
