#!/usr/bin/env python3
"""
Integration Tests for Complete Bev System
"""

import unittest
import sys
import os
import time
import json
import asyncio
import tempfile
import shutil
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.research_coordinator import ResearchCoordinator
from src.agents.code_optimizer import CodeOptimizer
from src.agents.memory_manager import MemoryManager
from src.agents.tool_coordinator import ToolCoordinator
from src.intelligence.market_research import MarketIntelligence
from src.intelligence.breach_analyzer import BreachAnalyzer
from src.enhancement.watermark_research import WatermarkResearchPipeline
from src.enhancement.metadata_scrubber import MetadataScrubber
from src.pipeline.ocr_processor import DocumentOCR
from src.pipeline.document_analyzer import DocumentIntelligence


class TestSystemIntegration(unittest.TestCase):
    """Test complete system integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp(prefix='bev_test_')
        cls.coordinator = ResearchCoordinator()
        cls.memory = MemoryManager()
        cls.tools = ToolCoordinator()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_research_pipeline(self):
        """Test complete research pipeline"""
        # Create research task
        task = self.coordinator.create_task(
            query="Analyze cryptocurrency market trends",
            task_type="market_research",
            priority=8
        )
        
        # Execute research
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self.coordinator.execute_task(task.task_id)
        )
        
        # Store in memory
        if result.get('data'):
            embedding_id = self.memory.store_embedding(
                json.dumps(result['data']),
                {
                    'task_id': task.task_id,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Add to knowledge graph
            entity = self.memory.add_entity(
                f"Research_{task.task_id}",
                "research_output"
            )
            
            self.assertIsNotNone(embedding_id)
            self.assertIsNotNone(entity)
    
    def test_document_processing_pipeline(self):
        """Test document processing pipeline"""
        # Create test document
        test_doc = os.path.join(self.test_dir, 'test_doc.txt')
        with open(test_doc, 'w') as f:
            f.write("""
            Test Document for Processing
            
            This document contains sensitive information.
            Author: John Doe
            Date: 2024-01-01
            
            Content includes market analysis and security research.
            Keywords: cryptocurrency, blockchain, security, analysis.
            """)
        
        # Process document
        doc_intel = DocumentIntelligence()
        result = doc_intel.process_document(test_doc)
        
        self.assertIn('content_analysis', result)
        self.assertIn('intelligence', result)
        
        # Scrub metadata
        scrubber = MetadataScrubber()
        scrubbed = scrubber.scrub_file(test_doc)
        
        self.assertTrue(scrubbed.get('success', False))
    
    def test_enhancement_pipeline(self):
        """Test enhancement tools pipeline"""
        # Create test image with mock watermark
        test_image = os.path.join(self.test_dir, 'test_image.png')
        
        # Create simple test image
        import numpy as np
        import cv2
        
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.putText(img, 'TEST', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(test_image, img)
        
        # Test watermark detection
        pipeline = WatermarkResearchPipeline()
        result = pipeline.process_image(
            test_image,
            os.path.join(self.test_dir, 'clean_image.png')
        )
        
        self.assertIn('success', result)
        
        # Test metadata scrubbing
        scrubber = MetadataScrubber()
        scrub_result = scrubber.scrub_image(
            test_image,
            os.path.join(self.test_dir, 'scrubbed_image.png')
        )
        
        self.assertTrue(scrub_result.get('success', False))
    
    def test_multi_agent_coordination(self):
        """Test multi-agent coordination"""
        # Create multiple diverse tasks
        tasks = [
            self.coordinator.create_task(
                "Research market trends",
                "market_research",
                priority=7
            ),
            self.coordinator.create_task(
                "Analyze security vulnerabilities",
                "security_research",
                priority=9
            ),
            self.coordinator.create_task(
                "Optimize codebase performance",
                "code_analysis",
                priority=5
            )
        ]
        
        # Coordinate swarm execution
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            self.coordinator.coordinate_swarm([t.task_id for t in tasks])
        )
        
        self.assertEqual(len(results), 3)
        
        # Verify all tasks completed
        for result in results:
            self.assertIn('status', result)
            self.assertIn('task_id', result)
    
    def test_memory_integration(self):
        """Test memory system integration"""
        # Store various data types
        test_data = [
            {
                'content': 'Market analysis report on cryptocurrency',
                'type': 'report',
                'tags': ['crypto', 'market', 'analysis']
            },
            {
                'content': 'Security vulnerability assessment',
                'type': 'security',
                'tags': ['security', 'vulnerabilities', 'assessment']
            },
            {
                'content': 'Code optimization results',
                'type': 'code',
                'tags': ['optimization', 'performance', 'code']
            }
        ]
        
        # Store embeddings
        for data in test_data:
            self.memory.store_embedding(
                data['content'],
                {'type': data['type'], 'tags': data['tags']}
            )
        
        # Test similarity search
        results = self.memory.similarity_search(
            "cryptocurrency security",
            k=2
        )
        
        self.assertTrue(len(results) > 0)
        
        # Test knowledge graph
        entities = []
        for data in test_data:
            entity = self.memory.add_entity(
                data['type'],
                'document'
            )
            entities.append(entity)
        
        # Add relationships
        for i in range(len(entities) - 1):
            self.memory.add_relationship(
                entities[i],
                entities[i + 1],
                'related_to'
            )
    
    def test_tool_orchestration(self):
        """Test tool orchestration"""
        # Define complex task requiring multiple tools
        complex_task = {
            'objective': 'Complete security assessment',
            'steps': [
                {'tool': 'market_research', 'params': {'query': 'security tools market'}},
                {'tool': 'breach_analyzer', 'params': {'target': 'test_system'}},
                {'tool': 'document_analyzer', 'params': {'content': 'security report'}}
            ]
        }
        
        # Execute tools in sequence
        results = []
        for step in complex_task['steps']:
            tool_result = self.tools.execute_tool(
                step['tool'],
                step['params']
            )
            results.append(tool_result)
        
        self.assertEqual(len(results), 3)
    
    def test_error_recovery(self):
        """Test system error recovery"""
        # Test invalid task
        invalid_task = self.coordinator.create_task(
            query="",  # Invalid empty query
            task_type="invalid_type"
        )
        
        # Should handle gracefully
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self.coordinator.execute_task(invalid_task.task_id)
        )
        
        self.assertIn('status', result)
        
        # Test tool failure recovery
        try:
            self.tools.execute_tool(
                'non_existent_tool',
                {}
            )
        except Exception as e:
            # Should raise appropriate exception
            self.assertIsNotNone(e)
    
    def test_performance_optimization(self):
        """Test code optimization integration"""
        test_code = """
def inefficient_sort(arr):
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
    return arr
"""
        
        optimizer = CodeOptimizer()
        
        # Analyze performance
        analysis = optimizer.analyze_performance(test_code)
        self.assertIn('complexity', analysis)
        
        # Optimize
        optimized = optimizer.genetic_optimize(
            test_code,
            generations=3,
            population_size=5
        )
        
        self.assertIn('optimized_code', optimized)
        
        # Store optimization results
        self.memory.store_embedding(
            json.dumps(optimized),
            {'type': 'optimization', 'original_complexity': analysis['complexity']}
        )
    
    def test_concurrent_operations(self):
        """Test concurrent system operations"""
        import concurrent.futures
        
        def run_task(task_type, query):
            task = self.coordinator.create_task(query, task_type)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self.coordinator.execute_task(task.task_id)
            )
        
        # Run multiple tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(run_task, 'market_research', f'Query {i}')
                for i in range(3)
            ]
            
            results = [f.result(timeout=30) for f in futures]
        
        self.assertEqual(len(results), 3)


class TestSystemPerformance(unittest.TestCase):
    """Test system performance metrics"""
    
    def test_response_time(self):
        """Test system response times"""
        coordinator = ResearchCoordinator()
        
        start_time = time.time()
        task = coordinator.create_task(
            "Quick test query",
            "market_research"
        )
        creation_time = time.time() - start_time
        
        # Task creation should be fast
        self.assertLess(creation_time, 0.1)
        
        # Test execution time
        start_time = time.time()
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            coordinator.execute_task(task.task_id)
        )
        execution_time = time.time() - start_time
        
        # Log performance metrics
        print(f"Task creation: {creation_time:.4f}s")
        print(f"Task execution: {execution_time:.4f}s")
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency"""
        memory = MemoryManager()
        
        # Store large dataset
        large_data = "x" * 10000  # 10KB string
        
        start_time = time.time()
        for i in range(100):
            memory.store_embedding(
                f"{large_data}_{i}",
                {'index': i}
            )
        storage_time = time.time() - start_time
        
        # Test retrieval speed
        start_time = time.time()
        results = memory.similarity_search("test", k=10)
        search_time = time.time() - start_time
        
        print(f"Storage time (100 items): {storage_time:.4f}s")
        print(f"Search time: {search_time:.4f}s")
        
        # Should complete reasonably fast
        self.assertLess(storage_time, 10)
        self.assertLess(search_time, 1)
    
    def test_concurrent_scaling(self):
        """Test system scaling with concurrent load"""
        coordinator = ResearchCoordinator()
        
        # Create many tasks
        tasks = [
            coordinator.create_task(f"Task {i}", "market_research")
            for i in range(10)
        ]
        
        start_time = time.time()
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            coordinator.coordinate_swarm([t.task_id for t in tasks])
        )
        total_time = time.time() - start_time
        
        print(f"Parallel execution (10 tasks): {total_time:.4f}s")
        print(f"Average per task: {total_time/10:.4f}s")
        
        self.assertEqual(len(results), 10)
        
        # Should show parallelization benefit
        self.assertLess(total_time, 10)  # Should be faster than sequential


def run_integration_tests():
    """Run all integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSystemIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSystemPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 60)
    print("BEV SYSTEM INTEGRATION TESTS")
    print("=" * 60)
    
    success = run_integration_tests()
    
    print("=" * 60)
    print(f"Tests {'PASSED' if success else 'FAILED'}")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
