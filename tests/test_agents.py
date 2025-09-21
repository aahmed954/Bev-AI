#!/usr/bin/env python3
"""
Test Suite for Agent Components
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.research_coordinator import ResearchCoordinator, ResearchTask
from src.agents.code_optimizer import CodeOptimizer
from src.agents.memory_manager import MemoryManager
from src.agents.tool_coordinator import ToolCoordinator
import asyncio
import tempfile
import json


class TestResearchCoordinator(unittest.TestCase):
    """Test Research Coordinator functionality"""
    
    def setUp(self):
        self.coordinator = ResearchCoordinator()
    
    def test_task_creation(self):
        """Test task creation and validation"""
        task = self.coordinator.create_task(
            query="Test research query",
            task_type="market_research",
            priority=5
        )
        
        self.assertIsNotNone(task)
        self.assertEqual(task.query, "Test research query")
        self.assertEqual(task.status, "pending")
    
    def test_task_execution(self):
        """Test async task execution"""
        task = self.coordinator.create_task(
            query="Analyze Python performance",
            task_type="code_analysis"
        )
        
        # Run async execution
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self.coordinator.execute_task(task.task_id)
        )
        
        self.assertIsNotNone(result)
        self.assertIn('status', result)
    
    def test_swarm_coordination(self):
        """Test multi-agent swarm coordination"""
        tasks = [
            self.coordinator.create_task(f"Task {i}", "market_research")
            for i in range(3)
        ]
        
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            self.coordinator.coordinate_swarm([t.task_id for t in tasks])
        )
        
        self.assertEqual(len(results), 3)


class TestCodeOptimizer(unittest.TestCase):
    """Test Code Optimizer functionality"""
    
    def setUp(self):
        self.optimizer = CodeOptimizer()
    
    def test_performance_analysis(self):
        """Test code performance analysis"""
        test_code = """
def slow_function(n):
    result = []
    for i in range(n):
        result.append(i ** 2)
    return result
"""
        
        analysis = self.optimizer.analyze_performance(test_code)
        
        self.assertIn('complexity', analysis)
        self.assertIn('bottlenecks', analysis)
        self.assertIn('memory_usage', analysis)
    
    def test_genetic_optimization(self):
        """Test genetic algorithm optimization"""
        test_code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total
"""
        
        optimized = self.optimizer.genetic_optimize(
            test_code,
            generations=5,
            population_size=10
        )
        
        self.assertIsNotNone(optimized)
        self.assertIn('optimized_code', optimized)
        self.assertIn('improvement', optimized)
    
    def test_optimization_strategies(self):
        """Test different optimization strategies"""
        strategies = self.optimizer.optimization_strategies
        
        self.assertIn('vectorization', strategies)
        self.assertIn('loop_optimization', strategies)
        self.assertIn('memory_optimization', strategies)


class TestMemoryManager(unittest.TestCase):
    """Test Memory Manager functionality"""
    
    def setUp(self):
        self.memory = MemoryManager()
    
    def test_vector_storage(self):
        """Test vector embedding storage"""
        test_data = {
            'content': 'Test research content',
            'metadata': {'source': 'test', 'timestamp': '2024-01-01'}
        }
        
        # Store embedding
        embedding_id = self.memory.store_embedding(
            test_data['content'],
            test_data['metadata']
        )
        
        self.assertIsNotNone(embedding_id)
    
    def test_graph_storage(self):
        """Test knowledge graph storage"""
        # Add entities
        entity1 = self.memory.add_entity('TestEntity1', 'organization')
        entity2 = self.memory.add_entity('TestEntity2', 'person')
        
        # Add relationship
        rel = self.memory.add_relationship(
            entity1, entity2, 'works_for'
        )
        
        self.assertIsNotNone(rel)
    
    def test_similarity_search(self):
        """Test similarity search functionality"""
        # Add test data
        texts = [
            "Machine learning algorithms",
            "Deep learning neural networks",
            "Database management systems"
        ]
        
        for text in texts:
            self.memory.store_embedding(text, {'test': True})
        
        # Search
        results = self.memory.similarity_search(
            "artificial intelligence",
            k=2
        )
        
        self.assertIsNotNone(results)
        self.assertLessEqual(len(results), 2)
    
    def test_cache_operations(self):
        """Test Redis cache operations"""
        # Store in cache
        self.memory.cache_set('test_key', {'data': 'test_value'})
        
        # Retrieve from cache
        cached = self.memory.cache_get('test_key')
        
        self.assertIsNotNone(cached)
        self.assertEqual(cached['data'], 'test_value')


class TestToolCoordinator(unittest.TestCase):
    """Test Tool Coordinator functionality"""
    
    def setUp(self):
        self.coordinator = ToolCoordinator()
    
    def test_tool_registration(self):
        """Test tool registration"""
        # Check registered tools
        tools = self.coordinator.list_tools()
        
        self.assertIn('osint_tools', tools)
        self.assertIn('exploitation_tools', tools)
        self.assertIn('enhancement_tools', tools)
    
    def test_tool_selection(self):
        """Test intelligent tool selection"""
        task = {
            'type': 'market_research',
            'requirements': ['web_scraping', 'data_analysis']
        }
        
        selected_tools = self.coordinator.select_tools(task)
        
        self.assertIsNotNone(selected_tools)
        self.assertTrue(len(selected_tools) > 0)
    
    def test_tool_execution(self):
        """Test tool execution"""
        # Create temp file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for analysis")
            test_file = f.name
        
        try:
            result = self.coordinator.execute_tool(
                'document_analyzer',
                {'file_path': test_file}
            )
            
            self.assertIsNotNone(result)
        finally:
            os.unlink(test_file)
    
    def test_parallel_execution(self):
        """Test parallel tool execution"""
        tasks = [
            {'tool': 'market_research', 'params': {'query': 'test1'}},
            {'tool': 'breach_analyzer', 'params': {'target': 'test2'}}
        ]
        
        results = self.coordinator.execute_parallel(tasks)
        
        self.assertEqual(len(results), 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete system"""
    
    def setUp(self):
        self.research = ResearchCoordinator()
        self.memory = MemoryManager()
        self.tools = ToolCoordinator()
    
    def test_end_to_end_research(self):
        """Test complete research pipeline"""
        # Create research task
        task = self.research.create_task(
            "Analyze security vulnerabilities",
            "security_research"
        )
        
        # Execute with tools
        tools = self.tools.select_tools({
            'type': 'security_research',
            'requirements': ['vulnerability_scanning']
        })
        
        # Store results in memory
        result_data = {
            'task_id': task.task_id,
            'findings': 'Test vulnerabilities found',
            'tools_used': tools
        }
        
        # Store embedding
        self.memory.store_embedding(
            json.dumps(result_data),
            {'task_id': task.task_id}
        )
        
        # Verify storage
        search_results = self.memory.similarity_search(
            "vulnerabilities",
            k=1
        )
        
        self.assertTrue(len(search_results) > 0)
    
    def test_multi_agent_collaboration(self):
        """Test multi-agent collaboration"""
        # Create multiple tasks
        tasks = []
        for i in range(3):
            task = self.research.create_task(
                f"Research task {i}",
                "market_research"
            )
            tasks.append(task)
        
        # Coordinate swarm
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            self.research.coordinate_swarm([t.task_id for t in tasks])
        )
        
        # Store all results
        for result in results:
            if 'data' in result:
                self.memory.store_embedding(
                    str(result['data']),
                    {'swarm_execution': True}
                )
        
        # Verify coordination
        self.assertEqual(len(results), 3)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestResearchCoordinator))
    suite.addTests(loader.loadTestsFromTestCase(TestCodeOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryManager))
    suite.addTests(loader.loadTestsFromTestCase(TestToolCoordinator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
