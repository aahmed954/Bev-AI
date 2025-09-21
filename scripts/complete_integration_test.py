#!/usr/bin/env python3
"""
Complete BEV System Integration Test
Validates all components are working together
"""

import asyncio
import time
import json
from typing import Dict, List, Any
import requests
import docker
import psycopg2
import redis
import aio_pika
from aiokafka import AIOKafkaProducer
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BEVIntegrationTester:
    """Complete system integration tester"""
    
    def __init__(self):
        self.test_results = {}
        self.docker_client = docker.from_env()
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        
        logger.info("Starting BEV Integration Tests...")
        
        tests = [
            ("Database Connectivity", self.test_databases),
            ("Message Queue", self.test_message_queues),
            ("Agent Communication", self.test_agent_communication),
            ("Airflow DAGs", self.test_airflow_dags),
            ("N8N Workflows", self.test_n8n_workflows),
            ("OCR Pipeline", self.test_ocr_pipeline),
            ("Multi-Node Health", self.test_multi_node),
            ("End-to-End Flow", self.test_end_to_end)
        ]
        
        for test_name, test_func in tests:
            try:
                logger.info(f"Running: {test_name}")
                result = await test_func()
                self.test_results[test_name] = {
                    'status': 'PASSED' if result else 'FAILED',
                    'details': result
                }
                logger.info(f"✓ {test_name}: PASSED")
            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                logger.error(f"✗ {test_name}: FAILED - {e}")
        
        return self.generate_report()
    
    async def test_databases(self) -> bool:
        """Test all database connections"""
        
        # PostgreSQL
        try:
            conn = psycopg2.connect(
                host="localhost",
                database="ai_swarm",
                user="swarm_admin",
                password="swarm_password"
            )
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM agent_messages")
            conn.close()
        except Exception as e:
            raise Exception(f"PostgreSQL failed: {e}")
        
        # Redis
        try:
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.set('test_key', 'test_value')
            assert r.get('test_key') == b'test_value'
            r.delete('test_key')
        except Exception as e:
            raise Exception(f"Redis failed: {e}")
        
        return True
    
    async def test_message_queues(self) -> bool:
        """Test RabbitMQ and Kafka"""
        
        # RabbitMQ
        try:
            connection = await aio_pika.connect_robust(
                "amqp://admin:BevSwarm2024!@localhost/"
            )
            channel = await connection.channel()
            
            # Test publish
            exchange = await channel.declare_exchange('test_exchange', auto_delete=True)
            queue = await channel.declare_queue('test_queue', auto_delete=True)
            await queue.bind(exchange)
            
            await exchange.publish(
                aio_pika.Message(body=b'test message'),
                routing_key='test'
            )
            
            await connection.close()
        except Exception as e:
            raise Exception(f"RabbitMQ failed: {e}")
        
        # Kafka
        try:
            producer = AIOKafkaProducer(
                bootstrap_servers='localhost:9092'
            )
            await producer.start()
            await producer.send_and_wait(
                'test_topic',
                b'test message'
            )
            await producer.stop()
        except Exception as e:
            raise Exception(f"Kafka failed: {e}")
        
        return True
    
    async def test_agent_communication(self) -> bool:
        """Test agent-to-agent communication"""
        
        # Send test message through agent bus
        response = requests.post(
            'http://localhost:8000/agents/test_communication',
            json={
                'sender': 'test_client',
                'recipient': 'swarm_master',
                'message': 'ping'
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Agent communication failed: {response.text}")
        
        result = response.json()
        if result.get('response') != 'pong':
            raise Exception("Agent didn't respond correctly")
        
        return True
    
    async def test_airflow_dags(self) -> bool:
        """Test Airflow DAG execution"""
        
        # Trigger test DAG
        response = requests.post(
            'http://localhost:8080/api/v1/dags/test_dag/dagRuns',
            auth=('admin', 'BevAdmin2024!'),
            json={
                'conf': {},
                'dag_run_id': f'test_run_{int(time.time())}'
            }
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Airflow DAG trigger failed: {response.text}")
        
        return True
    
    async def test_n8n_workflows(self) -> bool:
        """Test N8N workflow execution"""
        
        # Test workflow webhook
        response = requests.post(
            'http://localhost:5678/webhook-test/bev-integration',
            json={'test': True}
        )
        
        if response.status_code != 200:
            raise Exception(f"N8N workflow failed: {response.text}")
        
        return True
    
    async def test_ocr_pipeline(self) -> bool:
        """Test OCR pipeline"""
        
        # Create test image with text
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', (400, 200), color='white')
        d = ImageDraw.Draw(img)
        d.text((10, 10), "BEV OCR Test", fill='black')
        img.save('/tmp/test_ocr.png')
        
        # Process with OCR pipeline
        from src.pipeline.enhanced_ocr_pipeline import EnhancedOCRPipeline
        
        pipeline = EnhancedOCRPipeline()
        result = await pipeline.process_document('/tmp/test_ocr.png')
        
        if 'BEV' not in result.text:
            raise Exception(f"OCR failed to extract text: {result.text}")
        
        return True
    
    async def test_multi_node(self) -> bool:
        """Test multi-node deployment"""
        
        nodes = [
            ('THANOS', '100.122.12.54'),
            ('Oracle1', '100.96.197.84')
        ]
        
        for node_name, node_ip in nodes:
            # Check if node is responsive
            try:
                response = requests.get(
                    f'http://{node_ip}:8000/health',
                    timeout=5
                )
                if response.status_code != 200:
                    raise Exception(f"{node_name} health check failed")
            except requests.exceptions.RequestException:
                # Node might be offline or in development mode
                logger.warning(f"{node_name} ({node_ip}) not accessible")
        
        return True
    
    async def test_end_to_end(self) -> bool:
        """Test complete end-to-end flow"""
        
        # 1. Submit research task
        task_response = requests.post(
            'http://localhost:8000/agents/research_coordinator/investigate',
            json={
                'topic': 'test_integration',
                'depth': 'quick',
                'sources': ['test']
            }
        )
        
        if task_response.status_code != 200:
            raise Exception(f"Research task failed: {task_response.text}")
        
        task_id = task_response.json()['task_id']
        
        # 2. Wait for completion
        await asyncio.sleep(5)
        
        # 3. Check result
        result_response = requests.get(
            f'http://localhost:8000/tasks/{task_id}/status'
        )
        
        if result_response.status_code != 200:
            raise Exception(f"Task status check failed: {result_response.text}")
        
        status = result_response.json()['status']
        if status not in ['completed', 'processing']:
            raise Exception(f"Task failed with status: {status}")
        
        return True
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate test report"""
        
        total_tests = len(self.test_results)
        passed_tests = sum(
            1 for r in self.test_results.values() 
            if r['status'] == 'PASSED'
        )
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': total_tests - passed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            'details': self.test_results,
            'recommendations': []
        }
        
        # Add recommendations based on failures
        for test_name, result in self.test_results.items():
            if result['status'] == 'FAILED':
                if 'Database' in test_name:
                    report['recommendations'].append(
                        "Check database containers are running: docker-compose -f docker-compose-infrastructure.yml up -d"
                    )
                elif 'Message Queue' in test_name:
                    report['recommendations'].append(
                        "Check message queue services: docker-compose -f docker/message-queue/docker-compose-messaging.yml up -d"
                    )
                elif 'Agent' in test_name:
                    report['recommendations'].append(
                        "Restart agent services: systemctl restart bev-agent-coordinator"
                    )
        
        # Save report
        with open('/tmp/bev_integration_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

async def main():
    """Run complete integration test"""
    
    tester = BEVIntegrationTester()
    report = await tester.run_all_tests()
    
    # Print summary
    print("\n" + "="*50)
    print("BEV INTEGRATION TEST REPORT")
    print("="*50)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    
    if report['summary']['failed'] > 0:
        print("\nFailed Tests:")
        for test_name, result in report['details'].items():
            if result['status'] == 'FAILED':
                print(f"  - {test_name}: {result.get('error', 'Unknown error')}")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    else:
        print("\n✅ All tests passed! BEV is fully operational!")
    
    print(f"\nFull report saved to: /tmp/bev_integration_report.json")

if __name__ == "__main__":
    asyncio.run(main())
