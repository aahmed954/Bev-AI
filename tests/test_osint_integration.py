#!/usr/bin/env python3
"""
Test suite for OSINT Integration Layer
Validates event processing, database integration, and avatar communication
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from avatar.osint_integration_layer import (
    OSINTEvent, OSINTEventType, ThreatLevel, InvestigationType,
    InvestigationState, OSINTEventProcessor, DatabaseConnector,
    OSINTAnalyzerConnector, MessageQueueHandler, OSINTIntegrationLayer
)

@pytest.fixture
async def event_processor():
    """Create event processor instance"""
    processor = OSINTEventProcessor()
    return processor

@pytest.fixture
async def sample_event():
    """Create sample OSINT event"""
    return OSINTEvent(
        event_type=OSINTEventType.BREACH_DISCOVERED,
        investigation_type=InvestigationType.BREACH_DATABASE,
        title="Test Breach Found",
        description="Test breach description",
        data={'source': 'TestDB', 'count': 1000},
        threat_level=ThreatLevel.HIGH,
        investigation_id="test-investigation-123"
    )

@pytest.fixture
async def integration_layer():
    """Create integration layer instance with mocked dependencies"""

    config = {
        'databases': {
            'postgres': {'host': 'localhost', 'port': 5432, 'user': 'test',
                        'password': 'test', 'database': 'test'},
            'neo4j': {'uri': 'bolt://localhost:7687', 'user': 'neo4j',
                     'password': 'test'},
            'redis': {'url': 'redis://localhost:6379'},
            'qdrant': {'host': 'localhost', 'port': 6333}
        },
        'messaging': {
            'kafka_brokers': 'localhost:9092',
            'rabbitmq_url': 'amqp://guest:guest@localhost/',
            'nats_url': 'nats://localhost:4222'
        },
        'avatar_websocket_url': 'ws://localhost:8091/ws',
        'response_timeout': 100,
        'batch_size': 10
    }

    with patch('avatar.osint_integration_layer.DatabaseConnector.initialize', new_callable=AsyncMock):
        with patch('avatar.osint_integration_layer.MessageQueueHandler.initialize', new_callable=AsyncMock):
            integration = OSINTIntegrationLayer(config)
            # Mock the WebSocket connection
            integration.avatar_websocket = AsyncMock()
            return integration

class TestOSINTEventProcessor:
    """Test OSINT event processing"""

    @pytest.mark.asyncio
    async def test_process_event(self, event_processor, sample_event):
        """Test basic event processing"""

        # Process event
        response = await event_processor.process_event(sample_event)

        # Verify response structure
        assert 'event_id' in response
        assert response['emotion'] == 'alert'
        assert response['threat_level'] == ThreatLevel.HIGH.value
        assert 'response_text' in response
        assert response['animation_cue'] == 'alert_pose'

    @pytest.mark.asyncio
    async def test_investigation_state_update(self, event_processor, sample_event):
        """Test investigation state updates during event processing"""

        # Create investigation state
        investigation_id = sample_event.investigation_id
        state = InvestigationState(
            investigation_id=investigation_id,
            investigation_type=InvestigationType.BREACH_DATABASE,
            started_at=datetime.now()
        )
        event_processor.active_investigations[investigation_id] = state

        # Process breach event
        await event_processor.process_event(sample_event)

        # Verify state was updated
        assert state.breaches_found == 1
        assert state.findings_count == 1
        assert state.events_processed == 1

    @pytest.mark.asyncio
    async def test_response_text_generation(self, event_processor):
        """Test response text generation for different event types"""

        # Test breakthrough event
        breakthrough_event = OSINTEvent(
            event_type=OSINTEventType.BREAKTHROUGH_MOMENT,
            investigation_type=InvestigationType.GRAPH_ANALYSIS,
            title="Major Discovery",
            data={'discovery': 'Important connection found'}
        )

        response = await event_processor.process_event(breakthrough_event)
        assert 'breakthrough' in response['response_text'].lower() or \
               'connected' in response['response_text'].lower()
        assert response['emotion'] == 'excited'

    @pytest.mark.asyncio
    async def test_emotion_mapping(self, event_processor):
        """Test emotion mapping for different event types"""

        emotions_to_test = [
            (OSINTEventType.THREAT_IDENTIFIED, 'concerned'),
            (OSINTEventType.PATTERN_DETECTED, 'excited'),
            (OSINTEventType.INVESTIGATION_COMPLETE, 'satisfied'),
            (OSINTEventType.INVESTIGATION_STARTED, 'focused')
        ]

        for event_type, expected_emotion in emotions_to_test:
            event = OSINTEvent(event_type=event_type)
            response = await event_processor.process_event(event)
            assert response['emotion'] == expected_emotion

class TestDatabaseConnector:
    """Test database connectivity and operations"""

    @pytest.mark.asyncio
    @patch('asyncpg.create_pool', new_callable=AsyncMock)
    @patch('neo4j.AsyncGraphDatabase.driver')
    @patch('redis.asyncio.from_url', new_callable=AsyncMock)
    @patch('qdrant_client.QdrantClient')
    async def test_initialize_connections(self, mock_qdrant, mock_redis,
                                         mock_neo4j, mock_postgres):
        """Test database connection initialization"""

        config = {
            'postgres': {'host': 'localhost', 'port': 5432, 'user': 'test',
                        'password': 'test', 'database': 'test'},
            'neo4j': {'uri': 'bolt://localhost', 'user': 'neo4j', 'password': 'test'},
            'redis': {'url': 'redis://localhost'},
            'qdrant': {'host': 'localhost', 'port': 6333}
        }

        connector = DatabaseConnector(config)
        await connector.initialize()

        # Verify all connections attempted
        mock_postgres.assert_called_once()
        mock_neo4j.assert_called_once()
        mock_redis.assert_called_once()
        mock_qdrant.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_investigation_state(self):
        """Test caching investigation state in Redis"""

        config = {
            'redis': {'url': 'redis://localhost'}
        }

        connector = DatabaseConnector(config)
        connector.redis_client = AsyncMock()

        # Create investigation state
        state = InvestigationState(
            investigation_id='test-123',
            investigation_type=InvestigationType.BREACH_DATABASE,
            started_at=datetime.now()
        )

        # Cache state
        await connector.cache_investigation_state(state)

        # Verify Redis setex was called
        connector.redis_client.setex.assert_called_once()
        call_args = connector.redis_client.setex.call_args
        assert call_args[0][0] == 'osint:investigation:test-123'
        assert call_args[0][1] == 3600  # TTL

class TestOSINTAnalyzerConnector:
    """Test OSINT analyzer service connections"""

    @pytest.mark.asyncio
    async def test_trigger_breach_analysis(self):
        """Test breach analysis trigger"""

        config = {'breach_analyzer_url': 'http://localhost:8081'}
        connector = OSINTAnalyzerConnector(config)

        # Mock HTTP client
        connector.http_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            'breaches': [
                {'name': 'TestBreach', 'sensitive': True}
            ]
        }
        mock_response.raise_for_status = Mock()
        connector.http_client.post.return_value = mock_response

        # Trigger analysis
        result = await connector.trigger_breach_analysis('test@example.com')

        # Verify request and response
        assert 'breaches' in result
        assert len(result['breaches']) == 1
        connector.http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_track_crypto_transactions(self):
        """Test cryptocurrency tracking"""

        config = {'crypto_analyzer_url': 'http://localhost:8083'}
        connector = OSINTAnalyzerConnector(config)

        # Mock HTTP client
        connector.http_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            'transactions': [
                {'hash': 'abc123', 'amount': 50000}
            ],
            'suspicious_activity': True
        }
        mock_response.raise_for_status = Mock()
        connector.http_client.post.return_value = mock_response

        # Track transactions
        result = await connector.track_crypto_transactions('wallet123')

        # Verify results
        assert 'transactions' in result
        assert result['suspicious_activity'] is True

class TestMessageQueueHandler:
    """Test message queue operations"""

    @pytest.mark.asyncio
    async def test_publish_event(self):
        """Test publishing events to message queues"""

        config = {
            'kafka_brokers': 'localhost:9092',
            'rabbitmq_url': 'amqp://localhost',
            'nats_url': 'nats://localhost'
        }

        handler = MessageQueueHandler(config)

        # Mock message queue clients
        handler.kafka_producer = AsyncMock()
        handler.rabbitmq_channel = AsyncMock()
        handler.nats_client = AsyncMock()

        # Create and publish event
        event = OSINTEvent(
            event_type=OSINTEventType.THREAT_IDENTIFIED,
            investigation_type=InvestigationType.THREAT_INTELLIGENCE,
            title="Test Threat"
        )

        await handler.publish_event(event)

        # Verify all queues received event
        handler.kafka_producer.send.assert_called_once()
        handler.nats_client.publish.assert_called_once()

class TestOSINTIntegrationLayer:
    """Test main integration layer functionality"""

    @pytest.mark.asyncio
    async def test_start_investigation(self, integration_layer):
        """Test starting new investigation"""

        # Mock analyzer connector
        integration_layer.analyzer_connector = AsyncMock()

        # Start investigation
        investigation_id = await integration_layer.start_investigation(
            InvestigationType.BREACH_DATABASE,
            'test@example.com',
            {'deep_scan': True}
        )

        # Verify investigation created
        assert investigation_id
        assert investigation_id in integration_layer.event_processor.active_investigations

        # Verify initial event queued
        assert not integration_layer.event_processor.event_queue.empty()

    @pytest.mark.asyncio
    async def test_correlate_findings(self, integration_layer):
        """Test correlation across investigations"""

        # Create test events with common entity
        event1 = OSINTEvent(
            investigation_id='inv1',
            data={'email': 'test@example.com'}
        )
        event2 = OSINTEvent(
            investigation_id='inv2',
            data={'email': 'test@example.com'}
        )

        integration_layer.event_processor.processed_events.extend([event1, event2])

        # Correlate findings
        result = await integration_layer.correlate_findings(['inv1', 'inv2'])

        # Verify correlation found
        assert 'correlations' in result
        assert len(result['correlations']) > 0
        assert result['correlations'][0]['entity'] == 'test@example.com'

    @pytest.mark.asyncio
    async def test_generate_threat_report(self, integration_layer):
        """Test threat report generation"""

        # Create investigation and events
        investigation_id = 'test-report-123'
        state = InvestigationState(
            investigation_id=investigation_id,
            investigation_type=InvestigationType.THREAT_INTELLIGENCE,
            started_at=datetime.now()
        )
        integration_layer.event_processor.active_investigations[investigation_id] = state

        # Add test events
        events = [
            OSINTEvent(
                investigation_id=investigation_id,
                event_type=OSINTEventType.THREAT_IDENTIFIED,
                threat_level=ThreatLevel.HIGH,
                title="High threat detected"
            ),
            OSINTEvent(
                investigation_id=investigation_id,
                event_type=OSINTEventType.PATTERN_DETECTED,
                threat_level=ThreatLevel.MEDIUM,
                title="Pattern found"
            )
        ]
        integration_layer.event_processor.processed_events.extend(events)

        # Generate report
        report = await integration_layer.generate_threat_report(investigation_id)

        # Verify report structure
        assert report['investigation_id'] == investigation_id
        assert report['findings']['total_events'] == 2
        assert report['threat_assessment']['level'] == 'HIGH'
        assert len(report['timeline']) == 2

    @pytest.mark.asyncio
    async def test_send_to_avatar(self, integration_layer):
        """Test sending responses to avatar system"""

        response = {
            'event_id': 'test-123',
            'emotion': 'excited',
            'response_text': 'Test response'
        }

        # Send to avatar
        await integration_layer._send_to_avatar(response)

        # Verify WebSocket send was called
        integration_layer.avatar_websocket.send.assert_called_once()

        # Verify message structure
        call_args = integration_layer.avatar_websocket.send.call_args
        message = json.loads(call_args[0][0])
        assert message['type'] == 'osint_update'
        assert message['data'] == response

class TestPerformance:
    """Test performance requirements"""

    @pytest.mark.asyncio
    async def test_response_time(self, event_processor):
        """Test event processing meets <100ms requirement"""

        import time

        event = OSINTEvent(
            event_type=OSINTEventType.BREACH_DISCOVERED,
            title="Test Event"
        )

        start = time.time()
        response = await event_processor.process_event(event)
        duration = (time.time() - start) * 1000

        # Should process in under 100ms
        assert duration < 100
        assert response is not None

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, integration_layer):
        """Test concurrent event processing"""

        # Create multiple events
        events = []
        for i in range(10):
            event = OSINTEvent(
                event_id=f"event-{i}",
                event_type=OSINTEventType.PATTERN_DETECTED,
                title=f"Event {i}"
            )
            events.append(event)

        # Process concurrently
        tasks = [
            integration_layer.event_processor.process_event(event)
            for event in events
        ]

        results = await asyncio.gather(*tasks)

        # Verify all processed
        assert len(results) == 10
        for result in results:
            assert 'event_id' in result

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, event_processor):
        """Test memory usage with event queue limits"""

        # Queue should have max size limit
        assert event_processor.event_queue.maxsize == 1000

        # Processed events deque should have max length
        assert event_processor.processed_events.maxlen == 1000

        # Add many events to test rotation
        for i in range(1500):
            event = OSINTEvent(event_id=f"event-{i}")
            event_processor.processed_events.append(event)

        # Should only keep last 1000
        assert len(event_processor.processed_events) == 1000

if __name__ == "__main__":
    pytest.main([__file__, "-v"])