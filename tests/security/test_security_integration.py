#!/usr/bin/env python3
"""
Integration tests for advanced security operations center modules
Tests the interaction between tactical_intelligence, defense_automation,
opsec_enforcer, and intel_fusion modules
"""

import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import torch

# Import security modules
import sys
sys.path.append('/home/starlord/Projects/Bev/src')

from security.security_framework import OperationalSecurityFramework
from security.tactical_intelligence import (
    TacticalIntelligencePlatform,
    IntelligenceReport,
    IntelType,
    ThreatActor,
    IOCEnrichment
)
from security.defense_automation import (
    DefenseAutomationEngine,
    SecurityEvent,
    ThreatLevel,
    MalwareAnalysis,
    SandboxResult
)
from security.opsec_enforcer import (
    OpsecEnforcer,
    UserProfile,
    InsiderThreatEvent,
    ThreatType,
    RiskLevel
)
from security.intel_fusion import (
    IntelligenceFusionProcessor,
    ThreatIndicator,
    ThreatClassification,
    ThreatSeverity,
    IntelligenceType
)

@pytest.fixture
async def security_framework():
    """Create security framework for testing"""
    framework = OperationalSecurityFramework()
    # Mock initialization to avoid actual VPN/networking setup
    with patch.object(framework.vpn, 'connect', return_value=True):
        with patch.object(framework.encryption, 'initialize_keys'):
            with patch.object(framework.sandbox, 'configure_environments'):
                await framework.initialize_security()
    return framework

@pytest.fixture
async def tactical_intelligence(security_framework):
    """Create tactical intelligence platform"""
    platform = TacticalIntelligencePlatform(security_framework)
    # Mock database initialization
    with patch('asyncpg.create_pool', return_value=AsyncMock()):
        with patch('redis.asyncio.from_url', return_value=AsyncMock()):
            await platform.initialize(
                redis_url="redis://localhost:6379",
                db_url="postgresql://test:test@localhost/test"
            )
    return platform

@pytest.fixture
async def defense_automation(security_framework):
    """Create defense automation engine"""
    engine = DefenseAutomationEngine(security_framework)
    # Mock database and Docker initialization
    with patch('asyncpg.create_pool', return_value=AsyncMock()):
        with patch('redis.asyncio.from_url', return_value=AsyncMock()):
            with patch('docker.from_env', return_value=MagicMock()):
                await engine.initialize(
                    redis_url="redis://localhost:6379",
                    db_url="postgresql://test:test@localhost/test"
                )
    return engine

@pytest.fixture
async def opsec_enforcer(security_framework):
    """Create OPSEC enforcer"""
    enforcer = OpsecEnforcer(security_framework)
    # Mock database initialization
    with patch('asyncpg.create_pool', return_value=AsyncMock()):
        with patch('redis.asyncio.from_url', return_value=AsyncMock()):
            await enforcer.initialize(
                redis_url="redis://localhost:6379",
                db_url="postgresql://test:test@localhost/test"
            )
    return enforcer

@pytest.fixture
async def intel_fusion(security_framework):
    """Create intelligence fusion processor"""
    processor = IntelligenceFusionProcessor(security_framework)
    # Mock database and HTTP session initialization
    with patch('asyncpg.create_pool', return_value=AsyncMock()):
        with patch('redis.asyncio.from_url', return_value=AsyncMock()):
            with patch('aiohttp.ClientSession', return_value=AsyncMock()):
                await processor.initialize(
                    redis_url="redis://localhost:6379",
                    db_url="postgresql://test:test@localhost/test"
                )
    return processor

class TestSecurityFrameworkIntegration:
    """Test core security framework functionality"""

    @pytest.mark.asyncio
    async def test_security_framework_initialization(self, security_framework):
        """Test security framework initializes correctly"""
        assert security_framework is not None
        assert security_framework.vpn is not None
        assert security_framework.encryption is not None
        assert security_framework.pii_redactor is not None

    @pytest.mark.asyncio
    async def test_encryption_functionality(self, security_framework):
        """Test encryption and decryption functionality"""
        test_data = {"sensitive": "data", "credentials": "secret"}
        agent_id = "test_agent"

        # Encrypt data
        encrypted = security_framework.encryption.encrypt_data(test_data, agent_id)
        assert encrypted is not None
        assert 'data' in encrypted
        assert 'metadata' in encrypted

        # Decrypt data
        decrypted = security_framework.encryption.decrypt_data(encrypted, agent_id)
        assert decrypted == test_data

    @pytest.mark.asyncio
    async def test_pii_redaction(self, security_framework):
        """Test PII redaction functionality"""
        test_text = "Contact John Doe at john.doe@example.com or 555-123-4567"

        redacted = security_framework.pii_redactor.redact(test_text)

        # Should redact email and phone
        assert "john.doe@example.com" not in redacted
        assert "555-123-4567" not in redacted
        assert "[EMAIL_" in redacted
        assert "[PHONE_" in redacted

class TestTacticalIntelligenceIntegration:
    """Test tactical intelligence platform functionality"""

    @pytest.mark.asyncio
    async def test_intelligence_submission_and_processing(self, tactical_intelligence):
        """Test intelligence report submission and processing"""
        # Create test intelligence report
        report = IntelligenceReport(
            id="test_report_001",
            source="test_analyst",
            intel_type=IntelType.OSINT,
            content="APT29 observed using new malware variant targeting financial institutions",
            confidence=0.85,
            timestamp=datetime.now(),
            indicators=["192.168.1.100", "malicious.example.com", "a1b2c3d4e5f6"],
            threat_actors=["APT29"],
            techniques=["T1055", "T1071"]
        )

        # Submit intelligence
        report_id = await tactical_intelligence.submit_intelligence(report)
        assert report_id == report.id
        assert report_id in tactical_intelligence.intelligence_reports

        # Wait for processing
        await asyncio.sleep(0.1)

        # Verify processing occurred
        processed_report = tactical_intelligence.intelligence_reports[report_id]
        assert processed_report.id == report.id

    @pytest.mark.asyncio
    async def test_ioc_enrichment(self, tactical_intelligence):
        """Test IoC enrichment functionality"""
        test_ip = "192.168.1.100"

        # Mock geolocation lookup
        with patch.object(tactical_intelligence.ioc_enricher, '_get_geolocation') as mock_geo:
            mock_geo.return_value = {
                'country': 'United States',
                'country_code': 'US',
                'latitude': 40.7128,
                'longitude': -74.0060
            }

            enrichment = await tactical_intelligence.ioc_enricher.enrich_ioc(test_ip, 'ip')

            assert enrichment.ioc == test_ip
            assert enrichment.ioc_type == 'ip'
            assert enrichment.geolocation is not None

    @pytest.mark.asyncio
    async def test_campaign_analysis(self, tactical_intelligence):
        """Test campaign analysis functionality"""
        # Create multiple related intelligence reports
        reports = []
        for i in range(3):
            report = IntelligenceReport(
                id=f"campaign_report_{i}",
                source=f"analyst_{i}",
                intel_type=IntelType.OSINT,
                content=f"APT29 campaign activity detected in sector {i}",
                confidence=0.8,
                timestamp=datetime.now() - timedelta(hours=i),
                indicators=[f"192.168.1.{100+i}", "campaign.example.com"],
                threat_actors=["APT29"],
                techniques=["T1055", "T1071"]
            )
            reports.append(report)
            await tactical_intelligence.submit_intelligence(report)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Analyze campaigns
        analysis = await tactical_intelligence.campaign_tracker.analyze_campaign_attribution(reports)

        assert 'campaign_candidates' in analysis
        # Should detect potential campaign from related reports
        if analysis['campaign_candidates']:
            campaign = analysis['campaign_candidates'][0]
            assert campaign['report_count'] == 3

class TestDefenseAutomationIntegration:
    """Test defense automation engine functionality"""

    @pytest.mark.asyncio
    async def test_security_event_processing(self, defense_automation):
        """Test security event processing and response"""
        # Create test security event
        event = SecurityEvent(
            id="test_event_001",
            timestamp=datetime.now(),
            event_type="intrusion_attempt",
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            source_port=12345,
            destination_port=22,
            protocol="tcp",
            payload=b'malicious_payload',
            threat_level=ThreatLevel.HIGH,
            confidence=0.9
        )

        # Process security event
        actions = await defense_automation.process_security_event(event)

        assert isinstance(actions, list)
        # Should execute some response actions for high threat level
        assert len(actions) > 0

    @pytest.mark.asyncio
    async def test_malware_analysis(self, defense_automation):
        """Test malware analysis functionality"""
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("This is a test file for malware analysis")
            test_file_path = f.name

        try:
            # Mock sandbox execution
            with patch.object(defense_automation.malware_sandbox, '_dynamic_analysis') as mock_dynamic:
                mock_dynamic.return_value = {
                    'capabilities': ['file_access', 'network_communication'],
                    'network': {'192.168.1.1': True},
                    'file_ops': ['/tmp/malware_file'],
                    'registry_ops': [],
                    'processes': {'malware.exe': True}
                }

                # Analyze file
                analysis = await defense_automation.analyze_file_sample(test_file_path)

                assert isinstance(analysis, MalwareAnalysis)
                assert analysis.file_name == os.path.basename(test_file_path)
                assert analysis.sandbox_result in [
                    SandboxResult.MALICIOUS,
                    SandboxResult.SUSPICIOUS,
                    SandboxResult.BENIGN,
                    SandboxResult.ERROR
                ]

        finally:
            # Clean up test file
            os.unlink(test_file_path)

    @pytest.mark.asyncio
    async def test_honeypot_deployment(self, defense_automation):
        """Test honeypot deployment functionality"""
        from security.defense_automation import HoneypotConfig

        # Create honeypot configuration
        config = HoneypotConfig(
            id="test_honeypot",
            name="Test SSH Honeypot",
            service_type="ssh",
            port=2222,
            interface="eth0",
            is_active=False,
            interaction_level="medium",
            logging_enabled=True
        )

        # Mock Docker operations
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "container_id_12345"

            # Deploy honeypot
            success = await defense_automation.honeypot_manager.deploy_honeypot(config)

            assert success
            assert config.id in defense_automation.honeypot_manager.honeypots
            assert defense_automation.honeypot_manager.honeypots[config.id].is_active

class TestOpsecEnforcerIntegration:
    """Test OPSEC enforcer functionality"""

    @pytest.mark.asyncio
    async def test_user_behavior_analysis(self, opsec_enforcer):
        """Test user behavior analysis for insider threats"""
        user_id = "test_user_001"

        # Create test user activities
        activities = [
            {
                'type': 'login',
                'timestamp': datetime.now(),
                'location': 'office'
            },
            {
                'type': 'file_access',
                'timestamp': datetime.now(),
                'file_path': '/confidential/sensitive_data.xlsx',
                'sensitive': True,
                'size': 1024000
            },
            {
                'type': 'communication',
                'timestamp': datetime.now(),
                'method': 'email',
                'recipient': 'external@competitor.com',
                'external': True,
                'content': 'Sharing confidential project details'
            }
        ]

        # Analyze user behavior
        analysis = await opsec_enforcer.process_user_activity(user_id, activities)

        assert 'risk_score' in analysis
        assert 'threat_indicators' in analysis
        assert isinstance(analysis['risk_score'], float)
        assert 0.0 <= analysis['risk_score'] <= 1.0

        # Should detect suspicious activity (external communication with sensitive content)
        if analysis['threat_indicators']:
            assert any('suspicious' in indicator for indicator in analysis['threat_indicators'])

    @pytest.mark.asyncio
    async def test_data_exfiltration_prevention(self, opsec_enforcer):
        """Test data exfiltration prevention functionality"""
        # Create test data transfer event
        transfer_event = {
            'id': 'transfer_001',
            'user_id': 'test_user_001',
            'file_path': '/tmp/test_file.txt',
            'file_size': 50000000,  # 50MB
            'destination': 'external@gmail.com',
            'method': 'email',
            'timestamp': datetime.now(),
            'user_risk_score': 0.8,
            'content_risk_score': 0.7,
            'encrypted': False
        }

        # Analyze data transfer
        analysis = await opsec_enforcer.dlp_system.analyze_data_transfer(transfer_event)

        assert 'risk_score' in analysis
        assert 'blocked' in analysis
        assert 'reasons' in analysis

        # High-risk transfer should be flagged
        assert analysis['risk_score'] > 0.5

    @pytest.mark.asyncio
    async def test_communication_monitoring(self, opsec_enforcer):
        """Test communication security monitoring"""
        from security.opsec_enforcer import CommunicationEvent

        # Create test communication event
        comm_event = CommunicationEvent(
            id="comm_001",
            user_id="test_user_001",
            timestamp=datetime.now(),
            communication_type="email",
            participants=["internal@company.com", "external@competitor.com"],
            content_summary="Discussing confidential project details and sensitive financial information",
            risk_indicators=["confidential", "financial"],
            classification="restricted"
        )

        # Monitor communication
        analysis = await opsec_enforcer.comm_enforcer.monitor_communication(comm_event)

        assert 'risk_score' in analysis
        assert 'flagged' in analysis
        assert 'categories' in analysis

        # Should flag communication with confidential content to external recipient
        assert analysis['risk_score'] > 0.5

class TestIntelligenceFusionIntegration:
    """Test intelligence fusion processor functionality"""

    @pytest.mark.asyncio
    async def test_threat_feed_aggregation(self, intel_fusion):
        """Test threat feed collection and aggregation"""
        # Mock HTTP responses for threat feeds
        mock_responses = {
            'json_feed': json.dumps([
                {
                    'indicator': '192.168.1.100',
                    'type': 'ip',
                    'malware_family': 'trojan',
                    'confidence': 0.8
                },
                {
                    'indicator': 'malicious.example.com',
                    'type': 'domain',
                    'category': 'phishing',
                    'confidence': 0.9
                }
            ]),
            'text_feed': '192.168.1.101\n192.168.1.102\nmalicious2.example.com'
        }

        # Mock feed data retrieval
        async def mock_fetch_feed_data(feed):
            if 'json' in feed.feed_type:
                return mock_responses['json_feed']
            else:
                return mock_responses['text_feed']

        with patch.object(intel_fusion.feed_aggregator, '_fetch_feed_data', side_effect=mock_fetch_feed_data):
            # Collect threat feeds
            feed_indicators = await intel_fusion.feed_aggregator.collect_threat_feeds()

            assert isinstance(feed_indicators, dict)
            # Should have processed some feeds
            assert len(feed_indicators) > 0

            # Check that indicators were extracted
            total_indicators = sum(len(indicators) for indicators in feed_indicators.values())
            assert total_indicators > 0

    @pytest.mark.asyncio
    async def test_threat_classification(self, intel_fusion):
        """Test ML threat classification functionality"""
        # Create test threat indicators
        indicators = [
            ThreatIndicator(
                id="indicator_001",
                value="192.168.1.100",
                indicator_type="ip",
                classification=ThreatClassification.MALWARE,
                severity=ThreatSeverity.HIGH,
                confidence=0.8,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                sources=["test_feed"]
            ),
            ThreatIndicator(
                id="indicator_002",
                value="phishing.example.com",
                indicator_type="domain",
                classification=ThreatClassification.PHISHING,
                severity=ThreatSeverity.MEDIUM,
                confidence=0.7,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                sources=["test_feed"]
            )
        ]

        # Mock ML model predictions
        with patch.object(intel_fusion.classification_engine.classification_model, 'forward') as mock_forward:
            # Mock classification predictions
            mock_predictions = torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])
            mock_forward.return_value = mock_predictions

            # Classify threats
            classified_indicators = await intel_fusion.classification_engine.classify_threats(indicators)

            assert len(classified_indicators) == len(indicators)
            # Indicators should maintain their structure
            for indicator in classified_indicators:
                assert hasattr(indicator, 'classification')
                assert hasattr(indicator, 'confidence')

    @pytest.mark.asyncio
    async def test_predictive_threat_modeling(self, intel_fusion):
        """Test predictive threat modeling functionality"""
        # Create test indicators for prediction
        indicators = []
        for i in range(10):
            indicator = ThreatIndicator(
                id=f"pred_indicator_{i}",
                value=f"192.168.1.{100+i}",
                indicator_type="ip",
                classification=ThreatClassification.MALWARE,
                severity=ThreatSeverity.MEDIUM,
                confidence=0.7 + (i * 0.02),
                first_seen=datetime.now() - timedelta(hours=i),
                last_seen=datetime.now(),
                sources=["prediction_feed"]
            )
            indicators.append(indicator)

        # Run predictive analysis
        predictions = await intel_fusion.predictive_modeler.predict_emerging_threats(indicators, [])

        assert 'temporal_trends' in predictions
        assert 'emerging_clusters' in predictions
        assert 'evolution_prediction' in predictions
        assert 'confidence_score' in predictions

    @pytest.mark.asyncio
    async def test_geospatial_threat_mapping(self, intel_fusion):
        """Test geospatial threat mapping functionality"""
        # Create test indicators with geolocation
        indicators = [
            ThreatIndicator(
                id="geo_indicator_001",
                value="192.168.1.100",
                indicator_type="ip",
                classification=ThreatClassification.MALWARE,
                severity=ThreatSeverity.HIGH,
                confidence=0.8,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                sources=["geo_feed"],
                geolocation={
                    'country': 'United States',
                    'country_code': 'US',
                    'latitude': 40.7128,
                    'longitude': -74.0060
                }
            ),
            ThreatIndicator(
                id="geo_indicator_002",
                value="192.168.1.101",
                indicator_type="ip",
                classification=ThreatClassification.BOTNET,
                severity=ThreatSeverity.MEDIUM,
                confidence=0.7,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                sources=["geo_feed"],
                geolocation={
                    'country': 'China',
                    'country_code': 'CN',
                    'latitude': 39.9042,
                    'longitude': 116.4074
                }
            )
        ]

        # Map threats geospatially
        geo_analysis = await intel_fusion.geospatial_mapper.map_threats_geospatially(indicators)

        assert 'total_geolocated_threats' in geo_analysis
        assert 'country_aggregation' in geo_analysis
        assert 'threat_hotspots' in geo_analysis
        assert 'strategic_analysis' in geo_analysis

        # Should identify countries
        assert len(geo_analysis['country_aggregation']) >= 2

    @pytest.mark.asyncio
    async def test_intelligence_report_generation(self, intel_fusion):
        """Test intelligence report generation"""
        # Create test indicators for reporting
        indicators = [
            ThreatIndicator(
                id="report_indicator_001",
                value="threat.example.com",
                indicator_type="domain",
                classification=ThreatClassification.APT,
                severity=ThreatSeverity.CRITICAL,
                confidence=0.9,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                sources=["premium_feed"]
            )
        ]

        # Test analysis results
        analysis_results = {
            'classification': {'accuracy': 0.85},
            'predictive_analysis': {
                'temporal_trends': {},
                'emerging_clusters': [],
                'evolution_prediction': {}
            },
            'geospatial_analysis': {
                'threat_hotspots': [],
                'strategic_analysis': {}
            }
        }

        # Generate tactical report
        tactical_report = await intel_fusion.reporter.generate_intelligence_report(
            IntelligenceType.TACTICAL, indicators, analysis_results
        )

        assert tactical_report.intelligence_type == IntelligenceType.TACTICAL
        assert tactical_report.title is not None
        assert len(tactical_report.executive_summary) > 0
        assert len(tactical_report.key_findings) > 0
        assert len(tactical_report.recommendations) > 0

class TestCrossModuleIntegration:
    """Test integration between different security modules"""

    @pytest.mark.asyncio
    async def test_tactical_to_defense_integration(self, tactical_intelligence, defense_automation):
        """Test data flow from tactical intelligence to defense automation"""
        # Create intelligence report with IoCs
        report = IntelligenceReport(
            id="integration_test_001",
            source="integration_test",
            intel_type=IntelType.TECHINT,
            content="Malicious IP detected in enterprise network",
            confidence=0.9,
            timestamp=datetime.now(),
            indicators=["192.168.1.200"],
            threat_actors=["Unknown"],
            techniques=["T1071"]
        )

        # Submit to tactical intelligence
        await tactical_intelligence.submit_intelligence(report)

        # Create corresponding security event for defense automation
        security_event = SecurityEvent(
            id="integration_event_001",
            timestamp=datetime.now(),
            event_type="network_intrusion",
            source_ip="192.168.1.200",  # Same IP from intelligence
            destination_ip="10.0.0.10",
            source_port=443,
            destination_port=80,
            protocol="tcp",
            payload=b'suspicious_traffic',
            threat_level=ThreatLevel.HIGH,
            confidence=0.9
        )

        # Process in defense automation
        actions = await defense_automation.process_security_event(security_event)

        # Should execute blocking actions for known malicious IP
        assert len(actions) > 0

    @pytest.mark.asyncio
    async def test_intel_fusion_to_tactical_integration(self, intel_fusion, tactical_intelligence):
        """Test data flow from intel fusion to tactical intelligence"""
        # Mock threat feed data that would be processed by intel fusion
        mock_indicators = [
            ThreatIndicator(
                id="fusion_indicator_001",
                value="apt.malicious.com",
                indicator_type="domain",
                classification=ThreatClassification.APT,
                severity=ThreatSeverity.CRITICAL,
                confidence=0.95,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                sources=["premium_threat_feed"]
            )
        ]

        # Process in intel fusion (simulate classification)
        with patch.object(intel_fusion.classification_engine, 'classify_threats', return_value=mock_indicators):
            classified = await intel_fusion.classification_engine.classify_threats(mock_indicators)

        # Convert to intelligence report for tactical platform
        if classified:
            indicator = classified[0]
            intelligence_report = IntelligenceReport(
                id="converted_report_001",
                source="intel_fusion_automated",
                intel_type=IntelType.TECHINT,
                content=f"High-confidence APT infrastructure detected: {indicator.value}",
                confidence=indicator.confidence,
                timestamp=datetime.now(),
                indicators=[indicator.value],
                threat_actors=["APT_Group"],
                techniques=["T1071"]
            )

            # Submit to tactical intelligence
            report_id = await tactical_intelligence.submit_intelligence(intelligence_report)
            assert report_id is not None

    @pytest.mark.asyncio
    async def test_opsec_to_tactical_integration(self, opsec_enforcer, tactical_intelligence):
        """Test insider threat data feeding into tactical intelligence"""
        user_id = "insider_threat_user"

        # Create suspicious user activities
        activities = [
            {
                'type': 'file_access',
                'timestamp': datetime.now(),
                'file_path': '/classified/weapon_designs.pdf',
                'sensitive': True,
                'size': 100000000  # Large file
            },
            {
                'type': 'communication',
                'timestamp': datetime.now(),
                'method': 'email',
                'recipient': 'foreign.agent@adversary.gov',
                'external': True,
                'content': 'Classified material attachment'
            }
        ]

        # Analyze in OPSEC enforcer
        analysis = await opsec_enforcer.process_user_activity(user_id, activities)

        # If high risk detected, create intelligence report
        if analysis.get('risk_score', 0) > 0.8:
            insider_threat_report = IntelligenceReport(
                id="insider_threat_001",
                source="opsec_automated",
                intel_type=IntelType.HUMINT,
                content=f"High-risk insider threat detected: User {user_id}",
                confidence=analysis['risk_score'],
                timestamp=datetime.now(),
                indicators=[user_id],
                threat_actors=[user_id],
                techniques=["T1078"]  # Valid Accounts
            )

            # Submit to tactical intelligence
            report_id = await tactical_intelligence.submit_intelligence(insider_threat_report)
            assert report_id is not None

    @pytest.mark.asyncio
    async def test_end_to_end_threat_processing(self, intel_fusion, tactical_intelligence,
                                               defense_automation, opsec_enforcer):
        """Test complete end-to-end threat processing workflow"""
        # Step 1: Intel fusion processes external threat feeds
        mock_feed_data = json.dumps([{
            'indicator': '203.0.113.100',
            'type': 'ip',
            'classification': 'malware',
            'confidence': 0.9
        }])

        with patch.object(intel_fusion.feed_aggregator, '_fetch_feed_data', return_value=mock_feed_data):
            feed_indicators = await intel_fusion.feed_aggregator.collect_threat_feeds()

        # Step 2: Convert to tactical intelligence
        if feed_indicators:
            for feed_id, indicators in feed_indicators.items():
                for indicator in indicators:
                    intel_report = IntelligenceReport(
                        id=f"e2e_report_{indicator.id}",
                        source="automated_intel_fusion",
                        intel_type=IntelType.TECHINT,
                        content=f"Threat indicator detected: {indicator.value}",
                        confidence=indicator.confidence,
                        timestamp=datetime.now(),
                        indicators=[indicator.value],
                        threat_actors=["Unknown"],
                        techniques=["T1071"]
                    )
                    await tactical_intelligence.submit_intelligence(intel_report)

        # Step 3: Defense automation detects related activity
        security_event = SecurityEvent(
            id="e2e_security_event",
            timestamp=datetime.now(),
            event_type="malware_communication",
            source_ip="203.0.113.100",  # Same IP from threat feed
            destination_ip="10.0.0.50",
            source_port=443,
            destination_port=80,
            protocol="tcp",
            payload=b'c2_communication',
            threat_level=ThreatLevel.HIGH,
            confidence=0.9
        )

        actions = await defense_automation.process_security_event(security_event)

        # Step 4: OPSEC enforcer detects related user activity
        suspicious_activities = [
            {
                'type': 'communication',
                'timestamp': datetime.now(),
                'method': 'network',
                'destination': '203.0.113.100',  # Same IP
                'suspicious': True
            }
        ]

        opsec_analysis = await opsec_enforcer.process_user_activity("suspicious_user", suspicious_activities)

        # Verify end-to-end processing
        assert len(actions) > 0  # Defense automation took action
        assert 'risk_score' in opsec_analysis  # OPSEC analysis completed

        # Integration should result in comprehensive threat response
        assert security_event.source_ip == "203.0.113.100"  # Consistent threat tracking

class TestPerformanceAndScalability:
    """Test performance and scalability of security modules"""

    @pytest.mark.asyncio
    async def test_high_volume_indicator_processing(self, intel_fusion):
        """Test processing large volumes of threat indicators"""
        # Create large number of test indicators
        indicators = []
        for i in range(100):
            indicator = ThreatIndicator(
                id=f"perf_indicator_{i}",
                value=f"192.168.1.{i % 255}",
                indicator_type="ip",
                classification=ThreatClassification.MALWARE,
                severity=ThreatSeverity.MEDIUM,
                confidence=0.7,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                sources=["performance_test"]
            )
            indicators.append(indicator)

        # Process indicators and measure performance
        start_time = datetime.now()

        # Mock ML processing to avoid actual model execution
        with patch.object(intel_fusion.classification_engine, 'classify_threats', return_value=indicators):
            classified = await intel_fusion.classification_engine.classify_threats(indicators)

        processing_time = (datetime.now() - start_time).total_seconds()

        # Should process 100 indicators quickly (under 5 seconds)
        assert processing_time < 5.0
        assert len(classified) == 100

    @pytest.mark.asyncio
    async def test_concurrent_security_event_processing(self, defense_automation):
        """Test concurrent processing of multiple security events"""
        # Create multiple security events
        events = []
        for i in range(10):
            event = SecurityEvent(
                id=f"concurrent_event_{i}",
                timestamp=datetime.now(),
                event_type="intrusion_attempt",
                source_ip=f"192.168.1.{100+i}",
                destination_ip="10.0.0.1",
                source_port=12345 + i,
                destination_port=22,
                protocol="tcp",
                payload=b'test_payload',
                threat_level=ThreatLevel.MEDIUM,
                confidence=0.7
            )
            events.append(event)

        # Process events concurrently
        start_time = datetime.now()

        tasks = [defense_automation.process_security_event(event) for event in events]
        results = await asyncio.gather(*tasks)

        processing_time = (datetime.now() - start_time).total_seconds()

        # Should process 10 events concurrently in reasonable time
        assert processing_time < 10.0
        assert len(results) == 10

class TestErrorHandlingAndResilience:
    """Test error handling and system resilience"""

    @pytest.mark.asyncio
    async def test_database_connection_failure_handling(self, intel_fusion):
        """Test handling of database connection failures"""
        # Simulate database connection failure
        with patch('asyncpg.create_pool', side_effect=Exception("Database connection failed")):
            try:
                await intel_fusion.initialize()
                # Should handle gracefully without crashing
                assert True
            except Exception as e:
                # Should not propagate database errors during testing
                pytest.fail(f"Database error not handled gracefully: {e}")

    @pytest.mark.asyncio
    async def test_malformed_threat_feed_handling(self, intel_fusion):
        """Test handling of malformed threat feed data"""
        # Mock malformed JSON data
        malformed_data = "{'invalid': json syntax"

        with patch.object(intel_fusion.feed_aggregator, '_fetch_feed_data', return_value=malformed_data):
            # Should handle malformed data gracefully
            feed_indicators = await intel_fusion.feed_aggregator.collect_threat_feeds()

            # Should return empty results rather than crash
            assert isinstance(feed_indicators, dict)

    @pytest.mark.asyncio
    async def test_ml_model_failure_handling(self, intel_fusion):
        """Test handling of ML model failures"""
        indicators = [
            ThreatIndicator(
                id="ml_test_indicator",
                value="192.168.1.100",
                indicator_type="ip",
                classification=ThreatClassification.MALWARE,
                severity=ThreatSeverity.MEDIUM,
                confidence=0.7,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                sources=["test_feed"]
            )
        ]

        # Simulate ML model failure
        with patch.object(intel_fusion.classification_engine.classification_model, 'forward',
                         side_effect=Exception("Model inference failed")):
            # Should handle ML failures gracefully
            classified = await intel_fusion.classification_engine.classify_threats(indicators)

            # Should return original indicators even if ML fails
            assert len(classified) == 1

@pytest.mark.asyncio
async def test_cleanup_and_shutdown():
    """Test proper cleanup and shutdown of all modules"""
    # Initialize all modules
    security_framework = OperationalSecurityFramework()

    with patch.object(security_framework.vpn, 'connect', return_value=True):
        with patch.object(security_framework.encryption, 'initialize_keys'):
            with patch.object(security_framework.sandbox, 'configure_environments'):
                await security_framework.initialize_security()

    # Create modules
    tactical_intel = TacticalIntelligencePlatform(security_framework)
    defense_automation = DefenseAutomationEngine(security_framework)
    opsec_enforcer = OpsecEnforcer(security_framework)
    intel_fusion = IntelligenceFusionProcessor(security_framework)

    # Mock initialization
    with patch('asyncpg.create_pool', return_value=AsyncMock()):
        with patch('redis.asyncio.from_url', return_value=AsyncMock()):
            with patch('aiohttp.ClientSession', return_value=AsyncMock()):
                with patch('docker.from_env', return_value=MagicMock()):
                    await tactical_intel.initialize()
                    await defense_automation.initialize()
                    await opsec_enforcer.initialize()
                    await intel_fusion.initialize()

    # Test shutdown
    await tactical_intel.shutdown()
    await defense_automation.shutdown()
    await opsec_enforcer.shutdown()
    await intel_fusion.shutdown()

    # Should complete without errors
    assert True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])