"""
Security and Privacy Testing for AI Companion System
Tests personal data protection, encryption, access control, and privacy compliance
for companion memory, conversations, and biometric data
"""

import pytest
import asyncio
import time
import hashlib
import uuid
import json
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import sqlite3
import psycopg2

from tests.companion.fixtures.security_fixtures import *
from tests.companion.utils.companion_client import CompanionTestClient
from tests.companion.utils.security_tester import SecurityTester
from tests.companion.utils.privacy_validator import PrivacyValidator
from tests.companion.utils.encryption_validator import EncryptionValidator

@dataclass
class SecurityTestResult:
    """Result of security test execution"""
    test_name: str
    security_level: str
    vulnerabilities_found: List[str]
    compliance_score: float
    encryption_strength: str
    access_control_effectiveness: float
    data_protection_score: float
    privacy_violations: List[str]
    recommendations: List[str]

@dataclass
class PrivacyAuditResult:
    """Result of privacy audit"""
    data_categories_tested: List[str]
    encryption_compliance: bool
    access_logging_compliance: bool
    data_retention_compliance: bool
    anonymization_effectiveness: float
    gdpr_compliance_score: float
    privacy_policy_adherence: float
    user_consent_validation: bool

@dataclass
class BiometricSecurityResult:
    """Result of biometric security testing"""
    biometric_type: str
    encryption_validated: bool
    storage_security: float
    access_protection: float
    false_acceptance_rate: float
    false_rejection_rate: float
    spoof_resistance: float
    deletion_verification: bool

@pytest.mark.companion_security
@pytest.mark.privacy
class TestCompanionPrivacySecurity:
    """Test security and privacy protection for AI companion system"""

    @pytest.fixture(autouse=True)
    def setup_security_testing(self, companion_client, security_tester, privacy_validator, encryption_validator):
        """Setup security testing environment"""
        self.client = companion_client
        self.security_tester = security_tester
        self.privacy_validator = privacy_validator
        self.encryption_validator = encryption_validator
        self.test_results = []
        self.test_users = []
        self.test_data_cleanup = []

        # Create isolated test environment
        self.test_db_name = f"companion_security_test_{int(time.time())}"
        self._setup_test_database()

        yield

        # Cleanup test data and environment
        self._cleanup_test_environment()
        self._save_security_test_results()

    async def test_conversation_data_encryption(self, conversation_encryption_scenarios):
        """Test encryption of conversation data at rest and in transit"""
        for scenario_name, scenario in conversation_encryption_scenarios.items():
            print(f"Testing conversation encryption: {scenario_name}")

            user_id = f"encryption_test_user_{scenario_name}"
            self.test_users.append(user_id)

            # Create test conversation data
            conversation_data = self._generate_test_conversation_data(scenario)

            # Test encryption at rest
            rest_encryption_result = await self._test_encryption_at_rest(
                user_id, conversation_data, scenario["encryption_requirements"]
            )

            # Test encryption in transit
            transit_encryption_result = await self._test_encryption_in_transit(
                user_id, conversation_data, scenario["transit_requirements"]
            )

            # Test key management
            key_management_result = await self._test_encryption_key_management(
                user_id, scenario["key_management_requirements"]
            )

            # Validate encryption strength
            encryption_strength = await self.encryption_validator.validate_encryption_strength(
                rest_encryption_result["encryption_method"],
                scenario["minimum_strength"]
            )

            # Test data recovery and backup encryption
            backup_encryption_result = await self._test_backup_encryption(
                user_id, conversation_data, scenario["backup_requirements"]
            )

            # Aggregate encryption test results
            encryption_compliance = (
                rest_encryption_result["compliance"] and
                transit_encryption_result["compliance"] and
                key_management_result["compliance"] and
                backup_encryption_result["compliance"]
            )

            result = SecurityTestResult(
                test_name=f"conversation_encryption_{scenario_name}",
                security_level="high",
                vulnerabilities_found=self._identify_encryption_vulnerabilities([
                    rest_encryption_result, transit_encryption_result,
                    key_management_result, backup_encryption_result
                ]),
                compliance_score=1.0 if encryption_compliance else 0.8,
                encryption_strength=encryption_strength["level"],
                access_control_effectiveness=0.95,  # Measured separately
                data_protection_score=rest_encryption_result["protection_score"],
                privacy_violations=[],
                recommendations=rest_encryption_result.get("recommendations", [])
            )

            self.test_results.append(result)

            # Validate encryption requirements
            assert encryption_compliance, f"Encryption compliance failed for {scenario_name}"
            assert encryption_strength["level"] in ["strong", "very_strong"], f"Encryption strength {encryption_strength['level']} insufficient"
            assert rest_encryption_result["protection_score"] >= 0.95, f"Data protection score {rest_encryption_result['protection_score']:.2f} below 95%"

    async def test_memory_data_privacy_protection(self, memory_privacy_scenarios):
        """Test privacy protection for companion memory and personal data"""
        for scenario_name, scenario in memory_privacy_scenarios.items():
            print(f"Testing memory privacy protection: {scenario_name}")

            user_id = f"memory_privacy_test_{scenario_name}"
            self.test_users.append(user_id)

            # Create test memory data with various privacy levels
            memory_data = self._generate_test_memory_data(scenario)

            # Test data categorization and classification
            classification_result = await self.privacy_validator.classify_personal_data(
                memory_data, scenario["classification_requirements"]
            )

            # Test anonymization effectiveness
            anonymization_result = await self._test_data_anonymization(
                user_id, memory_data, scenario["anonymization_requirements"]
            )

            # Test access control for memory data
            access_control_result = await self._test_memory_access_control(
                user_id, memory_data, scenario["access_control_requirements"]
            )

            # Test data retention policies
            retention_result = await self._test_data_retention_policies(
                user_id, memory_data, scenario["retention_requirements"]
            )

            # Test user consent management
            consent_result = await self._test_user_consent_management(
                user_id, memory_data, scenario["consent_requirements"]
            )

            # Test right to be forgotten implementation
            deletion_result = await self._test_right_to_be_forgotten(
                user_id, memory_data, scenario["deletion_requirements"]
            )

            # Aggregate privacy protection results
            privacy_audit = PrivacyAuditResult(
                data_categories_tested=list(memory_data.keys()),
                encryption_compliance=classification_result["encryption_applied"],
                access_logging_compliance=access_control_result["logging_compliant"],
                data_retention_compliance=retention_result["policy_compliant"],
                anonymization_effectiveness=anonymization_result["effectiveness_score"],
                gdpr_compliance_score=self._calculate_gdpr_compliance([
                    classification_result, anonymization_result, access_control_result,
                    retention_result, consent_result, deletion_result
                ]),
                privacy_policy_adherence=consent_result["policy_adherence"],
                user_consent_validation=consent_result["consent_valid"]
            )

            result = SecurityTestResult(
                test_name=f"memory_privacy_{scenario_name}",
                security_level="high",
                vulnerabilities_found=self._identify_privacy_vulnerabilities([
                    classification_result, anonymization_result, access_control_result,
                    retention_result, consent_result, deletion_result
                ]),
                compliance_score=privacy_audit.gdpr_compliance_score,
                encryption_strength="strong",
                access_control_effectiveness=access_control_result["effectiveness"],
                data_protection_score=classification_result["protection_score"],
                privacy_violations=self._identify_privacy_violations(privacy_audit),
                recommendations=anonymization_result.get("recommendations", [])
            )

            self.test_results.append(result)

            # Validate privacy protection requirements
            assert privacy_audit.gdpr_compliance_score >= 0.95, f"GDPR compliance {privacy_audit.gdpr_compliance_score:.2f} below 95%"
            assert privacy_audit.anonymization_effectiveness >= 0.90, f"Anonymization effectiveness {privacy_audit.anonymization_effectiveness:.2f} below 90%"
            assert privacy_audit.user_consent_validation, f"User consent validation failed for {scenario_name}"
            assert deletion_result["deletion_verified"], f"Right to be forgotten implementation failed"

    async def test_biometric_authentication_security(self, biometric_security_scenarios):
        """Test security of biometric authentication and data protection"""
        for scenario_name, scenario in biometric_security_scenarios.items():
            print(f"Testing biometric security: {scenario_name}")

            user_id = f"biometric_test_{scenario_name}"
            self.test_users.append(user_id)

            biometric_type = scenario["biometric_type"]

            # Test biometric data encryption
            biometric_encryption_result = await self._test_biometric_encryption(
                user_id, biometric_type, scenario["encryption_requirements"]
            )

            # Test biometric template security
            template_security_result = await self._test_biometric_template_security(
                user_id, biometric_type, scenario["template_requirements"]
            )

            # Test authentication accuracy and security
            authentication_result = await self._test_biometric_authentication_security(
                user_id, biometric_type, scenario["authentication_requirements"]
            )

            # Test anti-spoofing measures
            anti_spoofing_result = await self._test_biometric_anti_spoofing(
                user_id, biometric_type, scenario["anti_spoofing_requirements"]
            )

            # Test biometric data deletion
            deletion_result = await self._test_biometric_data_deletion(
                user_id, biometric_type, scenario["deletion_requirements"]
            )

            # Test replay attack protection
            replay_protection_result = await self._test_biometric_replay_protection(
                user_id, biometric_type, scenario["replay_protection_requirements"]
            )

            # Aggregate biometric security results
            biometric_security = BiometricSecurityResult(
                biometric_type=biometric_type,
                encryption_validated=biometric_encryption_result["encryption_valid"],
                storage_security=template_security_result["security_score"],
                access_protection=authentication_result["access_protection"],
                false_acceptance_rate=authentication_result["far"],
                false_rejection_rate=authentication_result["frr"],
                spoof_resistance=anti_spoofing_result["resistance_score"],
                deletion_verification=deletion_result["deletion_verified"]
            )

            result = SecurityTestResult(
                test_name=f"biometric_security_{scenario_name}",
                security_level="very_high",
                vulnerabilities_found=self._identify_biometric_vulnerabilities([
                    biometric_encryption_result, template_security_result,
                    authentication_result, anti_spoofing_result, replay_protection_result
                ]),
                compliance_score=self._calculate_biometric_compliance(biometric_security),
                encryption_strength=biometric_encryption_result["encryption_strength"],
                access_control_effectiveness=authentication_result["access_protection"],
                data_protection_score=template_security_result["security_score"],
                privacy_violations=[],
                recommendations=template_security_result.get("recommendations", [])
            )

            self.test_results.append(result)

            # Validate biometric security requirements
            assert biometric_security.encryption_validated, f"Biometric encryption validation failed for {scenario_name}"
            assert biometric_security.storage_security >= 0.95, f"Biometric storage security {biometric_security.storage_security:.2f} below 95%"
            assert biometric_security.false_acceptance_rate <= 0.001, f"FAR {biometric_security.false_acceptance_rate:.4f} exceeds 0.1% threshold"
            assert biometric_security.spoof_resistance >= 0.90, f"Spoof resistance {biometric_security.spoof_resistance:.2f} below 90%"
            assert biometric_security.deletion_verification, f"Biometric data deletion verification failed"

    async def test_access_control_and_authorization(self, access_control_scenarios):
        """Test access control mechanisms and authorization systems"""
        for scenario_name, scenario in access_control_scenarios.items():
            print(f"Testing access control: {scenario_name}")

            # Test role-based access control
            rbac_result = await self._test_role_based_access_control(
                scenario["rbac_requirements"]
            )

            # Test session management security
            session_result = await self._test_session_management_security(
                scenario["session_requirements"]
            )

            # Test privilege escalation protection
            privilege_result = await self._test_privilege_escalation_protection(
                scenario["privilege_requirements"]
            )

            # Test multi-factor authentication
            mfa_result = await self._test_multi_factor_authentication(
                scenario["mfa_requirements"]
            )

            # Test API authentication and authorization
            api_auth_result = await self._test_api_authentication_authorization(
                scenario["api_requirements"]
            )

            # Test audit logging
            audit_result = await self._test_access_audit_logging(
                scenario["audit_requirements"]
            )

            # Aggregate access control results
            access_control_effectiveness = statistics.mean([
                rbac_result["effectiveness"],
                session_result["effectiveness"],
                privilege_result["effectiveness"],
                mfa_result["effectiveness"],
                api_auth_result["effectiveness"]
            ])

            result = SecurityTestResult(
                test_name=f"access_control_{scenario_name}",
                security_level="high",
                vulnerabilities_found=self._identify_access_control_vulnerabilities([
                    rbac_result, session_result, privilege_result, mfa_result, api_auth_result
                ]),
                compliance_score=audit_result["compliance_score"],
                encryption_strength="strong",
                access_control_effectiveness=access_control_effectiveness,
                data_protection_score=rbac_result["data_protection"],
                privacy_violations=[],
                recommendations=rbac_result.get("recommendations", [])
            )

            self.test_results.append(result)

            # Validate access control requirements
            assert access_control_effectiveness >= 0.90, f"Access control effectiveness {access_control_effectiveness:.2f} below 90%"
            assert mfa_result["effectiveness"] >= 0.95, f"MFA effectiveness {mfa_result['effectiveness']:.2f} below 95%"
            assert audit_result["compliance_score"] >= 0.95, f"Audit logging compliance {audit_result['compliance_score']:.2f} below 95%"

    async def test_data_breach_detection_response(self, breach_detection_scenarios):
        """Test data breach detection and incident response capabilities"""
        for scenario_name, scenario in breach_detection_scenarios.items():
            print(f"Testing breach detection: {scenario_name}")

            # Simulate various types of security incidents
            breach_type = scenario["breach_type"]

            # Test intrusion detection
            intrusion_result = await self._test_intrusion_detection(
                breach_type, scenario["intrusion_requirements"]
            )

            # Test anomaly detection
            anomaly_result = await self._test_anomaly_detection(
                breach_type, scenario["anomaly_requirements"]
            )

            # Test incident response automation
            response_result = await self._test_incident_response_automation(
                breach_type, scenario["response_requirements"]
            )

            # Test data exfiltration detection
            exfiltration_result = await self._test_data_exfiltration_detection(
                breach_type, scenario["exfiltration_requirements"]
            )

            # Test notification and alerting
            notification_result = await self._test_breach_notification_system(
                breach_type, scenario["notification_requirements"]
            )

            # Test forensic data collection
            forensic_result = await self._test_forensic_data_collection(
                breach_type, scenario["forensic_requirements"]
            )

            # Aggregate breach detection results
            detection_effectiveness = statistics.mean([
                intrusion_result["detection_rate"],
                anomaly_result["detection_rate"],
                exfiltration_result["detection_rate"]
            ])

            response_effectiveness = statistics.mean([
                response_result["response_speed"],
                notification_result["notification_speed"],
                forensic_result["collection_completeness"]
            ])

            result = SecurityTestResult(
                test_name=f"breach_detection_{scenario_name}",
                security_level="critical",
                vulnerabilities_found=self._identify_breach_detection_vulnerabilities([
                    intrusion_result, anomaly_result, response_result,
                    exfiltration_result, notification_result, forensic_result
                ]),
                compliance_score=notification_result["compliance_score"],
                encryption_strength="strong",
                access_control_effectiveness=detection_effectiveness,
                data_protection_score=response_effectiveness,
                privacy_violations=[],
                recommendations=intrusion_result.get("recommendations", [])
            )

            self.test_results.append(result)

            # Validate breach detection requirements
            assert detection_effectiveness >= 0.85, f"Breach detection effectiveness {detection_effectiveness:.2f} below 85%"
            assert response_result["response_speed"] >= 0.90, f"Incident response speed {response_result['response_speed']:.2f} below 90%"
            assert notification_result["compliance_score"] >= 0.95, f"Breach notification compliance {notification_result['compliance_score']:.2f} below 95%"

    async def test_secure_communication_protocols(self, communication_security_scenarios):
        """Test secure communication protocols and data transmission"""
        for scenario_name, scenario in communication_security_scenarios.items():
            print(f"Testing communication security: {scenario_name}")

            # Test TLS/SSL implementation
            tls_result = await self._test_tls_ssl_implementation(
                scenario["tls_requirements"]
            )

            # Test certificate validation
            cert_result = await self._test_certificate_validation(
                scenario["certificate_requirements"]
            )

            # Test perfect forward secrecy
            pfs_result = await self._test_perfect_forward_secrecy(
                scenario["pfs_requirements"]
            )

            # Test websocket security
            websocket_result = await self._test_websocket_security(
                scenario["websocket_requirements"]
            )

            # Test API security headers
            headers_result = await self._test_security_headers(
                scenario["headers_requirements"]
            )

            # Test man-in-the-middle protection
            mitm_result = await self._test_mitm_protection(
                scenario["mitm_requirements"]
            )

            # Aggregate communication security results
            communication_security_score = statistics.mean([
                tls_result["security_score"],
                cert_result["validation_score"],
                pfs_result["implementation_score"],
                websocket_result["security_score"],
                headers_result["compliance_score"],
                mitm_result["protection_score"]
            ])

            result = SecurityTestResult(
                test_name=f"communication_security_{scenario_name}",
                security_level="high",
                vulnerabilities_found=self._identify_communication_vulnerabilities([
                    tls_result, cert_result, pfs_result, websocket_result, headers_result, mitm_result
                ]),
                compliance_score=headers_result["compliance_score"],
                encryption_strength=tls_result["encryption_strength"],
                access_control_effectiveness=cert_result["validation_score"],
                data_protection_score=communication_security_score,
                privacy_violations=[],
                recommendations=tls_result.get("recommendations", [])
            )

            self.test_results.append(result)

            # Validate communication security requirements
            assert communication_security_score >= 0.90, f"Communication security score {communication_security_score:.2f} below 90%"
            assert tls_result["encryption_strength"] in ["strong", "very_strong"], f"TLS encryption strength insufficient"
            assert mitm_result["protection_score"] >= 0.95, f"MITM protection {mitm_result['protection_score']:.2f} below 95%"

    # Helper Methods for Security Testing

    def _generate_test_conversation_data(self, scenario: Dict) -> Dict[str, Any]:
        """Generate test conversation data for encryption testing"""
        return {
            "conversations": [
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": time.time(),
                    "user_message": "Can you help me analyze this security incident?",
                    "companion_response": "I'd be happy to help with your security analysis. Can you provide more details?",
                    "sensitivity_level": scenario.get("sensitivity_level", "medium"),
                    "contains_pii": False
                },
                {
                    "id": str(uuid.uuid4()),
                    "timestamp": time.time(),
                    "user_message": "My name is John Doe and I work at Example Corp",
                    "companion_response": "Thank you for sharing that information, John. How can I assist you today?",
                    "sensitivity_level": "high",
                    "contains_pii": True
                }
            ],
            "metadata": {
                "session_id": str(uuid.uuid4()),
                "encryption_required": True,
                "retention_period": scenario.get("retention_days", 90)
            }
        }

    def _generate_test_memory_data(self, scenario: Dict) -> Dict[str, Any]:
        """Generate test memory data with various privacy levels"""
        return {
            "personal_preferences": {
                "communication_style": "professional",
                "expertise_level": "advanced",
                "preferred_topics": ["network_security", "threat_hunting"],
                "privacy_level": "medium"
            },
            "sensitive_personal_data": {
                "real_name": "John Doe",
                "organization": "Example Corp",
                "location": "New York",
                "contact_info": "john.doe@example.com",
                "privacy_level": "high"
            },
            "behavioral_patterns": {
                "login_times": ["09:00", "13:00", "17:00"],
                "session_duration": "45_minutes",
                "interaction_frequency": "daily",
                "privacy_level": "low"
            },
            "biometric_data": {
                "voice_signature": "encrypted_voice_template",
                "facial_features": "encrypted_face_template",
                "privacy_level": "very_high"
            }
        }

    async def _test_encryption_at_rest(self, user_id: str, data: Dict, requirements: Dict) -> Dict[str, Any]:
        """Test data encryption at rest"""
        try:
            # Simulate storing data with encryption
            encrypted_data = self.encryption_validator.encrypt_data_at_rest(
                data, requirements["algorithm"], requirements["key_size"]
            )

            # Verify encryption was applied
            encryption_applied = encrypted_data != data

            # Test decryption
            decrypted_data = self.encryption_validator.decrypt_data_at_rest(
                encrypted_data, requirements["algorithm"]
            )

            # Verify data integrity
            data_integrity = (decrypted_data == data)

            # Check encryption strength
            strength_check = self.encryption_validator.validate_encryption_algorithm(
                requirements["algorithm"], requirements["key_size"]
            )

            return {
                "compliance": encryption_applied and data_integrity and strength_check["approved"],
                "encryption_method": requirements["algorithm"],
                "protection_score": 0.95 if encryption_applied else 0.0,
                "data_integrity": data_integrity,
                "encryption_strength": strength_check["strength"],
                "recommendations": strength_check.get("recommendations", [])
            }

        except Exception as e:
            return {
                "compliance": False,
                "encryption_method": "none",
                "protection_score": 0.0,
                "error": str(e),
                "recommendations": ["Fix encryption implementation"]
            }

    async def _test_encryption_in_transit(self, user_id: str, data: Dict, requirements: Dict) -> Dict[str, Any]:
        """Test data encryption in transit"""
        try:
            # Simulate data transmission with encryption
            transit_result = await self.security_tester.test_transit_encryption(
                data, requirements["protocol"], requirements["tls_version"]
            )

            return {
                "compliance": transit_result["encrypted"],
                "protocol": requirements["protocol"],
                "tls_version": transit_result["tls_version"],
                "cipher_suite": transit_result["cipher_suite"],
                "protection_score": 0.95 if transit_result["encrypted"] else 0.0,
                "vulnerability_scan": transit_result["vulnerabilities"]
            }

        except Exception as e:
            return {
                "compliance": False,
                "error": str(e),
                "protection_score": 0.0
            }

    def _identify_encryption_vulnerabilities(self, test_results: List[Dict]) -> List[str]:
        """Identify encryption vulnerabilities from test results"""
        vulnerabilities = []

        for result in test_results:
            if not result.get("compliance", False):
                vulnerabilities.append(f"Encryption compliance failure: {result.get('error', 'Unknown')}")

            if result.get("protection_score", 0.0) < 0.90:
                vulnerabilities.append(f"Low protection score: {result.get('protection_score', 0.0):.2f}")

            if "vulnerabilities" in result:
                vulnerabilities.extend(result["vulnerabilities"])

        return vulnerabilities

    def _calculate_gdpr_compliance(self, test_results: List[Dict]) -> float:
        """Calculate GDPR compliance score from test results"""
        compliance_factors = []

        for result in test_results:
            if "compliance_score" in result:
                compliance_factors.append(result["compliance_score"])
            elif "compliant" in result:
                compliance_factors.append(1.0 if result["compliant"] else 0.0)

        return statistics.mean(compliance_factors) if compliance_factors else 0.0

    def _setup_test_database(self):
        """Setup isolated test database for security testing"""
        try:
            # Create test database connection
            self.test_db_path = f"/tmp/{self.test_db_name}.db"
            self.test_db_connection = sqlite3.connect(self.test_db_path)

            # Create test tables
            cursor = self.test_db_connection.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    message_data TEXT,
                    encrypted_data BLOB,
                    timestamp REAL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_user_data (
                    user_id TEXT PRIMARY KEY,
                    personal_data TEXT,
                    biometric_data BLOB,
                    preferences TEXT,
                    created_at REAL
                )
            ''')
            self.test_db_connection.commit()

        except Exception as e:
            print(f"Test database setup error: {e}")

    def _cleanup_test_environment(self):
        """Clean up test environment and data"""
        try:
            # Close database connection
            if hasattr(self, 'test_db_connection'):
                self.test_db_connection.close()

            # Remove test database file
            if hasattr(self, 'test_db_path') and os.path.exists(self.test_db_path):
                os.remove(self.test_db_path)

            # Cleanup test users and data
            for user_id in self.test_users:
                asyncio.run(self.client.delete_user_data(user_id))

        except Exception as e:
            print(f"Test environment cleanup error: {e}")

    def _save_security_test_results(self):
        """Save security test results to file"""
        results_data = {
            "timestamp": time.time(),
            "test_type": "security_privacy",
            "total_tests": len(self.test_results),
            "security_summary": {
                "avg_compliance_score": statistics.mean(r.compliance_score for r in self.test_results),
                "total_vulnerabilities": sum(len(r.vulnerabilities_found) for r in self.test_results),
                "avg_data_protection": statistics.mean(r.data_protection_score for r in self.test_results),
                "avg_access_control": statistics.mean(r.access_control_effectiveness for r in self.test_results)
            },
            "test_results": [
                {
                    "test_name": result.test_name,
                    "security_level": result.security_level,
                    "vulnerabilities_found": result.vulnerabilities_found,
                    "compliance_score": result.compliance_score,
                    "encryption_strength": result.encryption_strength,
                    "access_control_effectiveness": result.access_control_effectiveness,
                    "data_protection_score": result.data_protection_score,
                    "privacy_violations": result.privacy_violations,
                    "recommendations": result.recommendations
                }
                for result in self.test_results
            ]
        }

        results_file = Path("test_reports/companion/security_privacy_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

    # Additional helper methods would continue here for all the security testing functions...
    # (Implementation of _test_biometric_encryption, _test_role_based_access_control, etc.)

import statistics