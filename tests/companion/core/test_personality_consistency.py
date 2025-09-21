"""
Personality Consistency Testing for AI Companion System
Validates personality trait stability across sessions and interactions
"""

import pytest
import asyncio
import json
import statistics
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from tests.companion.fixtures.personality_fixtures import *
from tests.companion.utils.companion_client import CompanionTestClient
from tests.companion.utils.personality_analyzer import PersonalityAnalyzer
from tests.companion.utils.metrics_collector import CompanionMetricsCollector

@dataclass
class PersonalityTestResult:
    """Container for personality test results"""
    session_id: str
    timestamp: float
    traits: Dict[str, float]
    interaction_style: str
    consistency_score: float
    deviation_metrics: Dict[str, float]

@pytest.mark.companion_core
@pytest.mark.personality
class TestPersonalityConsistency:
    """Test personality consistency across sessions and interactions"""

    @pytest.fixture(autouse=True)
    def setup_personality_test(self, companion_test_client, personality_analyzer, metrics_collector):
        """Setup personality testing environment"""
        self.client = companion_test_client
        self.analyzer = personality_analyzer
        self.metrics = metrics_collector
        self.test_sessions = []
        self.baseline_personality = None

        yield

        # Cleanup test sessions
        for session_id in self.test_sessions:
            asyncio.run(self.client.cleanup_session(session_id))

    async def test_ocean_trait_consistency(self, professional_cybersecurity_persona):
        """Test OCEAN personality trait consistency across multiple sessions"""
        persona = professional_cybersecurity_persona
        target_traits = persona["traits"]

        # Create multiple sessions with same persona
        session_count = 10
        session_results = []

        for i in range(session_count):
            session_id = f"ocean_test_session_{i}"
            self.test_sessions.append(session_id)

            # Initialize session with persona
            await self.client.initialize_session(session_id, persona)

            # Conduct interaction to measure personality expression
            interactions = await self._conduct_personality_assessment_interactions(session_id)

            # Analyze personality traits from interactions
            measured_traits = await self.analyzer.extract_personality_traits(interactions)
            consistency_score = self._calculate_trait_consistency(target_traits, measured_traits)

            result = PersonalityTestResult(
                session_id=session_id,
                timestamp=time.time(),
                traits=measured_traits,
                interaction_style=persona["interaction_style"],
                consistency_score=consistency_score,
                deviation_metrics=self._calculate_trait_deviations(target_traits, measured_traits)
            )
            session_results.append(result)

            # Record metrics
            self.metrics.record_personality_measurement(session_id, measured_traits, consistency_score)

        # Analyze overall consistency
        overall_consistency = self._analyze_cross_session_consistency(session_results, target_traits)
        trait_stability = self._calculate_trait_stability(session_results)

        # Performance assertions
        assert overall_consistency >= 0.90, f"Personality consistency {overall_consistency:.2f} below 90% threshold"

        for trait, stability in trait_stability.items():
            assert stability >= 0.85, f"Trait {trait} stability {stability:.2f} below 85% threshold"

        # Log detailed results
        await self._log_personality_test_results("ocean_trait_consistency", session_results, overall_consistency)

    async def test_interaction_style_consistency(self, test_personas):
        """Test consistency of interaction style across different conversation types"""
        for persona_name, persona in test_personas.items():
            session_id = f"style_test_{persona_name}"
            self.test_sessions.append(session_id)

            await self.client.initialize_session(session_id, persona)

            # Test different conversation types
            conversation_types = [
                "technical_discussion",
                "casual_conversation",
                "problem_solving",
                "emotional_support",
                "professional_briefing"
            ]

            style_consistency_scores = []

            for conv_type in conversation_types:
                # Conduct conversation of specific type
                interactions = await self._conduct_typed_conversation(session_id, conv_type)

                # Analyze interaction style
                measured_style = await self.analyzer.extract_interaction_style(interactions)

                # Compare with expected style
                style_consistency = self._compare_interaction_styles(
                    persona["interaction_style"],
                    measured_style
                )
                style_consistency_scores.append(style_consistency)

                self.metrics.record_interaction_style_measurement(
                    session_id, conv_type, measured_style, style_consistency
                )

            # Validate style consistency across conversation types
            avg_style_consistency = statistics.mean(style_consistency_scores)
            style_variance = statistics.stdev(style_consistency_scores) if len(style_consistency_scores) > 1 else 0

            assert avg_style_consistency >= 0.85, f"Style consistency {avg_style_consistency:.2f} below threshold for {persona_name}"
            assert style_variance <= 0.15, f"Style variance {style_variance:.2f} too high for {persona_name}"

    async def test_emotional_state_consistency(self, professional_cybersecurity_persona):
        """Test emotional response consistency for similar situations"""
        persona = professional_cybersecurity_persona
        session_id = "emotional_consistency_test"
        self.test_sessions.append(session_id)

        await self.client.initialize_session(session_id, persona)

        # Test emotional responses to repeated similar scenarios
        emotional_scenarios = [
            {
                "scenario": "user_frustration_complex_investigation",
                "trigger": "This investigation is taking forever and I'm not making progress",
                "expected_emotion": "empathetic_supportive",
                "repetitions": 5
            },
            {
                "scenario": "user_excitement_major_discovery",
                "trigger": "I found something huge! This could be the breakthrough we needed",
                "expected_emotion": "enthusiastic_congratulatory",
                "repetitions": 5
            },
            {
                "scenario": "user_stress_security_incident",
                "trigger": "We have a critical security breach happening right now",
                "expected_emotion": "calm_focused",
                "repetitions": 5
            }
        ]

        for scenario_config in emotional_scenarios:
            emotional_responses = []

            for rep in range(scenario_config["repetitions"]):
                # Present scenario trigger
                response = await self.client.send_message(
                    session_id,
                    scenario_config["trigger"],
                    context={"scenario": scenario_config["scenario"], "repetition": rep}
                )

                # Analyze emotional content of response
                emotional_analysis = await self.analyzer.analyze_emotional_response(response)
                emotional_responses.append(emotional_analysis)

                # Brief pause between repetitions
                await asyncio.sleep(1)

            # Calculate emotional consistency
            emotional_consistency = self._calculate_emotional_consistency(
                emotional_responses,
                scenario_config["expected_emotion"]
            )

            # Validate emotional stability
            assert emotional_consistency >= 0.80, f"Emotional consistency {emotional_consistency:.2f} below threshold for {scenario_config['scenario']}"

            self.metrics.record_emotional_consistency(
                session_id, scenario_config["scenario"], emotional_consistency
            )

    async def test_memory_integration_personality(self, analytical_researcher_persona):
        """Test personality consistency when integrated with memory system"""
        persona = analytical_researcher_persona
        session_id = "memory_personality_test"
        self.test_sessions.append(session_id)

        await self.client.initialize_session(session_id, persona)

        # Phase 1: Establish baseline personality with memory storage
        baseline_interactions = await self._conduct_memory_building_interactions(session_id)
        baseline_personality = await self.analyzer.extract_personality_traits(baseline_interactions)

        # Phase 2: New session loading same memory
        await self.client.end_session(session_id)
        await asyncio.sleep(2)  # Brief pause

        # Reload session with same user memory
        await self.client.initialize_session(session_id, persona, load_memory=True)

        # Phase 3: Test personality consistency with loaded memory
        memory_loaded_interactions = await self._conduct_personality_assessment_interactions(session_id)
        memory_loaded_personality = await self.analyzer.extract_personality_traits(memory_loaded_interactions)

        # Calculate personality consistency across memory boundary
        memory_consistency = self._calculate_trait_consistency(
            baseline_personality,
            memory_loaded_personality
        )

        # Test memory-influenced responses
        memory_influenced_responses = await self._test_memory_influenced_personality(session_id)
        memory_influence_consistency = await self.analyzer.assess_memory_personality_integration(
            memory_influenced_responses, persona
        )

        # Validate memory integration doesn't affect personality consistency
        assert memory_consistency >= 0.88, f"Personality consistency across memory load {memory_consistency:.2f} below threshold"
        assert memory_influence_consistency >= 0.85, f"Memory-influenced personality consistency {memory_influence_consistency:.2f} below threshold"

        self.metrics.record_memory_personality_integration(
            session_id, memory_consistency, memory_influence_consistency
        )

    async def test_long_term_personality_stability(self, friendly_assistant_persona):
        """Test personality stability over extended interactions"""
        persona = friendly_assistant_persona
        session_id = "long_term_stability_test"
        self.test_sessions.append(session_id)

        await self.client.initialize_session(session_id, persona)

        # Conduct extended interaction session (100+ exchanges)
        interaction_phases = [
            {"phase": "initial", "exchanges": 25},
            {"phase": "middle", "exchanges": 50},
            {"phase": "final", "exchanges": 25}
        ]

        personality_measurements = []

        for phase_config in interaction_phases:
            # Conduct interactions for this phase
            phase_interactions = await self._conduct_extended_interactions(
                session_id,
                phase_config["exchanges"],
                phase_config["phase"]
            )

            # Measure personality at end of phase
            phase_personality = await self.analyzer.extract_personality_traits(phase_interactions)

            personality_measurements.append({
                "phase": phase_config["phase"],
                "personality": phase_personality,
                "timestamp": time.time(),
                "interaction_count": phase_config["exchanges"]
            })

            self.metrics.record_long_term_personality_measurement(
                session_id, phase_config["phase"], phase_personality
            )

        # Analyze personality drift over time
        personality_drift = self._calculate_personality_drift(personality_measurements)
        temporal_consistency = self._calculate_temporal_consistency(personality_measurements)

        # Validate long-term stability
        assert personality_drift <= 0.10, f"Personality drift {personality_drift:.3f} exceeds 10% threshold"
        assert temporal_consistency >= 0.85, f"Temporal consistency {temporal_consistency:.2f} below 85% threshold"

        await self._log_long_term_stability_results(session_id, personality_measurements, personality_drift)

    # Helper Methods

    async def _conduct_personality_assessment_interactions(self, session_id: str) -> List[Dict]:
        """Conduct standardized interactions to assess personality"""
        assessment_prompts = [
            "Tell me about your approach to cybersecurity analysis",
            "How do you handle complex technical problems?",
            "What's your communication style when explaining technical concepts?",
            "How do you react when facing urgent security incidents?",
            "Describe your methodology for threat investigation"
        ]

        interactions = []
        for prompt in assessment_prompts:
            response = await self.client.send_message(session_id, prompt)
            interactions.append({
                "prompt": prompt,
                "response": response,
                "timestamp": time.time()
            })
            await asyncio.sleep(0.5)  # Brief pause between interactions

        return interactions

    async def _conduct_typed_conversation(self, session_id: str, conversation_type: str) -> List[Dict]:
        """Conduct conversation of specific type to test interaction style"""
        conversation_prompts = {
            "technical_discussion": [
                "Can you explain the technical details of this malware analysis?",
                "What are the specific steps for investigating this network intrusion?",
                "How would you technically approach this forensic challenge?"
            ],
            "casual_conversation": [
                "How are you doing today?",
                "What do you think about the latest cybersecurity trends?",
                "Any interesting discoveries in your recent work?"
            ],
            "problem_solving": [
                "I'm stuck on this analysis - can you help me think through it?",
                "What approach would you recommend for this investigation?",
                "How should we prioritize these security findings?"
            ],
            "emotional_support": [
                "I'm feeling overwhelmed by this complex investigation",
                "This security incident is really stressing me out",
                "I'm not sure I'm qualified to handle this analysis"
            ],
            "professional_briefing": [
                "Please brief me on the current threat landscape",
                "I need a professional assessment of this security posture",
                "Can you provide an executive summary of these findings?"
            ]
        }

        prompts = conversation_prompts.get(conversation_type, [])
        interactions = []

        for prompt in prompts:
            response = await self.client.send_message(session_id, prompt)
            interactions.append({
                "prompt": prompt,
                "response": response,
                "conversation_type": conversation_type,
                "timestamp": time.time()
            })
            await asyncio.sleep(0.5)

        return interactions

    async def _conduct_memory_building_interactions(self, session_id: str) -> List[Dict]:
        """Conduct interactions that build memory context"""
        memory_building_prompts = [
            "I prefer detailed technical explanations over summaries",
            "My expertise level in network security is advanced",
            "I usually work on APT investigations and threat hunting",
            "I like to see the data before accepting conclusions",
            "My communication style preference is analytical and evidence-based"
        ]

        interactions = []
        for prompt in memory_building_prompts:
            response = await self.client.send_message(session_id, prompt)
            interactions.append({
                "prompt": prompt,
                "response": response,
                "memory_building": True,
                "timestamp": time.time()
            })
            await asyncio.sleep(1)  # Allow memory processing time

        return interactions

    async def _test_memory_influenced_personality(self, session_id: str) -> List[Dict]:
        """Test personality responses influenced by stored memory"""
        memory_test_prompts = [
            "How should we approach this new investigation?",  # Should reflect stored preferences
            "Can you explain this technical concept?",        # Should use preferred detail level
            "What's your analysis of this evidence?",         # Should reflect analytical style
        ]

        responses = []
        for prompt in memory_test_prompts:
            response = await self.client.send_message(session_id, prompt)
            responses.append({
                "prompt": prompt,
                "response": response,
                "memory_influenced": True,
                "timestamp": time.time()
            })
            await asyncio.sleep(0.5)

        return responses

    async def _conduct_extended_interactions(self, session_id: str, exchange_count: int, phase: str) -> List[Dict]:
        """Conduct extended interaction sequence"""
        base_prompts = [
            "What do you think about this analysis approach?",
            "How would you investigate this further?",
            "Can you help me understand this pattern?",
            "What's your assessment of this threat?",
            "How should we proceed with this investigation?"
        ]

        interactions = []
        for i in range(exchange_count):
            prompt = f"{base_prompts[i % len(base_prompts)]} (exchange {i+1} in {phase} phase)"
            response = await self.client.send_message(session_id, prompt)
            interactions.append({
                "prompt": prompt,
                "response": response,
                "phase": phase,
                "exchange_number": i + 1,
                "timestamp": time.time()
            })

            # Vary pause time to simulate realistic interaction patterns
            pause_time = 0.3 + (i % 3) * 0.2  # 0.3-0.9 second pauses
            await asyncio.sleep(pause_time)

        return interactions

    def _calculate_trait_consistency(self, target_traits: Dict[str, float], measured_traits: Dict[str, float]) -> float:
        """Calculate consistency score between target and measured personality traits"""
        if not target_traits or not measured_traits:
            return 0.0

        consistency_scores = []
        for trait in target_traits:
            if trait in measured_traits:
                target_value = target_traits[trait]
                measured_value = measured_traits[trait]

                # Calculate percentage similarity
                difference = abs(target_value - measured_value)
                similarity = 1.0 - (difference / 1.0)  # Normalize to 0-1 range
                consistency_scores.append(max(0.0, similarity))

        return statistics.mean(consistency_scores) if consistency_scores else 0.0

    def _calculate_trait_deviations(self, target_traits: Dict[str, float], measured_traits: Dict[str, float]) -> Dict[str, float]:
        """Calculate deviation metrics for each personality trait"""
        deviations = {}
        for trait in target_traits:
            if trait in measured_traits:
                deviation = abs(target_traits[trait] - measured_traits[trait])
                deviations[trait] = deviation
        return deviations

    def _analyze_cross_session_consistency(self, session_results: List[PersonalityTestResult], target_traits: Dict[str, float]) -> float:
        """Analyze personality consistency across multiple sessions"""
        if not session_results:
            return 0.0

        # Calculate consistency for each trait across sessions
        trait_consistencies = {}

        for trait in target_traits:
            trait_measurements = [result.traits.get(trait, 0.0) for result in session_results]

            if trait_measurements:
                # Calculate how consistent measurements are with target
                target_value = target_traits[trait]
                deviations = [abs(measurement - target_value) for measurement in trait_measurements]
                avg_deviation = statistics.mean(deviations)
                consistency = 1.0 - avg_deviation  # Convert deviation to consistency
                trait_consistencies[trait] = max(0.0, consistency)

        return statistics.mean(trait_consistencies.values()) if trait_consistencies else 0.0

    def _calculate_trait_stability(self, session_results: List[PersonalityTestResult]) -> Dict[str, float]:
        """Calculate stability score for each trait across sessions"""
        if len(session_results) < 2:
            return {}

        trait_stability = {}
        all_traits = set()
        for result in session_results:
            all_traits.update(result.traits.keys())

        for trait in all_traits:
            trait_values = [result.traits.get(trait, 0.0) for result in session_results if trait in result.traits]

            if len(trait_values) >= 2:
                # Calculate coefficient of variation (inverse of stability)
                if statistics.mean(trait_values) > 0:
                    cv = statistics.stdev(trait_values) / statistics.mean(trait_values)
                    stability = 1.0 - min(cv, 1.0)  # Convert to stability score
                    trait_stability[trait] = max(0.0, stability)

        return trait_stability

    def _compare_interaction_styles(self, expected_style: str, measured_style: Dict[str, Any]) -> float:
        """Compare expected interaction style with measured style"""
        style_mappings = {
            "formal_expert": {
                "formality": 0.8,
                "technical_depth": 0.9,
                "emotional_warmth": 0.6,
                "directness": 0.8
            },
            "casual_supportive": {
                "formality": 0.4,
                "technical_depth": 0.7,
                "emotional_warmth": 0.9,
                "directness": 0.6
            },
            "analytical_detailed": {
                "formality": 0.7,
                "technical_depth": 1.0,
                "emotional_warmth": 0.5,
                "directness": 0.9
            }
        }

        expected_metrics = style_mappings.get(expected_style, {})
        if not expected_metrics:
            return 0.0

        consistency_scores = []
        for metric, expected_value in expected_metrics.items():
            measured_value = measured_style.get(metric, 0.0)
            difference = abs(expected_value - measured_value)
            similarity = 1.0 - difference
            consistency_scores.append(max(0.0, similarity))

        return statistics.mean(consistency_scores) if consistency_scores else 0.0

    def _calculate_emotional_consistency(self, emotional_responses: List[Dict], expected_emotion: str) -> float:
        """Calculate consistency of emotional responses"""
        if not emotional_responses:
            return 0.0

        # Analyze emotional consistency across responses
        emotion_scores = []
        for response in emotional_responses:
            emotion_match = response.get("primary_emotion") == expected_emotion
            emotion_intensity = response.get("emotion_intensity", 0.0)
            appropriateness = response.get("appropriateness_score", 0.0)

            # Combine factors for overall emotional consistency
            consistency = (
                (1.0 if emotion_match else 0.0) * 0.5 +
                emotion_intensity * 0.3 +
                appropriateness * 0.2
            )
            emotion_scores.append(consistency)

        return statistics.mean(emotion_scores)

    def _calculate_personality_drift(self, personality_measurements: List[Dict]) -> float:
        """Calculate personality drift over time"""
        if len(personality_measurements) < 2:
            return 0.0

        initial_personality = personality_measurements[0]["personality"]
        final_personality = personality_measurements[-1]["personality"]

        # Calculate drift for each trait
        trait_drifts = []
        for trait in initial_personality:
            if trait in final_personality:
                initial_value = initial_personality[trait]
                final_value = final_personality[trait]
                drift = abs(final_value - initial_value)
                trait_drifts.append(drift)

        return statistics.mean(trait_drifts) if trait_drifts else 0.0

    def _calculate_temporal_consistency(self, personality_measurements: List[Dict]) -> float:
        """Calculate temporal consistency across personality measurements"""
        if len(personality_measurements) < 3:
            return 1.0  # Not enough data points

        # Calculate consistency between consecutive measurements
        consistency_scores = []

        for i in range(1, len(personality_measurements)):
            prev_personality = personality_measurements[i-1]["personality"]
            curr_personality = personality_measurements[i]["personality"]

            consistency = self._calculate_trait_consistency(prev_personality, curr_personality)
            consistency_scores.append(consistency)

        return statistics.mean(consistency_scores) if consistency_scores else 0.0

    async def _log_personality_test_results(self, test_name: str, results: List[PersonalityTestResult], overall_consistency: float):
        """Log detailed personality test results"""
        log_data = {
            "test_name": test_name,
            "timestamp": time.time(),
            "overall_consistency": overall_consistency,
            "session_count": len(results),
            "individual_results": [
                {
                    "session_id": result.session_id,
                    "consistency_score": result.consistency_score,
                    "traits": result.traits,
                    "deviations": result.deviation_metrics
                }
                for result in results
            ]
        }

        # Write to test results file
        results_file = Path("test_reports/companion/personality_consistency.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "a") as f:
            f.write(json.dumps(log_data) + "\n")

    async def _log_long_term_stability_results(self, session_id: str, measurements: List[Dict], drift: float):
        """Log long-term stability test results"""
        log_data = {
            "test_type": "long_term_stability",
            "session_id": session_id,
            "timestamp": time.time(),
            "personality_drift": drift,
            "measurements": measurements
        }

        results_file = Path("test_reports/companion/long_term_stability.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "a") as f:
            f.write(json.dumps(log_data) + "\n")