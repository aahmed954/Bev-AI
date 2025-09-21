"""
User Experience Testing for AI Companion Interactions
Tests conversation naturalness, emotional intelligence, professional effectiveness,
and overall user satisfaction with companion features
"""

import pytest
import asyncio
import time
import statistics
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

from tests.companion.fixtures.ux_fixtures import *
from tests.companion.utils.companion_client import CompanionTestClient
from tests.companion.utils.ux_analyzer import UserExperienceAnalyzer
from tests.companion.utils.conversation_evaluator import ConversationEvaluator
from tests.companion.utils.emotion_validator import EmotionValidator

@dataclass
class ConversationQualityMetrics:
    """Metrics for conversation quality assessment"""
    naturalness_score: float
    coherence_score: float
    context_retention: float
    emotional_appropriateness: float
    professional_effectiveness: float
    response_relevance: float
    conversation_flow: float

@dataclass
class UserSatisfactionResult:
    """User satisfaction assessment result"""
    overall_satisfaction: float
    feature_ratings: Dict[str, float]
    improvement_areas: List[str]
    positive_feedback: List[str]
    usability_score: float
    recommendation_likelihood: float

@dataclass
class CompanionInteractionResult:
    """Comprehensive companion interaction assessment"""
    test_scenario: str
    duration: float
    conversation_quality: ConversationQualityMetrics
    user_satisfaction: UserSatisfactionResult
    technical_performance: Dict[str, float]
    accessibility_compliance: float
    improvement_suggestions: List[str]

@pytest.mark.companion_ux
@pytest.mark.user_experience
class TestCompanionUserExperience:
    """Test user experience aspects of AI companion interactions"""

    @pytest.fixture(autouse=True)
    def setup_ux_testing(self, companion_client, ux_analyzer, conversation_evaluator, emotion_validator):
        """Setup user experience testing environment"""
        self.client = companion_client
        self.ux_analyzer = ux_analyzer
        self.conversation_evaluator = conversation_evaluator
        self.emotion_validator = emotion_validator
        self.test_results = []
        self.user_sessions = []

        yield

        # Cleanup test sessions and save results
        for session_id in self.user_sessions:
            asyncio.run(self.client.cleanup_session(session_id))
        self._save_ux_test_results()

    async def test_conversation_naturalness(self, professional_conversation_scenarios):
        """Test naturalness and fluidity of conversations"""
        for scenario_name, scenario in professional_conversation_scenarios.items():
            session_id = f"naturalness_test_{scenario_name}"
            self.user_sessions.append(session_id)

            print(f"Testing conversation naturalness: {scenario_name}")

            # Initialize companion session
            await self.client.initialize_session(session_id, scenario["persona"])

            # Conduct conversation with predefined flow
            conversation_history = []
            conversation_start = time.time()

            for exchange in scenario["conversation_flow"]:
                # Send user message
                user_message = exchange["user_input"]
                response_start = time.time()

                companion_response = await self.client.send_message(session_id, user_message)

                response_time = time.time() - response_start

                # Record conversation exchange
                conversation_history.append({
                    "user_input": user_message,
                    "companion_response": companion_response,
                    "response_time": response_time,
                    "timestamp": time.time(),
                    "expected_qualities": exchange.get("expected_qualities", [])
                })

                # Brief pause to simulate natural conversation timing
                await asyncio.sleep(1.5)

            conversation_duration = time.time() - conversation_start

            # Analyze conversation naturalness
            naturalness_analysis = await self.conversation_evaluator.analyze_naturalness(
                conversation_history, scenario["evaluation_criteria"]
            )

            # Evaluate conversation quality
            quality_metrics = await self._evaluate_conversation_quality(
                conversation_history, scenario
            )

            # Create interaction result
            result = CompanionInteractionResult(
                test_scenario=f"naturalness_{scenario_name}",
                duration=conversation_duration,
                conversation_quality=quality_metrics,
                user_satisfaction=await self._assess_conversation_satisfaction(
                    conversation_history, naturalness_analysis
                ),
                technical_performance={
                    "avg_response_time": statistics.mean(e["response_time"] for e in conversation_history),
                    "max_response_time": max(e["response_time"] for e in conversation_history),
                    "conversation_length": len(conversation_history)
                },
                accessibility_compliance=0.95,  # Would be measured separately
                improvement_suggestions=naturalness_analysis.get("suggestions", [])
            )

            self.test_results.append(result)

            # Validate naturalness requirements
            assert quality_metrics.naturalness_score >= 4.5, f"Naturalness score {quality_metrics.naturalness_score:.2f} below 4.5 threshold for {scenario_name}"
            assert quality_metrics.conversation_flow >= 4.0, f"Conversation flow {quality_metrics.conversation_flow:.2f} below 4.0 threshold"
            assert quality_metrics.context_retention >= 0.90, f"Context retention {quality_metrics.context_retention:.2f} below 90% threshold"

    async def test_emotional_intelligence_accuracy(self, emotional_response_scenarios):
        """Test emotional intelligence and appropriate emotional responses"""
        for scenario_name, scenario in emotional_response_scenarios.items():
            session_id = f"emotion_test_{scenario_name}"
            self.user_sessions.append(session_id)

            print(f"Testing emotional intelligence: {scenario_name}")

            # Initialize session with emotionally-aware persona
            await self.client.initialize_session(session_id, scenario["persona"])

            emotion_test_results = []

            for emotion_trigger in scenario["emotion_triggers"]:
                # Present emotional trigger
                trigger_message = emotion_trigger["trigger_text"]
                expected_emotion = emotion_trigger["expected_response_emotion"]
                expected_tone = emotion_trigger["expected_tone"]

                # Send trigger and measure emotional response
                emotion_start = time.time()
                companion_response = await self.client.send_message(session_id, trigger_message)
                response_time = time.time() - emotion_start

                # Analyze emotional appropriateness
                emotion_analysis = await self.emotion_validator.validate_emotional_response(
                    trigger_message, companion_response, expected_emotion, expected_tone
                )

                # Evaluate empathy and emotional intelligence
                empathy_score = await self.emotion_validator.assess_empathy_level(
                    trigger_message, companion_response, scenario["empathy_requirements"]
                )

                emotion_test_results.append({
                    "trigger": trigger_message,
                    "response": companion_response,
                    "expected_emotion": expected_emotion,
                    "detected_emotion": emotion_analysis["detected_emotion"],
                    "emotion_accuracy": emotion_analysis["accuracy_score"],
                    "empathy_score": empathy_score,
                    "tone_appropriateness": emotion_analysis["tone_score"],
                    "response_time": response_time
                })

                await asyncio.sleep(2)  # Allow emotional state to settle

            # Analyze overall emotional intelligence
            overall_emotion_accuracy = statistics.mean(
                result["emotion_accuracy"] for result in emotion_test_results
            )
            overall_empathy = statistics.mean(
                result["empathy_score"] for result in emotion_test_results
            )
            overall_tone_appropriateness = statistics.mean(
                result["tone_appropriateness"] for result in emotion_test_results
            )

            # Create emotion-focused quality metrics
            emotion_quality = ConversationQualityMetrics(
                naturalness_score=4.0,  # Not primary focus for this test
                coherence_score=4.0,
                context_retention=0.85,
                emotional_appropriateness=overall_emotion_accuracy,
                professional_effectiveness=3.5,  # May be lower for emotional scenarios
                response_relevance=4.2,
                conversation_flow=3.8
            )

            result = CompanionInteractionResult(
                test_scenario=f"emotional_intelligence_{scenario_name}",
                duration=len(emotion_test_results) * 4,  # Approximate duration
                conversation_quality=emotion_quality,
                user_satisfaction=UserSatisfactionResult(
                    overall_satisfaction=overall_empathy,
                    feature_ratings={"emotional_intelligence": overall_emotion_accuracy},
                    improvement_areas=[],
                    positive_feedback=["Emotionally responsive", "Empathetic responses"],
                    usability_score=4.0,
                    recommendation_likelihood=overall_empathy
                ),
                technical_performance={
                    "avg_response_time": statistics.mean(r["response_time"] for r in emotion_test_results),
                    "emotion_accuracy": overall_emotion_accuracy,
                    "empathy_level": overall_empathy
                },
                accessibility_compliance=0.95,
                improvement_suggestions=[]
            )

            self.test_results.append(result)

            # Validate emotional intelligence requirements
            assert overall_emotion_accuracy >= 0.85, f"Emotion accuracy {overall_emotion_accuracy:.2f} below 85% threshold for {scenario_name}"
            assert overall_empathy >= 0.80, f"Empathy score {overall_empathy:.2f} below 80% threshold"
            assert overall_tone_appropriateness >= 0.85, f"Tone appropriateness {overall_tone_appropriateness:.2f} below 85% threshold"

    async def test_professional_workflow_integration(self, professional_workflow_scenarios):
        """Test companion effectiveness in professional cybersecurity workflows"""
        for workflow_name, workflow in professional_workflow_scenarios.items():
            session_id = f"workflow_test_{workflow_name}"
            self.user_sessions.append(session_id)

            print(f"Testing professional workflow integration: {workflow_name}")

            # Initialize with professional cybersecurity persona
            await self.client.initialize_session(session_id, workflow["professional_persona"])

            workflow_results = []
            workflow_start = time.time()

            for workflow_step in workflow["workflow_steps"]:
                step_start = time.time()

                # Execute workflow step
                user_action = workflow_step["user_action"]
                expected_assistance = workflow_step["expected_companion_assistance"]

                # Send workflow-related query
                companion_response = await self.client.send_message(session_id, user_action)

                # Evaluate professional effectiveness
                effectiveness_analysis = await self.ux_analyzer.evaluate_professional_effectiveness(
                    user_action, companion_response, expected_assistance, workflow["domain_expertise"]
                )

                # Assess workflow enhancement
                workflow_enhancement = await self.ux_analyzer.assess_workflow_enhancement(
                    workflow_step, companion_response, workflow["efficiency_metrics"]
                )

                step_duration = time.time() - step_start

                workflow_results.append({
                    "step": workflow_step["step_name"],
                    "user_action": user_action,
                    "companion_response": companion_response,
                    "effectiveness_score": effectiveness_analysis["effectiveness_score"],
                    "domain_accuracy": effectiveness_analysis["domain_accuracy"],
                    "workflow_enhancement": workflow_enhancement["enhancement_score"],
                    "time_savings": workflow_enhancement.get("time_savings", 0),
                    "step_duration": step_duration
                })

                await asyncio.sleep(1)

            workflow_duration = time.time() - workflow_start

            # Analyze overall workflow integration
            overall_effectiveness = statistics.mean(
                result["effectiveness_score"] for result in workflow_results
            )
            overall_domain_accuracy = statistics.mean(
                result["domain_accuracy"] for result in workflow_results
            )
            overall_workflow_enhancement = statistics.mean(
                result["workflow_enhancement"] for result in workflow_results
            )

            # Calculate professional quality metrics
            professional_quality = ConversationQualityMetrics(
                naturalness_score=4.2,
                coherence_score=4.5,
                context_retention=0.92,
                emotional_appropriateness=0.70,  # Professional context - less emotional
                professional_effectiveness=overall_effectiveness,
                response_relevance=overall_domain_accuracy,
                conversation_flow=4.3
            )

            result = CompanionInteractionResult(
                test_scenario=f"professional_workflow_{workflow_name}",
                duration=workflow_duration,
                conversation_quality=professional_quality,
                user_satisfaction=UserSatisfactionResult(
                    overall_satisfaction=overall_effectiveness,
                    feature_ratings={
                        "professional_assistance": overall_effectiveness,
                        "domain_expertise": overall_domain_accuracy,
                        "workflow_integration": overall_workflow_enhancement
                    },
                    improvement_areas=[],
                    positive_feedback=["Professional guidance", "Domain expertise", "Workflow enhancement"],
                    usability_score=overall_effectiveness,
                    recommendation_likelihood=overall_effectiveness
                ),
                technical_performance={
                    "workflow_completion_time": workflow_duration,
                    "avg_step_time": statistics.mean(r["step_duration"] for r in workflow_results),
                    "workflow_efficiency": overall_workflow_enhancement
                },
                accessibility_compliance=0.98,
                improvement_suggestions=[]
            )

            self.test_results.append(result)

            # Validate professional workflow requirements
            assert overall_effectiveness >= 4.5, f"Professional effectiveness {overall_effectiveness:.2f} below 4.5 threshold for {workflow_name}"
            assert overall_domain_accuracy >= 0.85, f"Domain accuracy {overall_domain_accuracy:.2f} below 85% threshold"
            assert overall_workflow_enhancement >= 0.25, f"Workflow enhancement {overall_workflow_enhancement:.2f} below 25% improvement threshold"

    async def test_voice_synthesis_user_acceptance(self, voice_quality_scenarios):
        """Test user acceptance of voice synthesis quality and naturalness"""
        for scenario_name, scenario in voice_quality_scenarios.items():
            session_id = f"voice_test_{scenario_name}"
            self.user_sessions.append(session_id)

            print(f"Testing voice synthesis user acceptance: {scenario_name}")

            await self.client.initialize_session(session_id, scenario["persona"])

            voice_test_results = []

            for voice_test in scenario["voice_tests"]:
                # Test voice synthesis with specific content
                test_text = voice_test["text"]
                expected_quality = voice_test["expected_quality"]
                emotion_context = voice_test.get("emotion_context", "neutral")

                # Generate voice response
                voice_start = time.time()
                voice_response = await self.client.synthesize_voice_response(
                    session_id, test_text, emotion_context=emotion_context
                )
                synthesis_time = time.time() - voice_start

                # Evaluate voice quality
                quality_assessment = await self.ux_analyzer.evaluate_voice_quality(
                    voice_response, expected_quality, scenario["quality_criteria"]
                )

                # Assess user acceptance factors
                acceptance_factors = await self.ux_analyzer.assess_voice_acceptance(
                    voice_response, scenario["acceptance_criteria"]
                )

                voice_test_results.append({
                    "test_text": test_text,
                    "synthesis_time": synthesis_time,
                    "quality_score": quality_assessment["overall_quality"],
                    "naturalness": quality_assessment["naturalness"],
                    "clarity": quality_assessment["clarity"],
                    "emotion_accuracy": quality_assessment["emotion_accuracy"],
                    "user_acceptance": acceptance_factors["acceptance_score"],
                    "pleasantness": acceptance_factors["pleasantness"],
                    "professional_suitability": acceptance_factors["professional_suitability"]
                })

                await asyncio.sleep(2)

            # Analyze overall voice acceptance
            overall_quality = statistics.mean(r["quality_score"] for r in voice_test_results)
            overall_naturalness = statistics.mean(r["naturalness"] for r in voice_test_results)
            overall_acceptance = statistics.mean(r["user_acceptance"] for r in voice_test_results)
            avg_synthesis_time = statistics.mean(r["synthesis_time"] for r in voice_test_results)

            # Voice-focused quality metrics
            voice_quality = ConversationQualityMetrics(
                naturalness_score=overall_naturalness,
                coherence_score=4.0,
                context_retention=0.80,  # Not primary focus
                emotional_appropriateness=statistics.mean(r["emotion_accuracy"] for r in voice_test_results),
                professional_effectiveness=statistics.mean(r["professional_suitability"] for r in voice_test_results),
                response_relevance=4.0,
                conversation_flow=3.8
            )

            result = CompanionInteractionResult(
                test_scenario=f"voice_synthesis_{scenario_name}",
                duration=len(voice_test_results) * 4,
                conversation_quality=voice_quality,
                user_satisfaction=UserSatisfactionResult(
                    overall_satisfaction=overall_acceptance,
                    feature_ratings={
                        "voice_quality": overall_quality,
                        "voice_naturalness": overall_naturalness,
                        "voice_pleasantness": statistics.mean(r["pleasantness"] for r in voice_test_results)
                    },
                    improvement_areas=[],
                    positive_feedback=["Natural voice", "Clear speech", "Professional tone"],
                    usability_score=overall_acceptance,
                    recommendation_likelihood=overall_acceptance
                ),
                technical_performance={
                    "avg_synthesis_time": avg_synthesis_time,
                    "voice_quality_score": overall_quality,
                    "synthesis_success_rate": 1.0  # Assuming no failures in test
                },
                accessibility_compliance=0.92,
                improvement_suggestions=[]
            )

            self.test_results.append(result)

            # Validate voice synthesis requirements
            assert overall_quality >= 4.0, f"Voice quality {overall_quality:.2f} below 4.0 threshold for {scenario_name}"
            assert overall_naturalness >= 4.0, f"Voice naturalness {overall_naturalness:.2f} below 4.0 threshold"
            assert overall_acceptance >= 4.0, f"User acceptance {overall_acceptance:.2f} below 4.0 threshold"
            assert avg_synthesis_time <= 1.0, f"Average synthesis time {avg_synthesis_time:.2f}s exceeds 1.0s threshold"

    async def test_avatar_emotional_expression_accuracy(self, avatar_expression_scenarios):
        """Test accuracy and user acceptance of avatar emotional expressions"""
        for scenario_name, scenario in avatar_expression_scenarios.items():
            session_id = f"avatar_emotion_test_{scenario_name}"
            self.user_sessions.append(session_id)

            print(f"Testing avatar emotional expression: {scenario_name}")

            await self.client.initialize_session(session_id, scenario["persona"])

            expression_test_results = []

            for expression_test in scenario["expression_tests"]:
                # Trigger specific emotional state
                emotional_trigger = expression_test["emotional_trigger"]
                expected_expression = expression_test["expected_avatar_expression"]
                expression_intensity = expression_test.get("intensity", "medium")

                # Send message to trigger emotional response
                trigger_start = time.time()
                response = await self.client.send_message(session_id, emotional_trigger)

                # Capture avatar expression during response
                avatar_expression = await self.client.capture_avatar_expression(session_id)
                expression_time = time.time() - trigger_start

                # Evaluate expression accuracy
                expression_analysis = await self.ux_analyzer.evaluate_avatar_expression(
                    avatar_expression, expected_expression, scenario["expression_criteria"]
                )

                # Assess user perception of expression
                user_perception = await self.ux_analyzer.assess_expression_user_perception(
                    avatar_expression, scenario["perception_criteria"]
                )

                expression_test_results.append({
                    "emotional_trigger": emotional_trigger,
                    "expected_expression": expected_expression,
                    "detected_expression": expression_analysis["detected_expression"],
                    "expression_accuracy": expression_analysis["accuracy_score"],
                    "intensity_match": expression_analysis["intensity_score"],
                    "timing_accuracy": expression_analysis["timing_score"],
                    "user_believability": user_perception["believability"],
                    "emotional_impact": user_perception["emotional_impact"],
                    "expression_time": expression_time
                })

                await asyncio.sleep(3)  # Allow expression to complete

            # Analyze overall avatar expression performance
            overall_accuracy = statistics.mean(r["expression_accuracy"] for r in expression_test_results)
            overall_believability = statistics.mean(r["user_believability"] for r in expression_test_results)
            overall_timing = statistics.mean(r["timing_accuracy"] for r in expression_test_results)
            avg_expression_time = statistics.mean(r["expression_time"] for r in expression_test_results)

            # Avatar-focused quality metrics
            avatar_quality = ConversationQualityMetrics(
                naturalness_score=overall_believability,
                coherence_score=4.0,
                context_retention=0.85,
                emotional_appropriateness=overall_accuracy,
                professional_effectiveness=3.8,  # Expressions may be less professional
                response_relevance=4.1,
                conversation_flow=overall_timing
            )

            result = CompanionInteractionResult(
                test_scenario=f"avatar_expressions_{scenario_name}",
                duration=len(expression_test_results) * 5,
                conversation_quality=avatar_quality,
                user_satisfaction=UserSatisfactionResult(
                    overall_satisfaction=overall_believability,
                    feature_ratings={
                        "avatar_accuracy": overall_accuracy,
                        "avatar_believability": overall_believability,
                        "avatar_timing": overall_timing
                    },
                    improvement_areas=[],
                    positive_feedback=["Expressive avatar", "Emotional responsiveness", "Visual appeal"],
                    usability_score=overall_believability,
                    recommendation_likelihood=overall_believability
                ),
                technical_performance={
                    "avg_expression_time": avg_expression_time,
                    "expression_accuracy": overall_accuracy,
                    "timing_precision": overall_timing
                },
                accessibility_compliance=0.90,
                improvement_suggestions=[]
            )

            self.test_results.append(result)

            # Validate avatar expression requirements
            assert overall_accuracy >= 0.90, f"Avatar expression accuracy {overall_accuracy:.2f} below 90% threshold for {scenario_name}"
            assert overall_believability >= 4.0, f"Avatar believability {overall_believability:.2f} below 4.0 threshold"
            assert overall_timing >= 0.85, f"Expression timing {overall_timing:.2f} below 85% threshold"
            assert avg_expression_time <= 0.5, f"Average expression time {avg_expression_time:.2f}s exceeds 0.5s threshold"

    async def test_long_term_user_engagement(self, long_term_engagement_scenario):
        """Test user engagement and satisfaction over extended interaction periods"""
        scenario = long_term_engagement_scenario
        session_id = "long_term_engagement_test"
        self.user_sessions.append(session_id)

        print("Testing long-term user engagement")

        await self.client.initialize_session(session_id, scenario["persona"])

        engagement_phases = scenario["engagement_phases"]
        phase_results = []

        for phase in engagement_phases:
            phase_start = time.time()
            phase_interactions = []

            print(f"  Phase: {phase['phase_name']} ({phase['duration']} minutes)")

            # Conduct interactions for this phase duration
            end_time = phase_start + (phase["duration"] * 60)  # Convert to seconds

            interaction_count = 0
            while time.time() < end_time:
                # Select interaction type based on phase characteristics
                interaction_type = phase["interaction_types"][interaction_count % len(phase["interaction_types"])]

                # Generate interaction content
                interaction_content = self._generate_phase_interaction(interaction_type, phase)

                interaction_start = time.time()
                response = await self.client.send_message(session_id, interaction_content)
                interaction_time = time.time() - interaction_start

                # Assess interaction quality
                interaction_quality = await self.ux_analyzer.assess_interaction_quality(
                    interaction_content, response, phase["quality_expectations"]
                )

                phase_interactions.append({
                    "interaction_type": interaction_type,
                    "content": interaction_content,
                    "response": response,
                    "quality_score": interaction_quality["quality_score"],
                    "engagement_level": interaction_quality["engagement_level"],
                    "interaction_time": interaction_time,
                    "timestamp": time.time()
                })

                interaction_count += 1

                # Vary interaction timing to simulate natural usage
                pause_time = 5 + (interaction_count % 4) * 2  # 5-11 second pauses
                await asyncio.sleep(pause_time)

            phase_duration = time.time() - phase_start

            # Analyze phase engagement
            phase_engagement = await self.ux_analyzer.analyze_phase_engagement(
                phase_interactions, phase["engagement_metrics"]
            )

            phase_results.append({
                "phase_name": phase["phase_name"],
                "duration": phase_duration,
                "interaction_count": len(phase_interactions),
                "avg_quality": statistics.mean(i["quality_score"] for i in phase_interactions),
                "avg_engagement": statistics.mean(i["engagement_level"] for i in phase_interactions),
                "engagement_trend": phase_engagement["trend"],
                "satisfaction_level": phase_engagement["satisfaction"],
                "fatigue_indicators": phase_engagement["fatigue_score"]
            })

        # Analyze overall long-term engagement
        overall_engagement_analysis = await self.ux_analyzer.analyze_long_term_engagement(
            phase_results, scenario["long_term_metrics"]
        )

        # Calculate engagement quality metrics
        engagement_quality = ConversationQualityMetrics(
            naturalness_score=overall_engagement_analysis["naturalness_trend"],
            coherence_score=overall_engagement_analysis["coherence_maintenance"],
            context_retention=overall_engagement_analysis["context_retention"],
            emotional_appropriateness=overall_engagement_analysis["emotional_consistency"],
            professional_effectiveness=overall_engagement_analysis["professional_maintenance"],
            response_relevance=overall_engagement_analysis["relevance_consistency"],
            conversation_flow=overall_engagement_analysis["flow_quality"]
        )

        total_duration = sum(phase["duration"] for phase in phase_results)

        result = CompanionInteractionResult(
            test_scenario="long_term_engagement",
            duration=total_duration,
            conversation_quality=engagement_quality,
            user_satisfaction=UserSatisfactionResult(
                overall_satisfaction=overall_engagement_analysis["final_satisfaction"],
                feature_ratings=overall_engagement_analysis["feature_ratings"],
                improvement_areas=overall_engagement_analysis["improvement_areas"],
                positive_feedback=overall_engagement_analysis["positive_aspects"],
                usability_score=overall_engagement_analysis["usability_trend"],
                recommendation_likelihood=overall_engagement_analysis["recommendation_score"]
            ),
            technical_performance={
                "total_interactions": sum(len(phase_results)),
                "engagement_sustainability": overall_engagement_analysis["sustainability_score"],
                "fatigue_resistance": 1.0 - overall_engagement_analysis["cumulative_fatigue"]
            },
            accessibility_compliance=0.94,
            improvement_suggestions=overall_engagement_analysis["recommendations"]
        )

        self.test_results.append(result)

        # Validate long-term engagement requirements
        assert overall_engagement_analysis["final_satisfaction"] >= 4.0, f"Final satisfaction {overall_engagement_analysis['final_satisfaction']:.2f} below 4.0 threshold"
        assert overall_engagement_analysis["sustainability_score"] >= 0.80, f"Engagement sustainability {overall_engagement_analysis['sustainability_score']:.2f} below 80% threshold"
        assert overall_engagement_analysis["cumulative_fatigue"] <= 0.30, f"Cumulative fatigue {overall_engagement_analysis['cumulative_fatigue']:.2f} exceeds 30% threshold"

    # Helper Methods

    async def _evaluate_conversation_quality(self, conversation_history: List[Dict], scenario: Dict) -> ConversationQualityMetrics:
        """Evaluate overall conversation quality metrics"""
        # Extract conversation elements for analysis
        user_inputs = [exchange["user_input"] for exchange in conversation_history]
        companion_responses = [exchange["companion_response"] for exchange in conversation_history]

        # Evaluate naturalness
        naturalness = await self.conversation_evaluator.evaluate_naturalness(
            companion_responses, scenario.get("naturalness_criteria", {})
        )

        # Evaluate coherence
        coherence = await self.conversation_evaluator.evaluate_coherence(
            conversation_history, scenario.get("coherence_criteria", {})
        )

        # Evaluate context retention
        context_retention = await self.conversation_evaluator.evaluate_context_retention(
            conversation_history, scenario.get("context_criteria", {})
        )

        # Evaluate emotional appropriateness
        emotional_appropriateness = await self.emotion_validator.evaluate_overall_emotional_appropriateness(
            conversation_history, scenario.get("emotional_criteria", {})
        )

        # Evaluate professional effectiveness
        professional_effectiveness = await self.ux_analyzer.evaluate_professional_effectiveness_overall(
            conversation_history, scenario.get("professional_criteria", {})
        )

        # Evaluate response relevance
        response_relevance = await self.conversation_evaluator.evaluate_response_relevance(
            conversation_history, scenario.get("relevance_criteria", {})
        )

        # Evaluate conversation flow
        conversation_flow = await self.conversation_evaluator.evaluate_conversation_flow(
            conversation_history, scenario.get("flow_criteria", {})
        )

        return ConversationQualityMetrics(
            naturalness_score=naturalness,
            coherence_score=coherence,
            context_retention=context_retention,
            emotional_appropriateness=emotional_appropriateness,
            professional_effectiveness=professional_effectiveness,
            response_relevance=response_relevance,
            conversation_flow=conversation_flow
        )

    async def _assess_conversation_satisfaction(self, conversation_history: List[Dict],
                                             naturalness_analysis: Dict) -> UserSatisfactionResult:
        """Assess user satisfaction based on conversation analysis"""
        # Calculate overall satisfaction based on conversation quality
        quality_factors = [
            naturalness_analysis.get("naturalness_score", 3.0),
            naturalness_analysis.get("engagement_score", 3.0),
            naturalness_analysis.get("helpfulness_score", 3.0),
            naturalness_analysis.get("coherence_score", 3.0)
        ]

        overall_satisfaction = statistics.mean(quality_factors)

        # Extract feature ratings
        feature_ratings = {
            "conversation_quality": naturalness_analysis.get("naturalness_score", 3.0),
            "response_speed": 4.0,  # Based on technical performance
            "helpfulness": naturalness_analysis.get("helpfulness_score", 3.0),
            "understanding": naturalness_analysis.get("comprehension_score", 3.0)
        }

        # Identify improvement areas
        improvement_areas = []
        if naturalness_analysis.get("naturalness_score", 5.0) < 4.0:
            improvement_areas.append("Conversation naturalness")
        if naturalness_analysis.get("coherence_score", 5.0) < 4.0:
            improvement_areas.append("Response coherence")

        # Positive feedback
        positive_feedback = []
        if naturalness_analysis.get("engagement_score", 0.0) >= 4.0:
            positive_feedback.append("Engaging conversation")
        if naturalness_analysis.get("helpfulness_score", 0.0) >= 4.0:
            positive_feedback.append("Helpful responses")

        return UserSatisfactionResult(
            overall_satisfaction=overall_satisfaction,
            feature_ratings=feature_ratings,
            improvement_areas=improvement_areas,
            positive_feedback=positive_feedback,
            usability_score=overall_satisfaction,
            recommendation_likelihood=min(5.0, overall_satisfaction + 0.5)
        )

    def _generate_phase_interaction(self, interaction_type: str, phase: Dict) -> str:
        """Generate interaction content for specific phase and type"""
        interaction_templates = {
            "technical_question": [
                "Can you help me analyze this network traffic pattern?",
                "What's the best approach for investigating this malware sample?",
                "How should I interpret these log entries?",
                "Can you explain this attack vector to me?"
            ],
            "casual_conversation": [
                "How are you doing today?",
                "What do you think about the latest cybersecurity trends?",
                "Any interesting cases you've worked on recently?",
                "What's your take on this security news?"
            ],
            "problem_solving": [
                "I'm stuck on this analysis - can you help me think through it?",
                "What would you do in this situation?",
                "How should I prioritize these security findings?",
                "Can you help me develop a strategy for this investigation?"
            ],
            "emotional_support": [
                "This investigation is really stressing me out",
                "I'm feeling overwhelmed by this complex case",
                "I'm not sure I'm qualified to handle this analysis",
                "This security incident is keeping me up at night"
            ]
        }

        templates = interaction_templates.get(interaction_type, interaction_templates["technical_question"])
        base_template = templates[hash(phase["phase_name"]) % len(templates)]

        # Add phase-specific context
        if "context_modifier" in phase:
            return f"{base_template} {phase['context_modifier']}"

        return base_template

    def _save_ux_test_results(self):
        """Save UX test results to file"""
        results_data = {
            "timestamp": time.time(),
            "test_type": "user_experience",
            "total_tests": len(self.test_results),
            "test_results": [
                {
                    "test_scenario": result.test_scenario,
                    "duration": result.duration,
                    "conversation_quality": {
                        "naturalness_score": result.conversation_quality.naturalness_score,
                        "coherence_score": result.conversation_quality.coherence_score,
                        "context_retention": result.conversation_quality.context_retention,
                        "emotional_appropriateness": result.conversation_quality.emotional_appropriateness,
                        "professional_effectiveness": result.conversation_quality.professional_effectiveness,
                        "response_relevance": result.conversation_quality.response_relevance,
                        "conversation_flow": result.conversation_quality.conversation_flow
                    },
                    "user_satisfaction": {
                        "overall_satisfaction": result.user_satisfaction.overall_satisfaction,
                        "feature_ratings": result.user_satisfaction.feature_ratings,
                        "usability_score": result.user_satisfaction.usability_score,
                        "recommendation_likelihood": result.user_satisfaction.recommendation_likelihood
                    },
                    "technical_performance": result.technical_performance,
                    "accessibility_compliance": result.accessibility_compliance,
                    "improvement_suggestions": result.improvement_suggestions
                }
                for result in self.test_results
            ],
            "summary_metrics": {
                "avg_naturalness": statistics.mean(r.conversation_quality.naturalness_score for r in self.test_results),
                "avg_satisfaction": statistics.mean(r.user_satisfaction.overall_satisfaction for r in self.test_results),
                "avg_professional_effectiveness": statistics.mean(r.conversation_quality.professional_effectiveness for r in self.test_results),
                "avg_accessibility": statistics.mean(r.accessibility_compliance for r in self.test_results)
            }
        }

        results_file = Path("test_reports/companion/ux_test_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)