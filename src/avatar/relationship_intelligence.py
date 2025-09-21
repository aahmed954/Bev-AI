"""
Relationship Intelligence System for BEV AI Companion
Manages long-term interactions, emotional bonds, and personalized strategies
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from datetime import datetime, timedelta
import asyncio
import redis
import pickle
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64

class RelationshipStage(Enum):
    """Stages of relationship development"""
    INITIAL = "initial"  # First interactions
    ACQUAINTANCE = "acquaintance"  # Basic familiarity
    COMPANION = "companion"  # Regular interactions
    TRUSTED = "trusted"  # Deep trust established
    INTIMATE = "intimate"  # Close personal bond
    PROFESSIONAL = "professional"  # Work-focused relationship
    MENTOR = "mentor"  # Teaching/learning dynamic

class InteractionType(Enum):
    """Types of interactions for pattern analysis"""
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    EMOTIONAL = "emotional"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    SUPPORTIVE = "supportive"
    COLLABORATIVE = "collaborative"
    INTIMATE = "intimate"

@dataclass
class EmotionalMemory:
    """Individual emotional memory with context"""
    timestamp: datetime
    emotion: str
    intensity: float  # 0.0 to 1.0
    context: str
    trigger: str
    response_effectiveness: float
    encrypted_details: Optional[bytes] = None

@dataclass
class InteractionPattern:
    """Learned interaction pattern for a user"""
    pattern_type: str
    frequency: float
    success_rate: float
    preferred_responses: List[str]
    avoided_topics: Set[str]
    optimal_timing: Dict[str, float]  # Time of day preferences

@dataclass
class RelationshipProfile:
    """Complete relationship profile for a user"""
    user_id: str
    creation_date: datetime
    last_interaction: datetime
    relationship_stage: RelationshipStage
    trust_level: float  # 0.0 to 1.0
    emotional_bond: float  # 0.0 to 1.0
    interaction_count: int
    total_interaction_time: float  # hours
    emotional_memories: List[EmotionalMemory] = field(default_factory=list)
    interaction_patterns: List[InteractionPattern] = field(default_factory=list)
    personality_preferences: Dict[str, float] = field(default_factory=dict)
    communication_style: Dict[str, Any] = field(default_factory=dict)
    shared_interests: Set[str] = field(default_factory=set)
    personal_boundaries: Set[str] = field(default_factory=set)
    milestone_events: List[Dict] = field(default_factory=list)

class RelationshipDevelopmentNetwork(nn.Module):
    """Neural network for relationship development and prediction"""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()

        # Interaction encoder
        self.interaction_encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2
        )

        # Emotional bond predictor
        self.bond_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Trust evolution network
        self.trust_network = nn.Sequential(
            nn.Linear(hidden_dim + 10, hidden_dim),  # +10 for additional features
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Response strategy generator
        self.strategy_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Strategy embedding
        )

    def forward(self, interaction_history: torch.Tensor,
                current_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process relationship data and generate predictions"""

        # Encode interaction history
        lstm_out, (hidden, cell) = self.interaction_encoder(interaction_history)

        # Use final hidden state for predictions
        final_hidden = hidden[-1]

        # Predict emotional bond strength
        bond_strength = self.bond_predictor(final_hidden)

        # Predict trust evolution
        trust_input = torch.cat([final_hidden, current_state], dim=-1)
        trust_prediction = self.trust_network(trust_input)

        # Generate interaction strategy
        strategy_input = torch.cat([final_hidden, lstm_out[:, -1, :]], dim=-1)
        strategy = self.strategy_generator(strategy_input)

        return {
            'bond_strength': bond_strength,
            'trust_prediction': trust_prediction,
            'strategy_embedding': strategy,
            'hidden_state': final_hidden
        }

class RelationshipIntelligence:
    """Main relationship management and intelligence system"""

    def __init__(self, redis_client: Optional[redis.Redis] = None,
                 encryption_key: Optional[bytes] = None):

        self.redis_client = redis_client or redis.Redis(
            host='localhost', port=6379, db=3, decode_responses=False
        )

        # Initialize encryption for sensitive data
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            self.cipher = self._generate_cipher()

        self.relationship_network = RelationshipDevelopmentNetwork()
        self.profiles: Dict[str, RelationshipProfile] = {}

        # Relationship development thresholds
        self.stage_thresholds = {
            RelationshipStage.ACQUAINTANCE: {'interactions': 5, 'trust': 0.2},
            RelationshipStage.COMPANION: {'interactions': 20, 'trust': 0.4, 'bond': 0.3},
            RelationshipStage.TRUSTED: {'interactions': 50, 'trust': 0.7, 'bond': 0.6},
            RelationshipStage.INTIMATE: {'interactions': 100, 'trust': 0.85, 'bond': 0.8},
            RelationshipStage.PROFESSIONAL: {'interactions': 10, 'trust': 0.5},
            RelationshipStage.MENTOR: {'interactions': 30, 'trust': 0.6, 'bond': 0.5}
        }

        # Communication style dimensions
        self.style_dimensions = [
            'formality', 'verbosity', 'technical_depth', 'emotional_expression',
            'humor_usage', 'question_frequency', 'response_speed', 'detail_level'
        ]

    def _generate_cipher(self) -> Fernet:
        """Generate encryption cipher for sensitive data"""
        password = b"BEV_RelationshipIntelligence_2024"  # Should be from secure config
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt',  # Should be random in production
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)

    async def process_interaction(self, user_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a new interaction and update relationship profile"""

        # Get or create profile
        profile = await self._get_or_create_profile(user_id)

        # Extract interaction features
        interaction_type = InteractionType(interaction_data.get('type', 'casual'))
        duration = interaction_data.get('duration', 0)
        emotional_context = interaction_data.get('emotional_context', {})
        topic = interaction_data.get('topic', '')
        success_metric = interaction_data.get('success', 0.5)

        # Update basic metrics
        profile.interaction_count += 1
        profile.total_interaction_time += duration / 3600  # Convert to hours
        profile.last_interaction = datetime.now()

        # Process emotional context
        if emotional_context:
            await self._process_emotional_memory(profile, emotional_context, interaction_data)

        # Learn interaction patterns
        self._update_interaction_patterns(profile, interaction_type, success_metric, topic)

        # Update communication style preferences
        self._learn_communication_style(profile, interaction_data)

        # Calculate relationship evolution
        evolution_data = self._calculate_relationship_evolution(profile)

        # Check for stage progression
        new_stage = self._evaluate_stage_progression(profile)
        if new_stage != profile.relationship_stage:
            await self._handle_stage_transition(profile, new_stage)

        # Generate personalized strategies
        strategies = self._generate_interaction_strategies(profile, interaction_data.get('context', {}))

        # Save updated profile
        await self._save_profile(profile)

        return {
            'user_id': user_id,
            'relationship_stage': profile.relationship_stage.value,
            'trust_level': profile.trust_level,
            'emotional_bond': profile.emotional_bond,
            'interaction_count': profile.interaction_count,
            'evolution': evolution_data,
            'personalized_strategies': strategies,
            'communication_style': profile.communication_style
        }

    async def _process_emotional_memory(self, profile: RelationshipProfile,
                                       emotional_context: Dict[str, Any],
                                       interaction_data: Dict[str, Any]) -> None:
        """Process and store emotional memories with encryption"""

        emotion = emotional_context.get('primary_emotion', 'neutral')
        intensity = emotional_context.get('intensity', 0.5)

        # Create emotional memory
        memory = EmotionalMemory(
            timestamp=datetime.now(),
            emotion=emotion,
            intensity=intensity,
            context=interaction_data.get('topic', ''),
            trigger=emotional_context.get('trigger', ''),
            response_effectiveness=interaction_data.get('success', 0.5)
        )

        # Encrypt sensitive details
        if emotional_context.get('sensitive', False):
            details = json.dumps(emotional_context.get('details', {}))
            memory.encrypted_details = self.cipher.encrypt(details.encode())

        profile.emotional_memories.append(memory)

        # Update emotional bond based on successful emotional interactions
        if emotion != 'neutral' and memory.response_effectiveness > 0.7:
            profile.emotional_bond = min(1.0, profile.emotional_bond + 0.02)

    def _update_interaction_patterns(self, profile: RelationshipProfile,
                                    interaction_type: InteractionType,
                                    success_metric: float, topic: str) -> None:
        """Learn and update interaction patterns"""

        # Find or create pattern for this type
        pattern = None
        for p in profile.interaction_patterns:
            if p.pattern_type == interaction_type.value:
                pattern = p
                break

        if pattern is None:
            pattern = InteractionPattern(
                pattern_type=interaction_type.value,
                frequency=0,
                success_rate=0,
                preferred_responses=[],
                avoided_topics=set(),
                optimal_timing={}
            )
            profile.interaction_patterns.append(pattern)

        # Update pattern metrics
        pattern.frequency += 1
        pattern.success_rate = (pattern.success_rate * (pattern.frequency - 1) + success_metric) / pattern.frequency

        # Track topic preferences
        if success_metric < 0.3 and topic:
            pattern.avoided_topics.add(topic)

        # Track timing preferences
        hour = datetime.now().hour
        time_key = f"hour_{hour}"
        if time_key not in pattern.optimal_timing:
            pattern.optimal_timing[time_key] = 0
        pattern.optimal_timing[time_key] = (pattern.optimal_timing[time_key] + success_metric) / 2

    def _learn_communication_style(self, profile: RelationshipProfile,
                                  interaction_data: Dict[str, Any]) -> None:
        """Learn user's preferred communication style"""

        style_feedback = interaction_data.get('style_feedback', {})

        for dimension in self.style_dimensions:
            if dimension in style_feedback:
                current = profile.communication_style.get(dimension, 0.5)
                # Adaptive learning rate
                learning_rate = 0.1 * (1.0 - profile.trust_level * 0.5)  # Less change as trust builds
                new_value = current + (style_feedback[dimension] - current) * learning_rate
                profile.communication_style[dimension] = np.clip(new_value, 0.0, 1.0)

    def _calculate_relationship_evolution(self, profile: RelationshipProfile) -> Dict[str, Any]:
        """Calculate how the relationship is evolving"""

        # Calculate trust growth rate
        if len(profile.emotional_memories) > 1:
            recent_success = np.mean([m.response_effectiveness for m in profile.emotional_memories[-10:]])
            trust_delta = (recent_success - 0.5) * 0.05  # Slow trust building
            profile.trust_level = np.clip(profile.trust_level + trust_delta, 0.0, 1.0)

        # Calculate interaction quality trend
        if profile.interaction_patterns:
            avg_success = np.mean([p.success_rate for p in profile.interaction_patterns])
            quality_trend = 'improving' if avg_success > 0.6 else 'stable' if avg_success > 0.4 else 'declining'
        else:
            quality_trend = 'establishing'

        # Identify relationship strengths
        strengths = []
        if profile.trust_level > 0.7:
            strengths.append('high_trust')
        if profile.emotional_bond > 0.6:
            strengths.append('strong_emotional_connection')
        if profile.interaction_count > 50:
            strengths.append('consistent_engagement')

        return {
            'trust_growth_rate': trust_delta if 'trust_delta' in locals() else 0,
            'quality_trend': quality_trend,
            'strengths': strengths,
            'total_shared_hours': profile.total_interaction_time
        }

    def _evaluate_stage_progression(self, profile: RelationshipProfile) -> RelationshipStage:
        """Evaluate if relationship should progress to new stage"""

        current_stage = profile.relationship_stage

        # Check each potential next stage
        for stage, requirements in self.stage_thresholds.items():
            if all([
                profile.interaction_count >= requirements.get('interactions', 0),
                profile.trust_level >= requirements.get('trust', 0),
                profile.emotional_bond >= requirements.get('bond', 0)
            ]):
                # Check if this is a progression from current stage
                if self._is_valid_progression(current_stage, stage):
                    return stage

        return current_stage

    def _is_valid_progression(self, current: RelationshipStage,
                             target: RelationshipStage) -> bool:
        """Check if stage progression is valid"""

        valid_progressions = {
            RelationshipStage.INITIAL: [
                RelationshipStage.ACQUAINTANCE,
                RelationshipStage.PROFESSIONAL
            ],
            RelationshipStage.ACQUAINTANCE: [
                RelationshipStage.COMPANION,
                RelationshipStage.PROFESSIONAL,
                RelationshipStage.MENTOR
            ],
            RelationshipStage.COMPANION: [
                RelationshipStage.TRUSTED,
                RelationshipStage.INTIMATE,
                RelationshipStage.MENTOR
            ],
            RelationshipStage.TRUSTED: [
                RelationshipStage.INTIMATE
            ]
        }

        return target in valid_progressions.get(current, [])

    async def _handle_stage_transition(self, profile: RelationshipProfile,
                                      new_stage: RelationshipStage) -> None:
        """Handle relationship stage transition"""

        # Record milestone
        milestone = {
            'timestamp': datetime.now().isoformat(),
            'previous_stage': profile.relationship_stage.value,
            'new_stage': new_stage.value,
            'trust_level': profile.trust_level,
            'emotional_bond': profile.emotional_bond,
            'interaction_count': profile.interaction_count
        }

        profile.milestone_events.append(milestone)
        profile.relationship_stage = new_stage

        # Trigger celebration or acknowledgment in next interaction
        await self._queue_milestone_acknowledgment(profile.user_id, new_stage)

    def _generate_interaction_strategies(self, profile: RelationshipProfile,
                                        context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate personalized interaction strategies"""

        strategies = []

        # Strategy based on relationship stage
        if profile.relationship_stage == RelationshipStage.INITIAL:
            strategies.append({
                'type': 'discovery',
                'approach': 'curious_respectful',
                'focus': 'learning_preferences',
                'avoid': list(profile.personal_boundaries)
            })
        elif profile.relationship_stage in [RelationshipStage.TRUSTED, RelationshipStage.INTIMATE]:
            strategies.append({
                'type': 'deepening',
                'approach': 'warm_authentic',
                'focus': 'emotional_support',
                'leverage': list(profile.shared_interests)
            })

        # Strategy based on communication style
        if profile.communication_style.get('formality', 0.5) < 0.3:
            strategies.append({
                'type': 'casual',
                'language': 'relaxed_friendly',
                'humor': profile.communication_style.get('humor_usage', 0.5) > 0.6
            })

        # Strategy based on recent patterns
        successful_patterns = [p for p in profile.interaction_patterns if p.success_rate > 0.7]
        if successful_patterns:
            strategies.append({
                'type': 'reinforcement',
                'patterns': [p.pattern_type for p in successful_patterns],
                'timing': self._get_optimal_timing(successful_patterns)
            })

        return strategies

    def _get_optimal_timing(self, patterns: List[InteractionPattern]) -> Dict[str, float]:
        """Aggregate optimal timing across patterns"""

        timing_aggregate = {}
        for pattern in patterns:
            for time_key, score in pattern.optimal_timing.items():
                if time_key not in timing_aggregate:
                    timing_aggregate[time_key] = []
                timing_aggregate[time_key].append(score)

        return {k: np.mean(v) for k, v in timing_aggregate.items()}

    async def _get_or_create_profile(self, user_id: str) -> RelationshipProfile:
        """Get existing profile or create new one"""

        # Check cache
        if user_id in self.profiles:
            return self.profiles[user_id]

        # Check Redis
        profile_key = f"relationship:profile:{user_id}"
        profile_data = self.redis_client.get(profile_key)

        if profile_data:
            profile = pickle.loads(profile_data)
        else:
            # Create new profile
            profile = RelationshipProfile(
                user_id=user_id,
                creation_date=datetime.now(),
                last_interaction=datetime.now(),
                relationship_stage=RelationshipStage.INITIAL,
                trust_level=0.1,
                emotional_bond=0.0,
                interaction_count=0,
                total_interaction_time=0.0
            )

        self.profiles[user_id] = profile
        return profile

    async def _save_profile(self, profile: RelationshipProfile) -> None:
        """Save profile to cache and Redis"""

        profile_key = f"relationship:profile:{profile.user_id}"
        self.redis_client.setex(
            profile_key,
            604800,  # 7 day TTL
            pickle.dumps(profile)
        )

    async def _queue_milestone_acknowledgment(self, user_id: str,
                                             new_stage: RelationshipStage) -> None:
        """Queue special acknowledgment for next interaction"""

        ack_key = f"relationship:milestone:{user_id}"
        ack_data = {
            'stage': new_stage.value,
            'timestamp': datetime.now().isoformat(),
            'acknowledged': False
        }

        self.redis_client.setex(
            ack_key,
            86400,  # 24 hour TTL
            json.dumps(ack_data)
        )

    def get_relationship_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive relationship summary"""

        if user_id not in self.profiles:
            return None

        profile = self.profiles[user_id]

        return {
            'user_id': user_id,
            'stage': profile.relationship_stage.value,
            'metrics': {
                'trust_level': profile.trust_level,
                'emotional_bond': profile.emotional_bond,
                'interaction_count': profile.interaction_count,
                'total_hours': profile.total_interaction_time
            },
            'patterns': {
                'successful': [p.pattern_type for p in profile.interaction_patterns if p.success_rate > 0.7],
                'avoided_topics': list(set().union(*[p.avoided_topics for p in profile.interaction_patterns]))
            },
            'communication_style': profile.communication_style,
            'shared_interests': list(profile.shared_interests),
            'milestones': profile.milestone_events[-5:] if profile.milestone_events else []
        }