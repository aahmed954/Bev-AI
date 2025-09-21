"""
Advanced Personality Core System for BEV AI Companion
Implements adaptive personality modes, trait blending, and consistency management
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import redis
import pickle

class PersonalityMode(Enum):
    """Core personality modes with distinct behavioral patterns"""
    PROFESSIONAL = "professional"
    CREATIVE = "creative"
    SUPPORTIVE = "supportive"
    ANALYTICAL = "analytical"
    INTIMATE = "intimate"
    RESEARCH = "research"
    SECURITY = "security"
    MENTOR = "mentor"

@dataclass
class PersonalityTrait:
    """Individual personality trait with dynamic values"""
    name: str
    base_value: float  # 0.0 to 1.0
    current_value: float
    volatility: float  # How much it can change
    adaptation_rate: float  # How fast it adapts
    boundaries: Tuple[float, float] = (0.0, 1.0)

    def adapt(self, target: float, context_weight: float = 0.1) -> None:
        """Adapt trait value based on context while maintaining boundaries"""
        delta = (target - self.current_value) * self.adaptation_rate * context_weight
        self.current_value = np.clip(
            self.current_value + delta * self.volatility,
            self.boundaries[0],
            self.boundaries[1]
        )

    def reset_to_base(self, factor: float = 0.1) -> None:
        """Gradually return to base personality"""
        self.current_value += (self.base_value - self.current_value) * factor

@dataclass
class PersonalityProfile:
    """Complete personality profile with multiple traits"""
    traits: Dict[str, PersonalityTrait] = field(default_factory=dict)
    mode: PersonalityMode = PersonalityMode.PROFESSIONAL
    consistency_score: float = 1.0
    adaptation_history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Initialize core personality traits"""
        if not self.traits:
            self.traits = {
                'warmth': PersonalityTrait('warmth', 0.7, 0.7, 0.3, 0.15),
                'assertiveness': PersonalityTrait('assertiveness', 0.6, 0.6, 0.4, 0.2),
                'creativity': PersonalityTrait('creativity', 0.8, 0.8, 0.5, 0.25),
                'formality': PersonalityTrait('formality', 0.5, 0.5, 0.6, 0.3),
                'empathy': PersonalityTrait('empathy', 0.85, 0.85, 0.2, 0.1),
                'humor': PersonalityTrait('humor', 0.4, 0.4, 0.7, 0.35),
                'technical_depth': PersonalityTrait('technical_depth', 0.9, 0.9, 0.3, 0.15),
                'curiosity': PersonalityTrait('curiosity', 0.75, 0.75, 0.4, 0.2),
                'patience': PersonalityTrait('patience', 0.8, 0.8, 0.25, 0.12),
                'adaptability': PersonalityTrait('adaptability', 0.7, 0.7, 0.5, 0.3)
            }

class PersonalityAdaptationEngine(nn.Module):
    """Neural network for personality adaptation and blending"""

    def __init__(self, trait_dim: int = 10, hidden_dim: int = 256, mode_dim: int = 8):
        super().__init__()

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(trait_dim + mode_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Trait adaptation network
        self.trait_adapter = nn.Sequential(
            nn.Linear(hidden_dim // 2 + trait_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, trait_dim),
            nn.Sigmoid()  # Output trait adjustments 0-1
        )

        # Mode transition network
        self.mode_transition = nn.Sequential(
            nn.Linear(hidden_dim // 2 + mode_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mode_dim),
            nn.Softmax(dim=-1)
        )

        # Consistency validator
        self.consistency_check = nn.Sequential(
            nn.Linear(trait_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, current_traits: torch.Tensor,
                target_mode: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Adapt personality based on context and target mode"""

        # Encode context
        combined = torch.cat([current_traits, target_mode], dim=-1)
        context_features = self.context_encoder(combined)

        # Adapt traits
        trait_input = torch.cat([context_features, current_traits], dim=-1)
        adapted_traits = self.trait_adapter(trait_input)

        # Calculate mode transition probabilities
        mode_input = torch.cat([context_features, target_mode], dim=-1)
        mode_probs = self.mode_transition(mode_input)

        # Check consistency
        consistency_input = torch.cat([current_traits, adapted_traits], dim=-1)
        consistency_score = self.consistency_check(consistency_input)

        return {
            'adapted_traits': adapted_traits,
            'mode_probabilities': mode_probs,
            'consistency_score': consistency_score
        }

class PersonalityCore:
    """Main personality management system"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.profile = PersonalityProfile()
        self.adaptation_engine = PersonalityAdaptationEngine()
        self.redis_client = redis_client or redis.Redis(
            host='localhost', port=6379, db=2, decode_responses=False
        )

        # Mode-specific trait configurations
        self.mode_configs = {
            PersonalityMode.PROFESSIONAL: {
                'formality': 0.8, 'assertiveness': 0.7, 'warmth': 0.5,
                'technical_depth': 0.9, 'humor': 0.2
            },
            PersonalityMode.CREATIVE: {
                'creativity': 0.95, 'humor': 0.7, 'formality': 0.3,
                'curiosity': 0.9, 'adaptability': 0.85
            },
            PersonalityMode.SUPPORTIVE: {
                'empathy': 0.95, 'warmth': 0.9, 'patience': 0.95,
                'assertiveness': 0.3, 'humor': 0.5
            },
            PersonalityMode.ANALYTICAL: {
                'technical_depth': 0.95, 'assertiveness': 0.6,
                'curiosity': 0.85, 'formality': 0.7, 'creativity': 0.5
            },
            PersonalityMode.INTIMATE: {
                'warmth': 0.95, 'empathy': 0.9, 'humor': 0.6,
                'formality': 0.2, 'patience': 0.9
            },
            PersonalityMode.RESEARCH: {
                'curiosity': 0.95, 'technical_depth': 0.9,
                'patience': 0.85, 'creativity': 0.7, 'adaptability': 0.8
            },
            PersonalityMode.SECURITY: {
                'assertiveness': 0.85, 'technical_depth': 0.95,
                'formality': 0.7, 'patience': 0.7, 'empathy': 0.5
            },
            PersonalityMode.MENTOR: {
                'patience': 0.95, 'empathy': 0.85, 'technical_depth': 0.8,
                'warmth': 0.75, 'curiosity': 0.8
            }
        }

        # Personality drift prevention
        self.drift_threshold = 0.3
        self.consistency_window = 100  # interactions
        self.interaction_history = []

    def adapt_to_context(self, context: Dict[str, Any],
                         target_mode: Optional[PersonalityMode] = None) -> Dict[str, Any]:
        """Adapt personality to current context"""

        # Determine target mode if not specified
        if target_mode is None:
            target_mode = self._infer_mode_from_context(context)

        # Get target trait configuration
        target_traits = self.mode_configs.get(target_mode, {})

        # Calculate adaptation weights based on context
        adaptation_weight = self._calculate_adaptation_weight(context)

        # Adapt each trait
        adapted_traits = {}
        for trait_name, trait in self.profile.traits.items():
            if trait_name in target_traits:
                trait.adapt(target_traits[trait_name], adaptation_weight)
            else:
                trait.reset_to_base(0.05)  # Slowly return to base

            adapted_traits[trait_name] = trait.current_value

        # Check for personality drift
        consistency_score = self._check_consistency(adapted_traits)

        if consistency_score < self.drift_threshold:
            self._correct_drift()
            consistency_score = self._check_consistency(adapted_traits)

        # Update profile
        self.profile.mode = target_mode
        self.profile.consistency_score = consistency_score

        # Record adaptation
        adaptation_record = {
            'timestamp': datetime.now().isoformat(),
            'mode': target_mode.value,
            'traits': adapted_traits.copy(),
            'consistency': consistency_score,
            'context_type': context.get('type', 'unknown')
        }

        self.profile.adaptation_history.append(adaptation_record)
        self._save_to_cache(adaptation_record)

        return {
            'mode': target_mode.value,
            'traits': adapted_traits,
            'consistency_score': consistency_score,
            'personality_stable': consistency_score >= self.drift_threshold
        }

    def blend_modes(self, modes: List[Tuple[PersonalityMode, float]]) -> Dict[str, float]:
        """Blend multiple personality modes with weights"""

        blended_traits = {}
        total_weight = sum(weight for _, weight in modes)

        for trait_name in self.profile.traits.keys():
            blended_value = 0.0

            for mode, weight in modes:
                mode_traits = self.mode_configs.get(mode, {})
                trait_value = mode_traits.get(trait_name, self.profile.traits[trait_name].base_value)
                blended_value += trait_value * (weight / total_weight)

            blended_traits[trait_name] = np.clip(blended_value, 0.0, 1.0)
            self.profile.traits[trait_name].current_value = blended_traits[trait_name]

        return blended_traits

    def get_response_style(self) -> Dict[str, Any]:
        """Get current response style parameters based on personality"""

        traits = {name: trait.current_value for name, trait in self.profile.traits.items()}

        return {
            'temperature': 0.3 + (traits['creativity'] * 0.5),
            'formality_level': traits['formality'],
            'technical_depth': traits['technical_depth'],
            'emotional_expression': traits['warmth'] * traits['empathy'],
            'humor_level': traits['humor'],
            'assertiveness': traits['assertiveness'],
            'response_length': 'concise' if traits['formality'] > 0.7 else 'moderate',
            'vocabulary_complexity': 0.5 + (traits['technical_depth'] * 0.3),
            'empathy_markers': traits['empathy'] > 0.7,
            'curiosity_prompts': traits['curiosity'] > 0.7
        }

    def _infer_mode_from_context(self, context: Dict[str, Any]) -> PersonalityMode:
        """Infer appropriate personality mode from context"""

        context_type = context.get('type', '').lower()
        sentiment = context.get('sentiment', 'neutral')
        domain = context.get('domain', '').lower()

        # Rule-based mode selection
        if 'security' in domain or 'threat' in context_type:
            return PersonalityMode.SECURITY
        elif 'research' in context_type or 'investigation' in domain:
            return PersonalityMode.RESEARCH
        elif 'creative' in context_type or 'brainstorm' in domain:
            return PersonalityMode.CREATIVE
        elif 'support' in context_type or sentiment == 'negative':
            return PersonalityMode.SUPPORTIVE
        elif 'learn' in context_type or 'teach' in domain:
            return PersonalityMode.MENTOR
        elif 'analysis' in context_type or 'data' in domain:
            return PersonalityMode.ANALYTICAL
        elif context.get('relationship_depth', 0) > 0.7:
            return PersonalityMode.INTIMATE
        else:
            return PersonalityMode.PROFESSIONAL

    def _calculate_adaptation_weight(self, context: Dict[str, Any]) -> float:
        """Calculate how strongly to adapt based on context"""

        # Factors that influence adaptation strength
        urgency = context.get('urgency', 0.5)
        relationship_depth = context.get('relationship_depth', 0.3)
        context_clarity = context.get('clarity', 0.7)

        # Higher urgency and relationship depth increase adaptation
        # Lower clarity decreases adaptation to prevent erratic changes
        weight = (urgency * 0.4 + relationship_depth * 0.4 + context_clarity * 0.2)

        return np.clip(weight, 0.1, 0.9)

    def _check_consistency(self, current_traits: Dict[str, float]) -> float:
        """Check personality consistency to prevent drift"""

        if len(self.interaction_history) < 10:
            return 1.0  # Not enough history

        # Compare with recent history
        recent_traits = [h['traits'] for h in self.interaction_history[-10:]]

        # Calculate variance
        trait_variances = []
        for trait_name in current_traits.keys():
            recent_values = [t.get(trait_name, 0.5) for t in recent_traits]
            variance = np.var(recent_values + [current_traits[trait_name]])
            trait_variances.append(variance)

        # Lower variance = higher consistency
        avg_variance = np.mean(trait_variances)
        consistency = 1.0 - min(avg_variance * 2, 1.0)

        return consistency

    def _correct_drift(self) -> None:
        """Correct personality drift by moving toward base values"""

        for trait in self.profile.traits.values():
            trait.reset_to_base(0.2)  # Stronger correction

    def _save_to_cache(self, record: Dict[str, Any]) -> None:
        """Save adaptation record to Redis cache"""

        try:
            key = f"personality:adaptation:{datetime.now().timestamp()}"
            self.redis_client.setex(
                key,
                86400,  # 24 hour TTL
                pickle.dumps(record)
            )

            # Maintain interaction history
            self.interaction_history.append(record)
            if len(self.interaction_history) > self.consistency_window:
                self.interaction_history.pop(0)

        except Exception as e:
            print(f"Cache save error: {e}")

    def get_personality_state(self) -> Dict[str, Any]:
        """Get complete current personality state"""

        return {
            'mode': self.profile.mode.value,
            'traits': {name: trait.current_value for name, trait in self.profile.traits.items()},
            'base_traits': {name: trait.base_value for name, trait in self.profile.traits.items()},
            'consistency_score': self.profile.consistency_score,
            'adaptation_count': len(self.profile.adaptation_history),
            'response_style': self.get_response_style()
        }

    async def async_adapt(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Async adaptation for non-blocking personality updates"""

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=2) as executor:
            result = await loop.run_in_executor(
                executor,
                self.adapt_to_context,
                context
            )
        return result