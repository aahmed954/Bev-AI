"""
Integration Layer for Personality and Relationship Systems with BEV OSINT Platform
Connects advanced AI companion features with existing security research capabilities
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import redis
from datetime import datetime
import torch

# Import personality and relationship components
from .personality_core import PersonalityCore, PersonalityMode
from .relationship_intelligence import RelationshipIntelligence, InteractionType
from .professional_roles import ProfessionalRoleSystem, ProfessionalRole, RoleContext
from .memory_privacy_architecture import MemoryStorage, MemoryType, PrivacyLevel

# Import existing BEV components
from ..mcp_server.server import MCPServer
from ..mcp_server.tools import OSINTToolRegistry

@dataclass
class CompanionContext:
    """Unified context for AI companion operations"""
    user_id: str
    session_id: str
    osint_active: bool
    personality_mode: PersonalityMode
    professional_role: Optional[ProfessionalRole]
    security_clearance: str
    current_task: Optional[Dict[str, Any]]
    interaction_history: List[Dict[str, Any]]

class BEVCompanionIntegration:
    """Main integration class for AI companion with OSINT platform"""

    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        # Initialize Redis connection
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=6)

        # Initialize personality and relationship systems
        self.personality_core = PersonalityCore(self.redis_client)
        self.relationship_system = RelationshipIntelligence(self.redis_client)
        self.role_system = ProfessionalRoleSystem(self.redis_client)
        self.memory_storage = MemoryStorage(self.redis_client)

        # Integration with OSINT tools
        self.osint_registry = OSINTToolRegistry()
        self.active_contexts: Dict[str, CompanionContext] = {}

        # Performance tracking
        self.metrics = {
            'personality_adaptations': 0,
            'role_transitions': 0,
            'osint_operations': 0,
            'memory_operations': 0
        }

    async def initialize_session(self, user_id: str,
                                initial_mode: Optional[PersonalityMode] = None) -> str:
        """Initialize a new companion session"""

        session_id = f"session_{user_id}_{datetime.now().timestamp()}"

        # Create companion context
        context = CompanionContext(
            user_id=user_id,
            session_id=session_id,
            osint_active=False,
            personality_mode=initial_mode or PersonalityMode.PROFESSIONAL,
            professional_role=None,
            security_clearance='standard',
            current_task=None,
            interaction_history=[]
        )

        self.active_contexts[session_id] = context

        # Load user relationship profile
        relationship_data = await self.relationship_system.process_interaction(
            user_id,
            {
                'type': 'session_start',
                'duration': 0,
                'topic': 'initialization'
            }
        )

        # Adapt personality based on relationship
        personality_context = {
            'type': 'session_initialization',
            'relationship_depth': relationship_data['emotional_bond'],
            'trust_level': relationship_data['trust_level'],
            'domain': 'general'
        }

        personality_state = self.personality_core.adapt_to_context(
            personality_context,
            initial_mode
        )

        # Store session initialization
        await self.memory_storage.store_memory(
            user_id,
            f"Session initialized with mode: {personality_state['mode']}",
            MemoryType.BEHAVIORAL,
            PrivacyLevel.PRIVATE,
            {'session_id': session_id}
        )

        return session_id

    async def process_user_input(self, session_id: str,
                                user_input: str,
                                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process user input with personality and role awareness"""

        if session_id not in self.active_contexts:
            return {'error': 'Invalid session'}

        context = self.active_contexts[session_id]

        # Analyze input for context
        input_analysis = await self._analyze_input(user_input, metadata)

        # Check if OSINT operation is needed
        if input_analysis['requires_osint']:
            return await self._handle_osint_request(context, user_input, input_analysis)

        # Check if role transition is needed
        if input_analysis['suggested_role'] and input_analysis['suggested_role'] != context.professional_role:
            await self._transition_role(context, input_analysis['suggested_role'])

        # Process through personality system
        personality_response = await self._generate_personality_response(
            context, user_input, input_analysis
        )

        # Update relationship system
        await self._update_relationship(context, user_input, personality_response)

        # Store interaction memory
        await self._store_interaction_memory(context, user_input, personality_response)

        return personality_response

    async def _analyze_input(self, user_input: str,
                            metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user input for context and intent"""

        analysis = {
            'requires_osint': False,
            'suggested_role': None,
            'emotion_detected': 'neutral',
            'urgency': 0.5,
            'complexity': 0.5,
            'domain': 'general'
        }

        # Check for OSINT keywords
        osint_keywords = ['investigate', 'search', 'find', 'analyze', 'threat', 'vulnerability', 'breach']
        if any(keyword in user_input.lower() for keyword in osint_keywords):
            analysis['requires_osint'] = True
            analysis['suggested_role'] = ProfessionalRole.SECURITY_ANALYST

        # Check for research keywords
        research_keywords = ['research', 'study', 'literature', 'data', 'hypothesis', 'analysis']
        if any(keyword in user_input.lower() for keyword in research_keywords):
            analysis['suggested_role'] = ProfessionalRole.RESEARCH_SPECIALIST

        # Detect emotional context (simplified)
        emotional_keywords = {
            'happy': ['happy', 'joy', 'excited', 'great'],
            'sad': ['sad', 'upset', 'disappointed'],
            'anxious': ['worried', 'anxious', 'concerned'],
            'angry': ['angry', 'frustrated', 'annoyed']
        }

        for emotion, keywords in emotional_keywords.items():
            if any(keyword in user_input.lower() for keyword in keywords):
                analysis['emotion_detected'] = emotion
                break

        # Assess complexity and urgency from metadata
        if metadata:
            analysis['urgency'] = metadata.get('urgency', 0.5)
            analysis['complexity'] = metadata.get('complexity', 0.5)
            analysis['domain'] = metadata.get('domain', 'general')

        return analysis

    async def _handle_osint_request(self, context: CompanionContext,
                                   user_input: str,
                                   analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle OSINT-related requests with personality awareness"""

        # Activate security analyst role if not already active
        if context.professional_role != ProfessionalRole.SECURITY_ANALYST:
            role_context = RoleContext(
                task_type='osint_investigation',
                urgency=analysis['urgency'],
                complexity=analysis['complexity'],
                domain='security',
                available_tools=list(self.osint_registry.tools.keys()),
                constraints={},
                objectives=['investigate', 'analyze', 'report'],
                stakeholders=[context.user_id]
            )

            await self.role_system.activate_role(
                ProfessionalRole.SECURITY_ANALYST,
                role_context
            )
            context.professional_role = ProfessionalRole.SECURITY_ANALYST

        # Execute OSINT task
        osint_task = {
            'type': 'osint_analysis',
            'query': user_input,
            'user_context': {
                'trust_level': context.security_clearance,
                'personality_mode': context.personality_mode.value
            }
        }

        osint_result = await self.role_system.execute_in_role(osint_task)

        # Adapt response style based on personality
        response_style = self.personality_core.get_response_style()

        # Format response with personality
        formatted_response = self._format_osint_response(
            osint_result['result'],
            response_style,
            context.personality_mode
        )

        self.metrics['osint_operations'] += 1

        return {
            'type': 'osint_response',
            'content': formatted_response,
            'raw_data': osint_result['result'],
            'personality_mode': context.personality_mode.value,
            'professional_role': context.professional_role.value
        }

    async def _transition_role(self, context: CompanionContext,
                              new_role: ProfessionalRole) -> None:
        """Transition to a new professional role"""

        role_context = RoleContext(
            task_type='dynamic_transition',
            urgency=0.5,
            complexity=0.5,
            domain='adaptive',
            available_tools=list(self.osint_registry.tools.keys()),
            constraints={},
            objectives=[],
            stakeholders=[context.user_id]
        )

        await self.role_system.seamless_transition(new_role, role_context)
        context.professional_role = new_role
        self.metrics['role_transitions'] += 1

    async def _generate_personality_response(self, context: CompanionContext,
                                           user_input: str,
                                           analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response with personality adaptation"""

        # Adapt personality to current context
        personality_context = {
            'type': analysis['domain'],
            'emotion': analysis['emotion_detected'],
            'urgency': analysis['urgency'],
            'relationship_depth': 0.5  # Get from relationship system
        }

        personality_state = self.personality_core.adapt_to_context(
            personality_context,
            context.personality_mode
        )

        # Get response style
        response_style = self.personality_core.get_response_style()

        # Generate response content (simplified for demonstration)
        response_content = self._generate_content(
            user_input,
            response_style,
            personality_state
        )

        self.metrics['personality_adaptations'] += 1

        return {
            'type': 'personality_response',
            'content': response_content,
            'personality_state': personality_state,
            'response_style': response_style,
            'emotion_expressed': analysis['emotion_detected']
        }

    async def _update_relationship(self, context: CompanionContext,
                                  user_input: str,
                                  response: Dict[str, Any]) -> None:
        """Update relationship system with interaction"""

        interaction_data = {
            'type': InteractionType.PROFESSIONAL.value if context.osint_active else InteractionType.CASUAL.value,
            'duration': 1.0,  # Placeholder
            'emotional_context': {
                'primary_emotion': response.get('emotion_expressed', 'neutral'),
                'intensity': 0.5
            },
            'topic': user_input[:50],  # First 50 chars as topic
            'success': 0.8,  # Placeholder success metric
            'style_feedback': {
                'formality': 0.5,
                'technical_depth': 0.7 if context.osint_active else 0.3
            }
        }

        await self.relationship_system.process_interaction(
            context.user_id,
            interaction_data
        )

    async def _store_interaction_memory(self, context: CompanionContext,
                                       user_input: str,
                                       response: Dict[str, Any]) -> None:
        """Store interaction in secure memory"""

        # Determine privacy level based on content
        privacy_level = PrivacyLevel.PRIVATE
        if context.osint_active:
            privacy_level = PrivacyLevel.PROFESSIONAL
        elif 'personal' in user_input.lower() or 'private' in user_input.lower():
            privacy_level = PrivacyLevel.SENSITIVE

        memory_content = {
            'user_input': user_input,
            'response_summary': response.get('content', '')[:200],
            'timestamp': datetime.now().isoformat(),
            'personality_mode': context.personality_mode.value,
            'professional_role': context.professional_role.value if context.professional_role else None
        }

        await self.memory_storage.store_memory(
            context.user_id,
            json.dumps(memory_content),
            MemoryType.RELATIONAL,
            privacy_level,
            {'session_id': context.session_id}
        )

        self.metrics['memory_operations'] += 1

    def _format_osint_response(self, osint_data: Dict[str, Any],
                              response_style: Dict[str, Any],
                              personality_mode: PersonalityMode) -> str:
        """Format OSINT results with personality-aware styling"""

        # Base formatting
        if personality_mode == PersonalityMode.PROFESSIONAL:
            intro = "Based on my analysis, "
            tone = "formal"
        elif personality_mode == PersonalityMode.SUPPORTIVE:
            intro = "I've carefully looked into this for you, and "
            tone = "warm"
        else:
            intro = "Here's what I found: "
            tone = "neutral"

        # Apply response style
        if response_style['technical_depth'] > 0.7:
            # Include technical details
            content = f"{intro}the technical assessment indicates {osint_data}"
        else:
            # Simplified version
            content = f"{intro}the findings suggest {self._simplify_osint(osint_data)}"

        # Add emotional markers if appropriate
        if response_style['empathy_markers']:
            content += " I hope this information helps you."

        return content

    def _simplify_osint(self, osint_data: Dict[str, Any]) -> str:
        """Simplify OSINT data for non-technical presentation"""
        # Simplified version of complex data
        if 'threat_level' in osint_data:
            return f"a {osint_data['threat_level']} threat level"
        elif 'findings' in osint_data:
            return f"{len(osint_data['findings'])} relevant findings"
        else:
            return "relevant security information"

    def _generate_content(self, user_input: str,
                         response_style: Dict[str, Any],
                         personality_state: Dict[str, Any]) -> str:
        """Generate response content based on personality"""

        # This is a simplified content generator
        # In production, this would interface with language models

        traits = personality_state['traits']

        # Build response based on traits
        if traits['warmth'] > 0.7:
            response = f"I understand your question about '{user_input[:30]}...'. "
        else:
            response = f"Regarding your query about '{user_input[:30]}...'. "

        if traits['technical_depth'] > 0.7:
            response += "From a technical perspective, "
        else:
            response += "Simply put, "

        # Add personality flavor
        if traits['humor'] > 0.6:
            response += "here's the interesting part - "
        elif traits['formality'] > 0.7:
            response += "the analysis indicates that "
        else:
            response += "what I can tell you is "

        # Placeholder conclusion
        response += "this requires further investigation based on the available data."

        return response

    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a companion session and save state"""

        if session_id not in self.active_contexts:
            return {'error': 'Invalid session'}

        context = self.active_contexts[session_id]

        # Save session summary
        session_summary = {
            'session_id': session_id,
            'user_id': context.user_id,
            'duration': datetime.now().isoformat(),
            'interactions': len(context.interaction_history),
            'personality_modes_used': [context.personality_mode.value],
            'professional_roles_used': [context.professional_role.value] if context.professional_role else [],
            'metrics': self.metrics.copy()
        }

        # Store session memory
        await self.memory_storage.store_memory(
            context.user_id,
            json.dumps(session_summary),
            MemoryType.BEHAVIORAL,
            PrivacyLevel.PRIVATE,
            {'session_end': True}
        )

        # Clean up
        del self.active_contexts[session_id]

        return {
            'status': 'session_ended',
            'summary': session_summary
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""

        return {
            'active_sessions': len(self.active_contexts),
            'total_metrics': self.metrics,
            'personality_state': self.personality_core.get_personality_state(),
            'role_summary': self.role_system.get_role_summary(),
            'memory_usage': {
                'redis': self.redis_client.info('memory')['used_memory_human'],
                'active_contexts': len(self.active_contexts)
            }
        }