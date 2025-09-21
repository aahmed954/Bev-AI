#!/usr/bin/env python3
"""
Advanced 3D Avatar Controller for BEV-AI OSINT Framework
Replaces Live2D with Gaussian Splatting + MetaHuman + Advanced AI
Optimized for RTX 4090 with real-time performance
"""

import asyncio
import logging
import json
import time
import base64
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

# Core ML/AI imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoProcessor,
    pipeline, Pipeline
)
import cv2
from PIL import Image
import soundfile as sf
import librosa

# 3D Rendering and Gaussian Splatting
try:
    import gsplat
    from gsplat import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    logging.warning("gsplat not available - falling back to basic rendering")

# Advanced TTS (Bark AI)
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from bark.generation import SUPPORTED_LANGS
    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    logging.warning("Bark AI not available - falling back to basic TTS")

# MetaHuman integration
try:
    import metahuman_sdk
    METAHUMAN_AVAILABLE = True
except ImportError:
    METAHUMAN_AVAILABLE = False
    logging.warning("MetaHuman SDK not available")

# FastAPI and WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import redis.asyncio as redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionState(Enum):
    """Advanced emotion states for OSINT research"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    FOCUSED = "focused"
    THINKING = "thinking"
    ANALYZING = "analyzing"
    ALERT = "alert"
    CONFIDENT = "confident"
    CURIOUS = "curious"
    PROCESSING = "processing"
    DISCOVERED = "discovered"
    CONCERNED = "concerned"
    SATISFIED = "satisfied"
    INVESTIGATING = "investigating"
    BREAKTHROUGH = "breakthrough"

class OSINTActivity(Enum):
    """OSINT research activity types"""
    BREACH_ANALYSIS = "breach_analysis"
    DARKNET_MONITORING = "darknet_monitoring"
    CRYPTO_TRACKING = "crypto_tracking"
    SOCIAL_INVESTIGATION = "social_investigation"
    THREAT_ANALYSIS = "threat_analysis"
    GRAPH_EXPLORATION = "graph_exploration"
    DATA_CORRELATION = "data_correlation"
    VULNERABILITY_RESEARCH = "vulnerability_research"

@dataclass
class AdvancedAvatarState:
    """Complete avatar state with OSINT context"""
    # Basic state
    emotion: EmotionState = EmotionState.NEUTRAL
    activity: OSINTActivity = OSINTActivity.THREAT_ANALYSIS
    expression_intensity: float = 1.0

    # 3D positioning and animation
    head_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    eye_position: Tuple[float, float] = (0.0, 0.0)
    mouth_shape: float = 0.0
    breathing_phase: float = 0.0

    # Advanced emotional state
    personality_traits: Dict[str, float] = field(default_factory=lambda: {
        'professionalism': 0.8,
        'curiosity': 0.9,
        'empathy': 0.7,
        'analytical_focus': 0.95,
        'enthusiasm': 0.75
    })

    # OSINT research context
    current_investigation: Optional[str] = None
    investigation_progress: float = 0.0
    findings_count: int = 0
    threat_level: float = 0.0
    last_discovery: Optional[str] = None

    # Performance tracking
    rendering_fps: float = 120.0
    voice_latency: float = 150.0  # milliseconds
    emotion_accuracy: float = 0.95

@dataclass
class EmotionVector:
    """Advanced emotion analysis vector"""
    # Primary emotions
    joy: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0

    # Research-specific emotions
    curiosity: float = 0.0
    focus: float = 0.0
    excitement: float = 0.0
    concern: float = 0.0
    satisfaction: float = 0.0
    breakthrough: float = 0.0

    # Context factors
    confidence: float = 0.5
    engagement: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)

class AdvancedEmotionEngine(nn.Module):
    """Transformer-based emotion recognition and synthesis"""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_emotions: int = 12):
        super().__init__()

        # Multimodal transformer for emotion understanding
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=4
        )

        # Audio emotion analysis
        self.audio_encoder = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.audio_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=2
        )

        # Visual emotion analysis
        self.visual_encoder = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.visual_pooling = nn.AdaptiveAvgPool2d((8, 8))
        self.visual_fc = nn.Linear(64 * 8 * 8, hidden_dim)

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )

        # Emotion prediction head
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_emotions),
            nn.Softmax(dim=-1)
        )

        # OSINT context integration
        self.osint_context_encoder = nn.Linear(64, hidden_dim)  # OSINT features

    def forward(self, text_features, audio_features, visual_features, osint_context=None):
        """Forward pass with multimodal emotion recognition"""

        # Process each modality
        text_encoded = self.text_encoder(text_features)

        audio_encoded = self.audio_encoder(audio_features)
        audio_encoded = self.audio_transformer(audio_encoded.transpose(1, 2))

        visual_encoded = self.visual_encoder(visual_features)
        visual_encoded = self.visual_pooling(visual_encoded)
        visual_encoded = self.visual_fc(visual_encoded.flatten(1))

        # Cross-modal attention
        combined_features = torch.stack([
            text_encoded.mean(dim=1),
            audio_encoded.mean(dim=1),
            visual_encoded
        ], dim=1)

        attended_features, _ = self.cross_attention(
            combined_features, combined_features, combined_features
        )

        # Include OSINT context if available
        if osint_context is not None:
            osint_encoded = self.osint_context_encoder(osint_context)
            attended_features = torch.cat([
                attended_features.flatten(1),
                osint_encoded
            ], dim=-1)

            # Adjust emotion head input size for OSINT context
            emotion_input = attended_features
        else:
            emotion_input = attended_features.flatten(1)

        # Predict emotions
        emotions = self.emotion_head(emotion_input)

        return emotions

class GaussianSplattingRenderer:
    """3D Gaussian Splatting renderer for photorealistic avatar"""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.model_path = model_path
        self.gaussian_model = None
        self.initialized = False

        # Rendering parameters optimized for RTX 4090
        self.render_config = {
            'image_height': 1080,
            'image_width': 1920,
            'fov_x': 1.2,
            'fov_y': 0.9,
            'near_plane': 0.01,
            'far_plane': 100.0,
            'sh_degree': 3,  # Spherical harmonics degree
            'densification_interval': 100,
            'opacity_reset_interval': 3000,
            'densify_grad_threshold': 0.0002
        }

    async def initialize(self):
        """Initialize Gaussian Splatting model"""
        if not GSPLAT_AVAILABLE:
            logger.error("gsplat not available - cannot initialize 3D renderer")
            return False

        try:
            # Load pre-trained Gaussian model
            self.gaussian_model = self._load_gaussian_model()

            # Optimize for RTX 4090
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_per_process_memory_fraction(0.6, device=self.device)

            self.initialized = True
            logger.info("Gaussian Splatting renderer initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Gaussian renderer: {e}")
            return False

    def _load_gaussian_model(self):
        """Load Gaussian Splatting model for avatar"""
        # This would load your MetaHuman-generated Gaussian model
        # For now, create a placeholder structure
        return {
            'gaussians': torch.randn(100000, 3, device=self.device),  # 100k Gaussians
            'features_dc': torch.randn(100000, 1, 3, device=self.device),
            'features_rest': torch.randn(100000, 15, 3, device=self.device),
            'opacities': torch.randn(100000, 1, device=self.device),
            'scales': torch.randn(100000, 3, device=self.device),
            'rotations': torch.randn(100000, 4, device=self.device)
        }

    async def render_frame(self, camera_params: Dict, emotion_state: EmotionState) -> np.ndarray:
        """Render single frame with emotion-based modifications"""
        if not self.initialized:
            logger.warning("Renderer not initialized")
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

        try:
            # Modify Gaussians based on emotion
            modified_gaussians = self._apply_emotion_deformation(emotion_state)

            # Render using gsplat
            rendered_image = self._gaussian_render(modified_gaussians, camera_params)

            # Convert to numpy for further processing
            image_array = rendered_image.cpu().numpy()
            image_array = (image_array * 255).astype(np.uint8)

            return image_array

        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def _apply_emotion_deformation(self, emotion_state: EmotionState) -> Dict:
        """Apply emotion-based deformations to Gaussian model"""
        # Emotion-specific deformation patterns
        emotion_deformations = {
            EmotionState.HAPPY: {'mouth_lift': 0.3, 'eye_crinkle': 0.2},
            EmotionState.FOCUSED: {'brow_furrow': 0.4, 'eye_narrow': 0.1},
            EmotionState.EXCITED: {'mouth_open': 0.5, 'eye_wide': 0.3},
            EmotionState.ALERT: {'head_forward': 0.2, 'eye_wide': 0.4},
            EmotionState.THINKING: {'head_tilt': 0.15, 'brow_furrow': 0.2},
            EmotionState.BREAKTHROUGH: {'eye_wide': 0.6, 'mouth_smile': 0.8}
        }

        deformation = emotion_deformations.get(emotion_state, {})

        # Apply deformations to Gaussian parameters
        modified_model = self.gaussian_model.copy()

        # Implement actual deformation logic here
        # This would modify facial feature Gaussians based on emotion

        return modified_model

    def _gaussian_render(self, gaussians: Dict, camera_params: Dict) -> torch.Tensor:
        """Render Gaussians using gsplat"""
        if not GSPLAT_AVAILABLE:
            # Fallback to simple rendering
            return torch.zeros((3, 1080, 1920), device=self.device)

        # Extract camera parameters
        viewmat = torch.tensor(camera_params.get('viewmat', np.eye(4)), device=self.device)
        projmat = torch.tensor(camera_params.get('projmat', np.eye(4)), device=self.device)

        # Render using gsplat
        rendered = rasterization(
            means=gaussians['gaussians'],
            quats=gaussians['rotations'],
            scales=gaussians['scales'],
            opacities=gaussians['opacities'],
            colors=gaussians['features_dc'].squeeze(1),
            viewmats=viewmat.unsqueeze(0),
            Ks=projmat[:3, :3].unsqueeze(0),
            width=self.render_config['image_width'],
            height=self.render_config['image_height']
        )

        return rendered.squeeze(0)

class AdvancedTTSEngine:
    """Advanced text-to-speech with Bark AI and emotion modulation"""

    def __init__(self, voice_profile: str = "professional_analyst"):
        self.voice_profile = voice_profile
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.initialized = False

        # Voice profiles for different OSINT contexts
        self.voice_profiles = {
            'professional_analyst': 'v2/en_speaker_6',  # Professional female voice
            'excited_researcher': 'v2/en_speaker_9',   # Enthusiastic tone
            'focused_investigator': 'v2/en_speaker_3', # Serious, focused
            'friendly_assistant': 'v2/en_speaker_7'    # Warm, helpful
        }

        # Emotion-based voice modulation
        self.emotion_modulation = {
            EmotionState.EXCITED: {'temperature': 0.9, 'top_k': 50},
            EmotionState.FOCUSED: {'temperature': 0.6, 'top_k': 30},
            EmotionState.ALERT: {'temperature': 0.7, 'top_k': 40},
            EmotionState.BREAKTHROUGH: {'temperature': 1.0, 'top_k': 60},
            EmotionState.THINKING: {'temperature': 0.5, 'top_k': 25}
        }

    async def initialize(self):
        """Initialize Bark TTS system"""
        if not BARK_AVAILABLE:
            logger.error("Bark AI not available - TTS disabled")
            return False

        try:
            # Preload models on GPU
            preload_models(device=self.device)
            self.initialized = True
            logger.info("Advanced TTS engine initialized with Bark AI")
            return True

        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            return False

    async def synthesize_speech(self, text: str, emotion: EmotionState,
                              osint_context: Optional[str] = None) -> bytes:
        """Generate speech with emotion and OSINT context awareness"""

        if not self.initialized:
            logger.warning("TTS not initialized - returning empty audio")
            return b''

        try:
            # Adapt voice profile based on OSINT context
            if osint_context and "breakthrough" in osint_context.lower():
                voice_profile = self.voice_profiles['excited_researcher']
            elif osint_context and "threat" in osint_context.lower():
                voice_profile = self.voice_profiles['focused_investigator']
            else:
                voice_profile = self.voice_profiles[self.voice_profile]

            # Get emotion modulation parameters
            modulation = self.emotion_modulation.get(emotion, {'temperature': 0.7, 'top_k': 40})

            # Generate audio with Bark
            audio_array = generate_audio(
                text,
                history_prompt=voice_profile,
                text_temp=modulation['temperature'],
                waveform_temp=modulation['temperature']
            )

            # Convert to bytes for transmission
            audio_bytes = self._audio_to_bytes(audio_array, SAMPLE_RATE)

            return audio_bytes

        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return b''

    def _audio_to_bytes(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio array to bytes"""
        # Normalize audio
        audio_normalized = audio_array / np.max(np.abs(audio_array))

        # Convert to 16-bit PCM
        audio_int16 = (audio_normalized * 32767).astype(np.int16)

        # Use soundfile to create WAV bytes
        import io
        buffer = io.BytesIO()
        sf.write(buffer, audio_int16, sample_rate, format='WAV')
        return buffer.getvalue()

class OSINTContextAnalyzer:
    """Analyzes OSINT research context for avatar responses"""

    def __init__(self):
        self.context_patterns = {
            'breach_discovery': {
                'keywords': ['breach', 'leak', 'exposed', 'compromised'],
                'emotion': EmotionState.ALERT,
                'response_style': 'urgent_professional'
            },
            'successful_correlation': {
                'keywords': ['connected', 'linked', 'correlation', 'pattern'],
                'emotion': EmotionState.BREAKTHROUGH,
                'response_style': 'excited_analytical'
            },
            'crypto_movement': {
                'keywords': ['transaction', 'wallet', 'bitcoin', 'ethereum'],
                'emotion': EmotionState.FOCUSED,
                'response_style': 'methodical_analysis'
            },
            'threat_identified': {
                'keywords': ['threat', 'malicious', 'suspicious', 'attack'],
                'emotion': EmotionState.CONCERNED,
                'response_style': 'serious_concern'
            }
        }

    def analyze_osint_context(self, investigation_data: Dict) -> Dict[str, Any]:
        """Analyze current OSINT investigation for avatar context"""

        context_info = {
            'primary_activity': investigation_data.get('activity_type', 'general_research'),
            'threat_level': self._calculate_threat_level(investigation_data),
            'progress_stage': investigation_data.get('progress', 0.0),
            'recent_findings': investigation_data.get('findings', []),
            'emotional_context': EmotionState.NEUTRAL
        }

        # Determine appropriate emotion based on findings
        findings_text = ' '.join(context_info['recent_findings'])
        for pattern_name, pattern_info in self.context_patterns.items():
            if any(keyword in findings_text.lower() for keyword in pattern_info['keywords']):
                context_info['emotional_context'] = pattern_info['emotion']
                context_info['response_style'] = pattern_info['response_style']
                break

        return context_info

    def _calculate_threat_level(self, investigation_data: Dict) -> float:
        """Calculate threat level from investigation data"""
        threat_indicators = investigation_data.get('threat_indicators', [])
        if not threat_indicators:
            return 0.0

        # Simple threat scoring
        threat_score = min(1.0, len(threat_indicators) * 0.2)
        return threat_score

class AdvancedAvatarController:
    """Main controller for advanced 3D avatar system"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.avatar_state = AdvancedAvatarState()
        self.emotion_engine = AdvancedEmotionEngine()
        self.tts_engine = AdvancedTTSEngine()
        self.renderer = GaussianSplattingRenderer(self.config['model_path'])
        self.osint_analyzer = OSINTContextAnalyzer()

        # Connection management
        self.websocket_connections: List[WebSocket] = []
        self.redis_client = None

        # Performance tracking
        self.performance_metrics = {
            'frames_rendered': 0,
            'average_fps': 0.0,
            'voice_generation_time': deque(maxlen=100),
            'emotion_processing_time': deque(maxlen=100)
        }

        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Animation loop control
        self.animation_running = False
        self.render_loop_task = None

    def _default_config(self) -> Dict:
        """Default configuration for advanced avatar"""
        return {
            'model_path': '/app/models/metahuman_osint_analyst',
            'redis_url': 'redis://redis:6379',
            'websocket_port': 8091,
            'target_fps': 120,
            'emotion_smoothing': 0.25,
            'voice_enabled': True,
            'gpu_memory_fraction': 0.6,
            'concurrent_processing': True,
            'osint_integration': True
        }

    async def initialize(self):
        """Initialize all avatar systems"""
        logger.info("Initializing Advanced Avatar Controller...")

        # Initialize Redis connection
        try:
            self.redis_client = await redis.from_url(
                self.config['redis_url'],
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")

        # Initialize rendering system
        if not await self.renderer.initialize():
            logger.error("Renderer initialization failed")
            return False

        # Initialize TTS engine
        if not await self.tts_engine.initialize():
            logger.error("TTS initialization failed")
            return False

        # Load emotion model to GPU
        self.emotion_engine = self.emotion_engine.to(self.emotion_engine.device)

        logger.info("Advanced Avatar Controller fully initialized")
        return True

    async def start_animation_loop(self):
        """Start the main animation rendering loop"""
        self.animation_running = True
        self.render_loop_task = asyncio.create_task(self._animation_loop())
        logger.info("Avatar animation loop started")

    async def _animation_loop(self):
        """Main animation rendering loop targeting 120 FPS"""
        target_frame_time = 1.0 / self.config['target_fps']

        while self.animation_running:
            loop_start = time.time()

            try:
                # Update avatar state
                await self._update_avatar_state()

                # Render current frame
                camera_params = self._get_default_camera_params()
                frame = await self.renderer.render_frame(
                    camera_params,
                    self.avatar_state.emotion
                )

                # Broadcast frame to all connected clients
                await self._broadcast_frame(frame)

                # Update performance metrics
                self.performance_metrics['frames_rendered'] += 1

                # Maintain target FPS
                elapsed = time.time() - loop_start
                if elapsed < target_frame_time:
                    await asyncio.sleep(target_frame_time - elapsed)

            except Exception as e:
                logger.error(f"Animation loop error: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error

    async def _update_avatar_state(self):
        """Update avatar state based on OSINT research context"""
        try:
            # Get current OSINT investigation status from Redis
            investigation_status = await self.redis_client.get("osint:current_investigation")

            if investigation_status:
                status_data = json.loads(investigation_status)

                # Analyze context for appropriate avatar response
                context = self.osint_analyzer.analyze_osint_context(status_data)

                # Update avatar state based on context
                self.avatar_state.emotion = context['emotional_context']
                self.avatar_state.current_investigation = status_data.get('investigation_id')
                self.avatar_state.investigation_progress = context['progress_stage']
                self.avatar_state.threat_level = context['threat_level']

                # Update personality traits based on investigation type
                if status_data.get('activity_type') == 'darknet_monitoring':
                    self.avatar_state.personality_traits['analytical_focus'] = 0.98
                elif status_data.get('activity_type') == 'breach_analysis':
                    self.avatar_state.personality_traits['empathy'] = 0.9

        except Exception as e:
            logger.debug(f"State update error: {e}")

    def _get_default_camera_params(self) -> Dict:
        """Get default camera parameters for avatar rendering"""
        return {
            'viewmat': np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 2],  # Camera distance
                [0, 0, 0, 1]
            ], dtype=np.float32),
            'projmat': self._create_projection_matrix(),
            'fov': 45.0,
            'aspect_ratio': 16/9
        }

    def _create_projection_matrix(self) -> np.ndarray:
        """Create projection matrix for rendering"""
        fov = np.radians(45.0)
        aspect = 16/9
        near = 0.01
        far = 100.0

        f = 1.0 / np.tan(fov / 2.0)
        proj = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

        return proj

    async def _broadcast_frame(self, frame: np.ndarray):
        """Broadcast rendered frame to all WebSocket connections"""
        if not self.websocket_connections:
            return

        # Encode frame as base64 for WebSocket transmission
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Create broadcast message
        message = {
            'type': 'avatar_frame',
            'frame': frame_base64,
            'state': {
                'emotion': self.avatar_state.emotion.value,
                'activity': self.avatar_state.activity.value,
                'progress': self.avatar_state.investigation_progress,
                'threat_level': self.avatar_state.threat_level
            },
            'timestamp': time.time()
        }

        # Broadcast to all connections
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except WebSocketDisconnect:
                disconnected.append(websocket)
            except Exception as e:
                logger.warning(f"Broadcast error: {e}")
                disconnected.append(websocket)

        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_connections.remove(ws)

    async def process_osint_update(self, update_data: Dict):
        """Process OSINT investigation updates for avatar response"""
        try:
            # Analyze the update for emotional context
            context = self.osint_analyzer.analyze_osint_context(update_data)

            # Generate appropriate avatar response
            response_text = self._generate_response_text(context, update_data)

            # Update avatar state
            self.avatar_state.emotion = context['emotional_context']

            # Generate voice response if enabled
            if self.config['voice_enabled'] and response_text:
                audio_data = await self.tts_engine.synthesize_speech(
                    response_text,
                    context['emotional_context'],
                    update_data.get('context', '')
                )

                # Broadcast audio to connected clients
                await self._broadcast_audio(audio_data, response_text)

            # Store update in Redis for persistence
            await self.redis_client.setex(
                "osint:current_investigation",
                3600,  # 1 hour expiry
                json.dumps(update_data)
            )

        except Exception as e:
            logger.error(f"OSINT update processing failed: {e}")

    def _generate_response_text(self, context: Dict, update_data: Dict) -> str:
        """Generate contextual response text for avatar"""

        activity_responses = {
            'breach_analysis': [
                "I found some interesting patterns in the breach data...",
                "The credential overlap analysis is revealing new connections.",
                "These breach patterns suggest a coordinated campaign."
            ],
            'darknet_monitoring': [
                "Monitoring marketplace activity shows unusual patterns.",
                "I've detected some concerning vendor behavior changes.",
                "The economic patterns are shifting in unexpected ways."
            ],
            'crypto_tracking': [
                "Transaction flow analysis is revealing new clustering patterns.",
                "I've identified some interesting wallet connections.",
                "The blockchain analysis is showing suspicious activity."
            ],
            'threat_analysis': [
                "Threat correlation analysis is complete.",
                "I've identified several high-priority threat indicators.",
                "The attack surface analysis reveals critical vulnerabilities."
            ]
        }

        activity_type = update_data.get('activity_type', 'general_research')
        responses = activity_responses.get(activity_type, ["Analysis complete."])

        # Select response based on context
        if context.get('threat_level', 0) > 0.7:
            return "⚠️ High threat level detected - immediate attention required!"
        elif update_data.get('findings_count', 0) > 10:
            return "Excellent progress! I've found significant intelligence."
        else:
            return np.random.choice(responses)

    async def _broadcast_audio(self, audio_data: bytes, text: str):
        """Broadcast audio response to connected clients"""
        if not audio_data:
            return

        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        message = {
            'type': 'avatar_speech',
            'audio': audio_base64,
            'text': text,
            'emotion': self.avatar_state.emotion.value,
            'timestamp': time.time()
        }

        # Broadcast to all connections
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Audio broadcast error: {e}")

    async def handle_websocket_connection(self, websocket: WebSocket):
        """Handle new WebSocket connection"""
        await websocket.accept()
        self.websocket_connections.append(websocket)

        try:
            # Send initial avatar state
            await websocket.send_text(json.dumps({
                'type': 'avatar_connected',
                'state': {
                    'emotion': self.avatar_state.emotion.value,
                    'activity': self.avatar_state.activity.value,
                    'personality': self.avatar_state.personality_traits
                },
                'capabilities': {
                    'voice_synthesis': BARK_AVAILABLE,
                    '3d_rendering': GSPLAT_AVAILABLE,
                    'metahuman_integration': METAHUMAN_AVAILABLE
                }
            }))

            # Handle incoming messages
            async for message in websocket.iter_text():
                await self._handle_websocket_message(websocket, message)

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if websocket in self.websocket_connections:
                self.websocket_connections.remove(websocket)

    async def _handle_websocket_message(self, websocket: WebSocket, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get('type')

            if message_type == 'user_input':
                # Process user interaction
                await self._process_user_input(data.get('input', ''))

            elif message_type == 'osint_update':
                # Process OSINT investigation update
                await self.process_osint_update(data.get('data', {}))

            elif message_type == 'emotion_override':
                # Manual emotion setting for testing
                emotion = data.get('emotion', 'neutral')
                self.avatar_state.emotion = EmotionState(emotion)

            elif message_type == 'performance_query':
                # Return performance metrics
                await websocket.send_text(json.dumps({
                    'type': 'performance_metrics',
                    'data': self.performance_metrics
                }))

        except Exception as e:
            logger.error(f"Message handling error: {e}")

    async def _process_user_input(self, user_input: str):
        """Process user input and generate avatar response"""
        try:
            # Analyze user input for emotional context
            # This could integrate with your existing OSINT tools

            # Generate appropriate response
            response_text = f"I understand you want to investigate: {user_input}"

            # Update avatar state
            self.avatar_state.emotion = EmotionState.FOCUSED

            # Generate voice response
            if self.config['voice_enabled']:
                audio_data = await self.tts_engine.synthesize_speech(
                    response_text,
                    EmotionState.FOCUSED,
                    osint_context="user_request"
                )
                await self._broadcast_audio(audio_data, response_text)

        except Exception as e:
            logger.error(f"User input processing failed: {e}")

    async def shutdown(self):
        """Graceful shutdown of avatar system"""
        logger.info("Shutting down Advanced Avatar Controller...")

        # Stop animation loop
        self.animation_running = False
        if self.render_loop_task:
            self.render_loop_task.cancel()

        # Close WebSocket connections
        for websocket in self.websocket_connections:
            try:
                await websocket.close()
            except:
                pass

        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()

        # Shutdown thread pool
        self.executor.shutdown(wait=True)

        logger.info("Advanced Avatar Controller shutdown complete")

# FastAPI application for avatar service
app = FastAPI(title="BEV Advanced Avatar System", version="1.0.0")

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global avatar controller instance
avatar_controller: Optional[AdvancedAvatarController] = None

@app.on_event("startup")
async def startup_event():
    """Initialize avatar system on startup"""
    global avatar_controller

    logger.info("Starting BEV Advanced Avatar System...")

    avatar_controller = AdvancedAvatarController()

    if await avatar_controller.initialize():
        await avatar_controller.start_animation_loop()
        logger.info("Avatar system ready for OSINT operations")
    else:
        logger.error("Avatar system initialization failed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global avatar_controller

    if avatar_controller:
        await avatar_controller.shutdown()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time avatar communication"""
    global avatar_controller

    if avatar_controller:
        await avatar_controller.handle_websocket_connection(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global avatar_controller

    status = {
        'status': 'healthy' if avatar_controller else 'initializing',
        'capabilities': {
            'gaussian_splatting': GSPLAT_AVAILABLE,
            'bark_tts': BARK_AVAILABLE,
            'metahuman': METAHUMAN_AVAILABLE
        },
        'performance': avatar_controller.performance_metrics if avatar_controller else {}
    }

    return JSONResponse(content=status)

@app.post("/osint/update")
async def osint_update_endpoint(update_data: Dict[str, Any]):
    """Endpoint for OSINT investigation updates"""
    global avatar_controller

    if avatar_controller:
        await avatar_controller.process_osint_update(update_data)
        return {"status": "processed"}
    else:
        raise HTTPException(status_code=503, detail="Avatar system not ready")

if __name__ == "__main__":
    import uvicorn

    # Run with optimal settings for RTX 4090
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8091,
        loop="uvloop",  # High-performance event loop
        workers=1,      # Single worker for GPU resource management
        log_level="info"
    )