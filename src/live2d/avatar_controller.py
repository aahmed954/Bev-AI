#!/usr/bin/env python3
"""
Live2D Avatar Controller
Real-time avatar with emotion engine and voice synthesis
"""

import asyncio
import logging
import json
import time
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import redis.asyncio as redis
import pyttsx3
import torch
import torch.nn as nn
import torchaudio
from transformers import (
    pipeline,
    AutoModelForAudioClassification,
    AutoProcessor
)
import cv2
from PIL import Image
import io
import wave
import pyaudio
from collections import deque
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmotionState(Enum):
    """Avatar emotion states"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    THINKING = "thinking"
    EXCITED = "excited"
    CONFIDENT = "confident"
    CONFUSED = "confused"
    LISTENING = "listening"

class AnimationState(Enum):
    """Avatar animation states"""
    IDLE = "idle"
    TALKING = "talking"
    BLINKING = "blinking"
    NODDING = "nodding"
    SHAKING_HEAD = "shaking_head"
    WAVING = "waving"
    THINKING_POSE = "thinking"
    CELEBRATING = "celebrating"

@dataclass
class AvatarState:
    """Current avatar state"""
    emotion: EmotionState = EmotionState.NEUTRAL
    animation: AnimationState = AnimationState.IDLE
    expression_intensity: float = 1.0
    mouth_openness: float = 0.0
    eye_openness: float = 1.0
    head_x: float = 0.0
    head_y: float = 0.0
    body_x: float = 0.0
    body_y: float = 0.0
    breath: float = 0.0
    talking: bool = False
    blinking: bool = False
    custom_params: Dict[str, float] = field(default_factory=dict)

@dataclass
class EmotionVector:
    """Emotion analysis vector"""
    happiness: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    surprise: float = 0.0
    fear: float = 0.0
    disgust: float = 0.0
    neutral: float = 1.0
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class EmotionEngine(nn.Module):
    """Neural network for emotion synthesis and transitions"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hidden=None):
        # LSTM for temporal dynamics
        lstm_out, hidden = self.lstm(x, hidden)

        # Self-attention for emotion relationships
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Final emotion prediction
        x = self.dropout(attn_out[:, -1, :])
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        emotions = self.softmax(self.fc2(x))

        return emotions, hidden

class VoiceSynthesizer:
    """Voice synthesis with emotion modulation"""

    def __init__(self):
        self.engine = pyttsx3.init()
        self.setup_voice()
        self.audio_buffer = deque(maxlen=10)

    def setup_voice(self):
        """Configure voice parameters"""
        voices = self.engine.getProperty('voices')
        # Try to set a female voice if available
        for voice in voices:
            if 'female' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

        self.engine.setProperty('rate', 150)  # Speed
        self.engine.setProperty('volume', 0.9)  # Volume

    def modulate_for_emotion(self, emotion: EmotionState):
        """Adjust voice parameters based on emotion"""
        emotion_params = {
            EmotionState.HAPPY: {'rate': 160, 'volume': 0.95, 'pitch': 1.1},
            EmotionState.SAD: {'rate': 130, 'volume': 0.7, 'pitch': 0.9},
            EmotionState.ANGRY: {'rate': 170, 'volume': 1.0, 'pitch': 0.95},
            EmotionState.EXCITED: {'rate': 180, 'volume': 1.0, 'pitch': 1.2},
            EmotionState.THINKING: {'rate': 140, 'volume': 0.8, 'pitch': 1.0},
            EmotionState.CONFUSED: {'rate': 135, 'volume': 0.85, 'pitch': 1.05},
        }

        params = emotion_params.get(emotion, {'rate': 150, 'volume': 0.9, 'pitch': 1.0})
        self.engine.setProperty('rate', params['rate'])
        self.engine.setProperty('volume', params['volume'])

    async def synthesize(self, text: str, emotion: EmotionState) -> bytes:
        """Synthesize speech with emotion"""
        try:
            self.modulate_for_emotion(emotion)

            # Generate audio to file (pyttsx3 limitation)
            temp_file = f"/tmp/speech_{hash(text)}.wav"
            self.engine.save_to_file(text, temp_file)
            self.engine.runAndWait()

            # Read the audio file
            with open(temp_file, 'rb') as f:
                audio_data = f.read()

            return audio_data

        except Exception as e:
            logger.error(f"Voice synthesis failed: {e}")
            return b''

class Live2DAvatarController:
    """Main Live2D avatar controller"""

    def __init__(self, config_path: str = "config/avatar.yaml"):
        self.config = self._load_config(config_path)
        self.avatar_state = AvatarState()
        self.emotion_engine = EmotionEngine()
        self.voice_synthesizer = VoiceSynthesizer()
        self.emotion_history = deque(maxlen=100)
        self.redis_client = None
        self.websocket_connections: List[WebSocket] = []
        self.animation_loop_task = None
        self.current_emotion_vector = EmotionVector()

        # Animation timing
        self.last_blink_time = time.time()
        self.blink_interval = 3.0
        self.breath_phase = 0.0

        # Emotion analyzer (using Hugging Face)
        try:
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
        except:
            logger.warning("Could not load emotion analyzer model")
            self.emotion_analyzer = None

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        return {
            'redis_url': 'redis://redis:6379',
            'model_path': '/app/models/avatar.model',
            'animation_fps': 60,
            'emotion_smoothing': 0.3,
            'voice_enabled': True,
            'websocket_heartbeat': 30
        }

    async def initialize(self):
        """Initialize avatar controller"""
        try:
            # Initialize Redis
            self.redis_client = await redis.from_url(
                self.config['redis_url'],
                encoding="utf-8",
                decode_responses=True
            )

            # Load saved state if exists
            await self._load_avatar_state()

            # Start animation loop
            self.animation_loop_task = asyncio.create_task(self._animation_loop())

            logger.info("Avatar controller initialized")

        except Exception as e:
            logger.error(f"Failed to initialize avatar: {e}")
            raise

    async def _load_avatar_state(self):
        """Load saved avatar state from Redis"""
        try:
            state_data = await self.redis_client.get("avatar:state")
            if state_data:
                state = json.loads(state_data)
                self.avatar_state.emotion = EmotionState(state.get('emotion', 'neutral'))
                self.avatar_state.expression_intensity = state.get('expression_intensity', 1.0)
        except Exception as e:
            logger.error(f"Failed to load avatar state: {e}")

    async def _save_avatar_state(self):
        """Save avatar state to Redis"""
        try:
            state_data = {
                'emotion': self.avatar_state.emotion.value,
                'animation': self.avatar_state.animation.value,
                'expression_intensity': self.avatar_state.expression_intensity,
                'timestamp': datetime.now().isoformat()
            }
            await self.redis_client.set("avatar:state", json.dumps(state_data))
        except Exception as e:
            logger.error(f"Failed to save avatar state: {e}")

    async def analyze_emotion(self, text: str) -> EmotionVector:
        """Analyze emotion from text"""
        try:
            if self.emotion_analyzer:
                results = self.emotion_analyzer(text)[0]
                emotion_vector = EmotionVector()

                # Map results to emotion vector
                for result in results:
                    label = result['label'].lower()
                    score = result['score']

                    if label == 'joy':
                        emotion_vector.happiness = score
                    elif label == 'sadness':
                        emotion_vector.sadness = score
                    elif label == 'anger':
                        emotion_vector.anger = score
                    elif label == 'surprise':
                        emotion_vector.surprise = score
                    elif label == 'fear':
                        emotion_vector.fear = score
                    elif label == 'disgust':
                        emotion_vector.disgust = score
                    elif label == 'neutral':
                        emotion_vector.neutral = score

                return emotion_vector
            else:
                # Fallback to simple keyword analysis
                return self._simple_emotion_analysis(text)

        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return EmotionVector()

    def _simple_emotion_analysis(self, text: str) -> EmotionVector:
        """Simple keyword-based emotion analysis"""
        emotion_vector = EmotionVector()
        text_lower = text.lower()

        # Simple keyword matching
        happy_words = ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing']
        sad_words = ['sad', 'unhappy', 'depressed', 'down', 'blue']
        angry_words = ['angry', 'mad', 'furious', 'annoyed', 'frustrated']

        for word in happy_words:
            if word in text_lower:
                emotion_vector.happiness += 0.3

        for word in sad_words:
            if word in text_lower:
                emotion_vector.sadness += 0.3

        for word in angry_words:
            if word in text_lower:
                emotion_vector.anger += 0.3

        # Normalize
        total = (emotion_vector.happiness + emotion_vector.sadness +
                emotion_vector.anger + emotion_vector.neutral)
        if total > 0:
            emotion_vector.happiness /= total
            emotion_vector.sadness /= total
            emotion_vector.anger /= total
            emotion_vector.neutral = max(0, 1 - (emotion_vector.happiness +
                                                  emotion_vector.sadness +
                                                  emotion_vector.anger))

        return emotion_vector

    async def update_emotion(self, emotion_vector: EmotionVector):
        """Update avatar emotion based on emotion vector"""
        try:
            # Store in history
            self.emotion_history.append(emotion_vector)
            self.current_emotion_vector = emotion_vector

            # Determine dominant emotion
            emotions = {
                EmotionState.HAPPY: emotion_vector.happiness,
                EmotionState.SAD: emotion_vector.sadness,
                EmotionState.ANGRY: emotion_vector.anger,
                EmotionState.SURPRISED: emotion_vector.surprise,
                EmotionState.NEUTRAL: emotion_vector.neutral
            }

            dominant_emotion = max(emotions.items(), key=lambda x: x[1])

            # Smooth transition
            if dominant_emotion[1] > 0.3:  # Threshold for emotion change
                await self.transition_emotion(dominant_emotion[0])

        except Exception as e:
            logger.error(f"Failed to update emotion: {e}")

    async def transition_emotion(self, target_emotion: EmotionState, duration: float = 1.0):
        """Smoothly transition to target emotion"""
        try:
            start_time = time.time()
            start_emotion = self.avatar_state.emotion

            # Calculate transition steps
            steps = int(duration * self.config['animation_fps'])

            for i in range(steps):
                progress = i / steps

                # Ease-in-out interpolation
                t = progress * progress * (3.0 - 2.0 * progress)

                # Update expression intensity
                self.avatar_state.expression_intensity = 0.5 + 0.5 * t

                # Send update to clients
                await self._broadcast_state()

                await asyncio.sleep(1.0 / self.config['animation_fps'])

            self.avatar_state.emotion = target_emotion
            await self._save_avatar_state()

            logger.info(f"Transitioned from {start_emotion.value} to {target_emotion.value}")

        except Exception as e:
            logger.error(f"Failed to transition emotion: {e}")

    async def speak(self, text: str, emotion: Optional[EmotionState] = None):
        """Make avatar speak with lip sync"""
        try:
            # Analyze emotion if not provided
            if emotion is None:
                emotion_vector = await self.analyze_emotion(text)
                await self.update_emotion(emotion_vector)
                emotion = self.avatar_state.emotion

            # Start talking animation
            self.avatar_state.animation = AnimationState.TALKING
            self.avatar_state.talking = True
            await self._broadcast_state()

            # Synthesize voice
            if self.config['voice_enabled']:
                audio_data = await self.voice_synthesizer.synthesize(text, emotion)

                # Broadcast audio to clients
                await self._broadcast_audio(audio_data)

                # Simulate lip sync
                await self._animate_lip_sync(text)

            # Stop talking animation
            self.avatar_state.animation = AnimationState.IDLE
            self.avatar_state.talking = False
            await self._broadcast_state()

        except Exception as e:
            logger.error(f"Failed to speak: {e}")

    async def _animate_lip_sync(self, text: str):
        """Animate mouth for lip sync"""
        try:
            # Simple phoneme-based lip sync
            phoneme_duration = 0.1  # seconds per phoneme

            for char in text:
                if char in 'aeiouAEIOU':
                    # Open mouth for vowels
                    self.avatar_state.mouth_openness = 0.8
                elif char in 'bpmBPM':
                    # Close mouth for bilabials
                    self.avatar_state.mouth_openness = 0.1
                elif char in 'fvFV':
                    # Partial open for fricatives
                    self.avatar_state.mouth_openness = 0.3
                elif char == ' ':
                    # Close for pauses
                    self.avatar_state.mouth_openness = 0.0
                else:
                    # Default position
                    self.avatar_state.mouth_openness = 0.4

                await self._broadcast_state()
                await asyncio.sleep(phoneme_duration)

            # Close mouth
            self.avatar_state.mouth_openness = 0.0
            await self._broadcast_state()

        except Exception as e:
            logger.error(f"Lip sync animation failed: {e}")

    async def perform_gesture(self, gesture: str):
        """Perform a specific gesture"""
        try:
            gesture_map = {
                'nod': AnimationState.NODDING,
                'shake': AnimationState.SHAKING_HEAD,
                'wave': AnimationState.WAVING,
                'think': AnimationState.THINKING_POSE,
                'celebrate': AnimationState.CELEBRATING
            }

            if gesture in gesture_map:
                self.avatar_state.animation = gesture_map[gesture]
                await self._broadcast_state()

                # Hold gesture for duration
                await asyncio.sleep(2.0)

                # Return to idle
                self.avatar_state.animation = AnimationState.IDLE
                await self._broadcast_state()

        except Exception as e:
            logger.error(f"Failed to perform gesture: {e}")

    async def _animation_loop(self):
        """Main animation loop for automatic behaviors"""
        while True:
            try:
                current_time = time.time()

                # Blinking
                if current_time - self.last_blink_time > self.blink_interval:
                    await self._blink()
                    self.last_blink_time = current_time
                    self.blink_interval = np.random.uniform(2, 5)

                # Breathing
                self.breath_phase += 0.05
                self.avatar_state.breath = 0.5 + 0.5 * np.sin(self.breath_phase)

                # Idle movement
                if self.avatar_state.animation == AnimationState.IDLE:
                    # Subtle head movement
                    self.avatar_state.head_x = 0.05 * np.sin(self.breath_phase * 0.5)
                    self.avatar_state.head_y = 0.03 * np.sin(self.breath_phase * 0.7)

                # Broadcast state periodically
                await self._broadcast_state()

                # Control loop rate
                await asyncio.sleep(1.0 / self.config['animation_fps'])

            except Exception as e:
                logger.error(f"Animation loop error: {e}")
                await asyncio.sleep(1.0)

    async def _blink(self):
        """Perform blink animation"""
        try:
            # Close eyes
            self.avatar_state.eye_openness = 0.0
            self.avatar_state.blinking = True
            await self._broadcast_state()

            await asyncio.sleep(0.1)

            # Open eyes
            self.avatar_state.eye_openness = 1.0
            self.avatar_state.blinking = False
            await self._broadcast_state()

        except Exception as e:
            logger.error(f"Blink animation failed: {e}")

    async def _broadcast_state(self):
        """Broadcast current state to all WebSocket connections"""
        if not self.websocket_connections:
            return

        state_data = {
            'type': 'state_update',
            'data': {
                'emotion': self.avatar_state.emotion.value,
                'animation': self.avatar_state.animation.value,
                'expression_intensity': self.avatar_state.expression_intensity,
                'mouth_openness': self.avatar_state.mouth_openness,
                'eye_openness': self.avatar_state.eye_openness,
                'head_x': self.avatar_state.head_x,
                'head_y': self.avatar_state.head_y,
                'body_x': self.avatar_state.body_x,
                'body_y': self.avatar_state.body_y,
                'breath': self.avatar_state.breath,
                'talking': self.avatar_state.talking,
                'blinking': self.avatar_state.blinking,
                'custom_params': self.avatar_state.custom_params,
                'timestamp': time.time()
            }
        }

        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(state_data)
            except:
                disconnected.append(websocket)

        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_connections.remove(ws)

    async def _broadcast_audio(self, audio_data: bytes):
        """Broadcast audio data to WebSocket connections"""
        if not self.websocket_connections:
            return

        audio_message = {
            'type': 'audio_data',
            'data': base64.b64encode(audio_data).decode('utf-8'),
            'format': 'wav'
        }

        for websocket in self.websocket_connections[:]:
            try:
                await websocket.send_json(audio_message)
            except:
                pass

    async def handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        await websocket.accept()
        self.websocket_connections.append(websocket)

        try:
            # Send initial state
            await self._broadcast_state()

            # Handle incoming messages
            while True:
                data = await websocket.receive_json()
                await self._handle_websocket_message(websocket, data)

        except WebSocketDisconnect:
            self.websocket_connections.remove(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if websocket in self.websocket_connections:
                self.websocket_connections.remove(websocket)

    async def _handle_websocket_message(self, websocket: WebSocket, data: Dict):
        """Handle incoming WebSocket message"""
        try:
            message_type = data.get('type')

            if message_type == 'speak':
                text = data.get('text', '')
                emotion = data.get('emotion')
                if emotion:
                    emotion = EmotionState(emotion)
                await self.speak(text, emotion)

            elif message_type == 'gesture':
                gesture = data.get('gesture')
                await self.perform_gesture(gesture)

            elif message_type == 'emotion':
                emotion = EmotionState(data.get('emotion', 'neutral'))
                await self.transition_emotion(emotion)

            elif message_type == 'set_param':
                param = data.get('param')
                value = data.get('value')
                if param and value is not None:
                    setattr(self.avatar_state, param, value)
                    await self._broadcast_state()

            elif message_type == 'ping':
                await websocket.send_json({'type': 'pong'})

        except Exception as e:
            logger.error(f"Failed to handle WebSocket message: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Get current avatar status"""
        return {
            'emotion': self.avatar_state.emotion.value,
            'animation': self.avatar_state.animation.value,
            'connected_clients': len(self.websocket_connections),
            'emotion_history_size': len(self.emotion_history),
            'voice_enabled': self.config['voice_enabled']
        }

# FastAPI application
app = FastAPI(title="Live2D Avatar Controller", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

avatar_controller = None

@app.on_event("startup")
async def startup_event():
    global avatar_controller
    avatar_controller = Live2DAvatarController()
    await avatar_controller.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    if avatar_controller:
        if avatar_controller.animation_loop_task:
            avatar_controller.animation_loop_task.cancel()
        if avatar_controller.redis_client:
            await avatar_controller.redis_client.close()

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/status")
async def get_status():
    if not avatar_controller:
        raise HTTPException(status_code=503, detail="Avatar controller not initialized")
    return await avatar_controller.get_status()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not avatar_controller:
        await websocket.close(code=1003, reason="Avatar controller not initialized")
        return
    await avatar_controller.handle_websocket(websocket)

@app.post("/speak")
async def speak(text: str, emotion: Optional[str] = None):
    if not avatar_controller:
        raise HTTPException(status_code=503, detail="Avatar controller not initialized")

    emotion_state = EmotionState(emotion) if emotion else None
    await avatar_controller.speak(text, emotion_state)
    return {"status": "speaking"}

@app.post("/gesture")
async def perform_gesture(gesture: str):
    if not avatar_controller:
        raise HTTPException(status_code=503, detail="Avatar controller not initialized")

    await avatar_controller.perform_gesture(gesture)
    return {"status": "gesture_performed"}

@app.post("/emotion")
async def set_emotion(emotion: str):
    if not avatar_controller:
        raise HTTPException(status_code=503, detail="Avatar controller not initialized")

    try:
        emotion_state = EmotionState(emotion)
        await avatar_controller.transition_emotion(emotion_state)
        return {"status": "emotion_set", "emotion": emotion}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid emotion: {emotion}")

@app.post("/analyze")
async def analyze_emotion(text: str):
    if not avatar_controller:
        raise HTTPException(status_code=503, detail="Avatar controller not initialized")

    emotion_vector = await avatar_controller.analyze_emotion(text)
    return {
        'happiness': emotion_vector.happiness,
        'sadness': emotion_vector.sadness,
        'anger': emotion_vector.anger,
        'surprise': emotion_vector.surprise,
        'fear': emotion_vector.fear,
        'disgust': emotion_vector.disgust,
        'neutral': emotion_vector.neutral
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8091)