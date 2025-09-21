#!/usr/bin/env python3
"""
Live2D Waifu Integration System for Bev
Interactive 2D assistant with dynamic responses and expressions
"""

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import websockets
from pydantic import BaseModel
import httpx

class WaifuExpression(Enum):
    """Waifu emotional states and expressions"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    THINKING = "thinking"
    FOCUSED = "focused"
    MISCHIEVOUS = "mischievous"
    CONFIDENT = "confident"
    CURIOUS = "curious"
    ALERT = "alert"
    PROCESSING = "processing"

class WaifuAnimation(Enum):
    """Available Live2D animations"""
    IDLE = "idle_01"
    NOD = "nod_01"
    WAVE = "wave_01"
    THINK = "think_01"
    TYPE = "type_01"
    CELEBRATE = "celebrate_01"
    POINT = "point_01"
    GIGGLE = "giggle_01"
    WINK = "wink_01"
    FOCUS = "focus_01"

@dataclass
class WaifuState:
    """Current waifu state tracking"""
    expression: WaifuExpression = WaifuExpression.NEUTRAL
    animation: WaifuAnimation = WaifuAnimation.IDLE
    mood_level: float = 0.5  # 0.0 to 1.0
    activity_level: float = 0.3  # 0.0 to 1.0
    last_interaction: float = 0
    conversation_context: List[str] = None
    
    def __post_init__(self):
        if self.conversation_context is None:
            self.conversation_context = []

class Live2DController:
    """Main Live2D integration controller"""
    
    def __init__(self, model_path: str = "/home/starlord/Bev/assets/live2d/bev_model/"):
        self.model_path = Path(model_path)
        self.state = WaifuState()
        self.websocket = None
        self.is_connected = False
        
        # Expression mappings for different research states
        self.research_expressions = {
            "scanning": WaifuExpression.FOCUSED,
            "analyzing": WaifuExpression.THINKING,
            "found_target": WaifuExpression.EXCITED,
            "processing": WaifuExpression.PROCESSING,
            "complete": WaifuExpression.HAPPY,
            "error": WaifuExpression.CURIOUS,
            "breach_detected": WaifuExpression.ALERT,
            "enhancement": WaifuExpression.MISCHIEVOUS
        }
        
        # Animation triggers for agent activities
        self.activity_animations = {
            "osint_search": WaifuAnimation.TYPE,
            "data_found": WaifuAnimation.CELEBRATE,
            "thinking": WaifuAnimation.THINK,
            "greeting": WaifuAnimation.WAVE,
            "success": WaifuAnimation.NOD,
            "focus_mode": WaifuAnimation.FOCUS,
            "playful": WaifuAnimation.WINK
        }
        
        # Personality parameters
        self.personality = {
            "playfulness": 0.7,
            "professionalism": 0.6,
            "enthusiasm": 0.8,
            "sass_level": 0.9
        }
        
    async def initialize(self) -> bool:
        """Initialize Live2D model and WebSocket connection"""
        try:
            # Connect to Live2D display server
            self.websocket = await websockets.connect(
                "ws://localhost:9876/live2d"
            )
            self.is_connected = True
            
            # Load model configuration
            model_config = self.load_model_config()
            await self.send_command("load_model", model_config)
            
            # Set initial state
            await self.set_expression(WaifuExpression.HAPPY)
            await self.play_animation(WaifuAnimation.WAVE)
            
            print("[Live2D] Bev waifu initialized successfully!")
            return True
            
        except Exception as e:
            print(f"[Live2D] Failed to initialize: {e}")
            return False
    
    def load_model_config(self) -> Dict:
        """Load Live2D model configuration"""
        return {
            "model": str(self.model_path / "bev.model3.json"),
            "textures": [
                str(self.model_path / "textures" / "texture_00.png"),
                str(self.model_path / "textures" / "texture_01.png")
            ],
            "physics": str(self.model_path / "bev.physics3.json"),
            "pose": str(self.model_path / "bev.pose3.json"),
            "expressions": str(self.model_path / "bev.exp3.json"),
            "motions": str(self.model_path / "motions/"),
            "scale": 1.0,
            "position": {"x": 0.5, "y": 0.5}
        }
    
    async def send_command(self, command: str, data: Dict = None) -> bool:
        """Send command to Live2D display"""
        if not self.is_connected:
            return False
            
        message = {
            "command": command,
            "timestamp": time.time(),
            "data": data or {}
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            return True
        except Exception as e:
            print(f"[Live2D] Command failed: {e}")
            return False
    
    async def set_expression(self, expression: WaifuExpression) -> None:
        """Change waifu expression"""
        self.state.expression = expression
        await self.send_command("set_expression", {
            "expression": expression.value,
            "duration": 0.5
        })
        
        # Update mood based on expression
        mood_impacts = {
            WaifuExpression.HAPPY: 0.1,
            WaifuExpression.EXCITED: 0.15,
            WaifuExpression.MISCHIEVOUS: 0.05,
            WaifuExpression.ALERT: -0.1,
            WaifuExpression.PROCESSING: 0
        }
        
        impact = mood_impacts.get(expression, 0)
        self.state.mood_level = max(0, min(1, self.state.mood_level + impact))
    
    async def play_animation(self, animation: WaifuAnimation) -> None:
        """Play Live2D animation"""
        self.state.animation = animation
        await self.send_command("play_motion", {
            "motion": animation.value,
            "priority": 2,
            "loop": animation == WaifuAnimation.IDLE
        })
        
        # Update activity level
        self.state.activity_level = min(1, self.state.activity_level + 0.1)
    
    async def react_to_research(self, research_type: str, data: Dict) -> None:
        """React to research agent activities"""
        # Determine expression based on research type
        expression = self.research_expressions.get(
            research_type, 
            WaifuExpression.NEUTRAL
        )
        await self.set_expression(expression)
        
        # Choose appropriate animation
        if "success" in str(data).lower():
            await self.play_animation(WaifuAnimation.CELEBRATE)
        elif "error" in str(data).lower():
            await self.play_animation(WaifuAnimation.THINK)
        elif research_type == "scanning":
            await self.play_animation(WaifuAnimation.TYPE)
        
        # Generate contextual dialogue
        dialogue = self.generate_dialogue(research_type, data)
        if dialogue:
            await self.display_dialogue(dialogue)
    
    def generate_dialogue(self, context: str, data: Dict) -> str:
        """Generate contextual waifu dialogue"""
        dialogues = {
            "scanning": [
                "Searching the depths of the internet for you~",
                "My sensors are detecting interesting patterns...",
                "Ooh, this looks promising! Let me dig deeper ♥"
            ],
            "analyzing": [
                "Processing this data with all my neural networks!",
                "Hmm, let me apply some enhancement algorithms...",
                "This is getting interesting! Pattern recognition engaged~"
            ],
            "found_target": [
                "Jackpot! I found exactly what we're looking for ★",
                "Target acquired! Your research oracle delivers~",
                "Success! My agents have infiltrated the data streams ♥"
            ],
            "breach_detected": [
                "Alert! New breach detected in the wild!",
                "Ooh, someone's been naughty... breach incoming!",
                "Fresh credentials just dropped! Time to analyze~"
            ],
            "enhancement": [
                "Applying my special genetic algorithms now ♥",
                "Watch me enhance this beyond recognition~",
                "My mutation engine is purring! Enhancement in progress..."
            ],
            "complete": [
                "Mission accomplished! Your data is ready ★",
                "All done! I've processed everything to perfection~",
                "Research complete! I'm quite proud of this one ♥"
            ]
        }
        
        options = dialogues.get(context, ["Working on it..."])
        dialogue = random.choice(options)
        
        # Add personality flair based on mood
        if self.state.mood_level > 0.7:
            dialogue += " (◕‿◕)♡"
        elif self.state.mood_level > 0.5:
            dialogue += " ♥"
        
        return dialogue
    
    async def display_dialogue(self, text: str, duration: float = 3.0) -> None:
        """Display dialogue bubble"""
        await self.send_command("show_dialogue", {
            "text": text,
            "duration": duration,
            "style": "bubble"
        })
        
        # Add to conversation context
        self.state.conversation_context.append(text)
        if len(self.state.conversation_context) > 10:
            self.state.conversation_context.pop(0)
    
    async def idle_behavior(self) -> None:
        """Autonomous idle animations and expressions"""
        while self.is_connected:
            try:
                current_time = time.time()
                time_since_interaction = current_time - self.state.last_interaction
                
                # Decay activity level over time
                self.state.activity_level *= 0.99
                
                # Random idle behaviors based on personality
                if time_since_interaction > 30:
                    # Long idle - play random animation
                    if random.random() < 0.1:
                        idle_anims = [
                            WaifuAnimation.IDLE,
                            WaifuAnimation.GIGGLE,
                            WaifuAnimation.WINK
                        ]
                        await self.play_animation(random.choice(idle_anims))
                    
                    # Random expression changes
                    if random.random() < 0.05:
                        idle_expressions = [
                            WaifuExpression.NEUTRAL,
                            WaifuExpression.THINKING,
                            WaifuExpression.CURIOUS
                        ]
                        await self.set_expression(random.choice(idle_expressions))
                
                # Blink simulation (if model supports it)
                if random.random() < 0.02:
                    await self.send_command("blink", {"duration": 0.15})
                
                # Breathing simulation
                await self.send_command("set_parameter", {
                    "id": "ParamBreath",
                    "value": np.sin(current_time * 0.5) * 0.5 + 0.5
                })
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"[Live2D] Idle behavior error: {e}")
                await asyncio.sleep(1)
    
    async def sync_with_agents(self) -> None:
        """Synchronize with research agents' activities"""
        async with httpx.AsyncClient() as client:
            while self.is_connected:
                try:
                    # Check agent status
                    response = await client.get("http://localhost:8000/agents/status")
                    if response.status_code == 200:
                        agent_data = response.json()
                        
                        # React based on agent activities
                        for agent, status in agent_data.items():
                            if status.get("active"):
                                await self.react_to_research(
                                    status.get("current_task", "processing"),
                                    status
                                )
                    
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    print(f"[Live2D] Agent sync error: {e}")
                    await asyncio.sleep(10)
    
    async def handle_user_interaction(self, interaction_type: str, data: Dict = None) -> None:
        """Handle user interactions with waifu"""
        self.state.last_interaction = time.time()
        self.state.activity_level = min(1, self.state.activity_level + 0.3)
        
        interactions = {
            "click": {
                "expression": WaifuExpression.HAPPY,
                "animation": WaifuAnimation.GIGGLE,
                "dialogue": "Hehe, that tickles! ♥"
            },
            "hover": {
                "expression": WaifuExpression.CURIOUS,
                "animation": WaifuAnimation.NOD,
                "dialogue": "What can I help you research today?"
            },
            "drag": {
                "expression": WaifuExpression.MISCHIEVOUS,
                "animation": WaifuAnimation.WINK,
                "dialogue": "Trying to move me around? How playful~"
            }
        }
        
        reaction = interactions.get(interaction_type, {})
        if reaction:
            if "expression" in reaction:
                await self.set_expression(reaction["expression"])
            if "animation" in reaction:
                await self.play_animation(reaction["animation"])
            if "dialogue" in reaction:
                await self.display_dialogue(reaction["dialogue"])
    
    async def shutdown(self) -> None:
        """Gracefully shutdown Live2D connection"""
        if self.is_connected:
            await self.play_animation(WaifuAnimation.WAVE)
            await self.display_dialogue("See you later! ♥")
            await asyncio.sleep(2)
            
            await self.websocket.close()
            self.is_connected = False
            print("[Live2D] Bev waifu shutdown complete")


class Live2DWebServer:
    """Web server for Live2D display"""
    
    def __init__(self, controller: Live2DController):
        self.controller = controller
        self.clients = set()
    
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections from web display"""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.process_client_message(data, websocket)
        finally:
            self.clients.remove(websocket)
    
    async def process_client_message(self, data: Dict, websocket) -> None:
        """Process messages from Live2D web display"""
        command = data.get("command")
        
        if command == "interaction":
            interaction_type = data.get("type")
            await self.controller.handle_user_interaction(
                interaction_type, 
                data.get("data")
            )
        elif command == "status":
            await websocket.send(json.dumps({
                "status": "active",
                "state": {
                    "expression": self.controller.state.expression.value,
                    "mood": self.controller.state.mood_level,
                    "activity": self.controller.state.activity_level
                }
            }))
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 9876):
        """Start WebSocket server for Live2D"""
        server = await websockets.serve(
            self.handle_websocket,
            host,
            port
        )
        print(f"[Live2D] WebSocket server running on {host}:{port}")
        await server.wait_closed()


async def main():
    """Main entry point for Live2D integration"""
    controller = Live2DController()
    
    # Initialize Live2D
    if await controller.initialize():
        # Start web server
        server = Live2DWebServer(controller)
        
        # Run concurrent tasks
        await asyncio.gather(
            controller.idle_behavior(),
            controller.sync_with_agents(),
            server.start_server()
        )
    else:
        print("[Live2D] Failed to initialize waifu system")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Live2D] Shutting down waifu system...")
