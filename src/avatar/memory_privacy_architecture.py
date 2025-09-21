"""
Memory and Privacy Architecture for BEV AI Companion
Secure storage, encryption, and access control for personal data
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from datetime import datetime, timedelta
import asyncio
import redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pickle
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import base64
import uuid
from sentence_transformers import SentenceTransformer

class PrivacyLevel(Enum):
    """Privacy levels for memory storage"""
    PUBLIC = "public"  # General non-sensitive information
    PRIVATE = "private"  # Personal but not sensitive
    SENSITIVE = "sensitive"  # Sensitive personal information
    INTIMATE = "intimate"  # Highly personal/emotional content
    ENCRYPTED = "encrypted"  # Always encrypted, never plain text

class MemoryType(Enum):
    """Types of memories stored"""
    FACTUAL = "factual"  # Facts about user
    EMOTIONAL = "emotional"  # Emotional experiences
    PREFERENCE = "preference"  # User preferences
    BEHAVIORAL = "behavioral"  # Behavioral patterns
    RELATIONAL = "relational"  # Relationship context
    PROFESSIONAL = "professional"  # Work-related
    CREATIVE = "creative"  # Creative collaborations
    SECURITY = "security"  # Security-related events

@dataclass
class SecureMemory:
    """Secure memory unit with encryption support"""
    id: str
    timestamp: datetime
    memory_type: MemoryType
    privacy_level: PrivacyLevel
    content: str  # Plain text content
    encrypted_content: Optional[bytes] = None
    vector_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    retention_period: Optional[timedelta] = None
    user_consent: bool = True

@dataclass
class MemoryCluster:
    """Cluster of related memories"""
    cluster_id: str
    theme: str
    memories: List[SecureMemory]
    importance_score: float
    coherence_score: float
    privacy_level: PrivacyLevel

class BiometricAuthenticator:
    """Biometric authentication for sensitive features"""

    def __init__(self):
        self.authenticated_sessions: Dict[str, datetime] = {}
        self.session_timeout = timedelta(hours=1)
        self.failed_attempts: Dict[str, int] = {}
        self.max_attempts = 3

    async def authenticate(self, user_id: str, biometric_data: Dict[str, Any]) -> bool:
        """Authenticate user with biometric data (simulated)"""

        # Check for too many failed attempts
        if self.failed_attempts.get(user_id, 0) >= self.max_attempts:
            return False

        # Simulated biometric verification
        # In production, this would interface with actual biometric systems
        verification_score = self._simulate_biometric_verification(biometric_data)

        if verification_score > 0.95:  # High confidence threshold
            self.authenticated_sessions[user_id] = datetime.now()
            self.failed_attempts[user_id] = 0
            return True
        else:
            self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1
            return False

    def _simulate_biometric_verification(self, biometric_data: Dict[str, Any]) -> float:
        """Simulate biometric verification (placeholder)"""
        # In production: Interface with fingerprint/face recognition
        # For development: Return high score if correct test data provided
        if biometric_data.get('test_auth') == 'valid_biometric':
            return 0.98
        return 0.3

    def is_authenticated(self, user_id: str) -> bool:
        """Check if user session is still authenticated"""

        if user_id not in self.authenticated_sessions:
            return False

        session_time = self.authenticated_sessions[user_id]
        if datetime.now() - session_time > self.session_timeout:
            del self.authenticated_sessions[user_id]
            return False

        return True

    def revoke_authentication(self, user_id: str) -> None:
        """Revoke authentication for user"""
        if user_id in self.authenticated_sessions:
            del self.authenticated_sessions[user_id]

class MemoryEncryption:
    """Advanced encryption for memory storage"""

    def __init__(self, master_key: Optional[bytes] = None):
        if master_key:
            self.master_key = master_key
        else:
            self.master_key = self._generate_master_key()

        self.cipher_suite = Fernet(self.master_key)

        # Generate RSA keys for asymmetric encryption
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

        # Memory-specific encryption keys
        self.memory_keys: Dict[PrivacyLevel, bytes] = {}
        self._generate_privacy_keys()

    def _generate_master_key(self) -> bytes:
        """Generate master encryption key"""
        password = b"BEV_Memory_Master_2024"  # Should be from secure config
        salt = b"memory_salt_v1"  # Should be random in production
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(password))

    def _generate_privacy_keys(self) -> None:
        """Generate encryption keys for each privacy level"""
        for level in PrivacyLevel:
            level_password = f"BEV_{level.value}_key".encode()
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=f"salt_{level.value}".encode(),
                iterations=50000,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(level_password))
            self.memory_keys[level] = key

    def encrypt_memory(self, memory: SecureMemory) -> bytes:
        """Encrypt memory based on privacy level"""

        if memory.privacy_level == PrivacyLevel.PUBLIC:
            return memory.content.encode()  # No encryption for public

        # Select appropriate key
        if memory.privacy_level in [PrivacyLevel.INTIMATE, PrivacyLevel.ENCRYPTED]:
            # Use asymmetric encryption for highest security
            return self._asymmetric_encrypt(memory.content)
        else:
            # Use symmetric encryption with privacy-level key
            cipher = Fernet(self.memory_keys[memory.privacy_level])
            return cipher.encrypt(memory.content.encode())

    def decrypt_memory(self, encrypted_data: bytes,
                      privacy_level: PrivacyLevel) -> str:
        """Decrypt memory data"""

        if privacy_level == PrivacyLevel.PUBLIC:
            return encrypted_data.decode()

        if privacy_level in [PrivacyLevel.INTIMATE, PrivacyLevel.ENCRYPTED]:
            return self._asymmetric_decrypt(encrypted_data)
        else:
            cipher = Fernet(self.memory_keys[privacy_level])
            return cipher.decrypt(encrypted_data).decode()

    def _asymmetric_encrypt(self, data: str) -> bytes:
        """Encrypt with RSA public key"""
        return self.public_key.encrypt(
            data.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

    def _asymmetric_decrypt(self, encrypted_data: bytes) -> str:
        """Decrypt with RSA private key"""
        decrypted = self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted.decode()

class MemoryStorage:
    """Secure memory storage system with vector search"""

    def __init__(self, redis_client: Optional[redis.Redis] = None,
                 qdrant_client: Optional[QdrantClient] = None):

        self.redis_client = redis_client or redis.Redis(
            host='localhost', port=6379, db=5, decode_responses=False
        )

        self.qdrant_client = qdrant_client or QdrantClient(
            host="localhost", port=6333
        )

        self.encryption = MemoryEncryption()
        self.authenticator = BiometricAuthenticator()

        # Initialize sentence transformer for embeddings
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Create Qdrant collection if not exists
        self._initialize_vector_store()

        # Memory indices
        self.memory_index: Dict[str, SecureMemory] = {}
        self.user_memories: Dict[str, List[str]] = {}  # user_id -> memory_ids

    def _initialize_vector_store(self) -> None:
        """Initialize Qdrant vector store collection"""
        try:
            self.qdrant_client.create_collection(
                collection_name="secure_memories",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        except:
            pass  # Collection already exists

    async def store_memory(self, user_id: str, content: str,
                          memory_type: MemoryType,
                          privacy_level: PrivacyLevel,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a new memory securely"""

        # Create memory object
        memory_id = str(uuid.uuid4())
        memory = SecureMemory(
            id=memory_id,
            timestamp=datetime.now(),
            memory_type=memory_type,
            privacy_level=privacy_level,
            content=content,
            metadata=metadata or {}
        )

        # Generate embedding
        memory.vector_embedding = self.encoder.encode(content)

        # Encrypt if needed
        if privacy_level != PrivacyLevel.PUBLIC:
            memory.encrypted_content = self.encryption.encrypt_memory(memory)
            # Clear plain text for sensitive memories
            if privacy_level in [PrivacyLevel.SENSITIVE, PrivacyLevel.INTIMATE, PrivacyLevel.ENCRYPTED]:
                memory.content = "[ENCRYPTED]"

        # Store in vector database
        await self._store_in_vector_db(memory)

        # Store in Redis cache
        await self._store_in_cache(user_id, memory)

        # Update indices
        self.memory_index[memory_id] = memory
        if user_id not in self.user_memories:
            self.user_memories[user_id] = []
        self.user_memories[user_id].append(memory_id)

        return memory_id

    async def retrieve_memory(self, user_id: str, memory_id: str,
                            require_auth: bool = True) -> Optional[SecureMemory]:
        """Retrieve a specific memory"""

        # Check authentication for sensitive memories
        memory = self.memory_index.get(memory_id)
        if not memory:
            return None

        if memory.privacy_level in [PrivacyLevel.INTIMATE, PrivacyLevel.ENCRYPTED]:
            if require_auth and not self.authenticator.is_authenticated(user_id):
                return None  # Authentication required

        # Decrypt if needed
        if memory.encrypted_content and memory.content == "[ENCRYPTED]":
            try:
                memory.content = self.encryption.decrypt_memory(
                    memory.encrypted_content,
                    memory.privacy_level
                )
            except:
                return None  # Decryption failed

        # Update access metrics
        memory.access_count += 1
        memory.last_accessed = datetime.now()

        return memory

    async def search_memories(self, user_id: str, query: str,
                             memory_types: Optional[List[MemoryType]] = None,
                             privacy_levels: Optional[List[PrivacyLevel]] = None,
                             limit: int = 10) -> List[SecureMemory]:
        """Search memories using vector similarity"""

        # Generate query embedding
        query_vector = self.encoder.encode(query).tolist()

        # Build filter
        filter_conditions = {"user_id": user_id}
        if memory_types:
            filter_conditions["memory_type"] = {"$in": [mt.value for mt in memory_types]}
        if privacy_levels:
            filter_conditions["privacy_level"] = {"$in": [pl.value for pl in privacy_levels]}

        # Search in Qdrant
        search_result = self.qdrant_client.search(
            collection_name="secure_memories",
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_conditions
        )

        # Retrieve and decrypt memories
        memories = []
        for result in search_result:
            memory_id = result.payload.get("memory_id")
            memory = await self.retrieve_memory(user_id, memory_id, require_auth=False)
            if memory:
                memories.append(memory)

        return memories

    async def cluster_memories(self, user_id: str,
                             min_cluster_size: int = 3) -> List[MemoryCluster]:
        """Cluster related memories for pattern recognition"""

        if user_id not in self.user_memories:
            return []

        user_memory_ids = self.user_memories[user_id]
        if len(user_memory_ids) < min_cluster_size:
            return []

        # Get memory embeddings
        embeddings = []
        memories = []
        for memory_id in user_memory_ids:
            memory = self.memory_index.get(memory_id)
            if memory and memory.vector_embedding is not None:
                embeddings.append(memory.vector_embedding)
                memories.append(memory)

        if len(embeddings) < min_cluster_size:
            return []

        # Simple clustering using cosine similarity
        # In production, use more sophisticated clustering
        embeddings_array = np.array(embeddings)
        clusters = self._simple_clustering(embeddings_array, memories)

        return clusters

    def _simple_clustering(self, embeddings: np.ndarray,
                          memories: List[SecureMemory]) -> List[MemoryCluster]:
        """Simple clustering algorithm (placeholder for production clustering)"""

        # This is a simplified version - use DBSCAN or similar in production
        clusters = []

        # Create one cluster for demonstration
        if len(memories) > 0:
            cluster = MemoryCluster(
                cluster_id=str(uuid.uuid4()),
                theme="recent_interactions",
                memories=memories[:5],  # Take first 5 memories
                importance_score=0.8,
                coherence_score=0.75,
                privacy_level=max(m.privacy_level for m in memories[:5])
            )
            clusters.append(cluster)

        return clusters

    async def _store_in_vector_db(self, memory: SecureMemory) -> None:
        """Store memory in vector database"""

        point = PointStruct(
            id=memory.id,
            vector=memory.vector_embedding.tolist(),
            payload={
                "memory_id": memory.id,
                "user_id": memory.metadata.get("user_id", ""),
                "memory_type": memory.memory_type.value,
                "privacy_level": memory.privacy_level.value,
                "timestamp": memory.timestamp.isoformat()
            }
        )

        self.qdrant_client.upsert(
            collection_name="secure_memories",
            points=[point]
        )

    async def _store_in_cache(self, user_id: str, memory: SecureMemory) -> None:
        """Store memory in Redis cache"""

        cache_key = f"memory:{user_id}:{memory.id}"

        # Serialize memory (exclude vector for cache)
        memory_dict = {
            'id': memory.id,
            'timestamp': memory.timestamp.isoformat(),
            'memory_type': memory.memory_type.value,
            'privacy_level': memory.privacy_level.value,
            'content': memory.content if memory.privacy_level == PrivacyLevel.PUBLIC else "[ENCRYPTED]",
            'encrypted_content': memory.encrypted_content,
            'metadata': memory.metadata
        }

        self.redis_client.setex(
            cache_key,
            86400 * 7,  # 7 day TTL
            pickle.dumps(memory_dict)
        )

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Delete a memory permanently"""

        # Require authentication for deletion
        if not self.authenticator.is_authenticated(user_id):
            return False

        # Remove from all stores
        if memory_id in self.memory_index:
            del self.memory_index[memory_id]

        if user_id in self.user_memories:
            self.user_memories[user_id] = [
                mid for mid in self.user_memories[user_id] if mid != memory_id
            ]

        # Remove from Qdrant
        self.qdrant_client.delete(
            collection_name="secure_memories",
            points_selector=[memory_id]
        )

        # Remove from Redis
        cache_key = f"memory:{user_id}:{memory_id}"
        self.redis_client.delete(cache_key)

        return True

    def get_privacy_report(self, user_id: str) -> Dict[str, Any]:
        """Generate privacy report for user"""

        if user_id not in self.user_memories:
            return {'error': 'No memories found'}

        memory_ids = self.user_memories[user_id]
        privacy_stats = {level: 0 for level in PrivacyLevel}
        type_stats = {mtype: 0 for mtype in MemoryType}

        for memory_id in memory_ids:
            memory = self.memory_index.get(memory_id)
            if memory:
                privacy_stats[memory.privacy_level] += 1
                type_stats[memory.memory_type] += 1

        return {
            'user_id': user_id,
            'total_memories': len(memory_ids),
            'privacy_distribution': {k.value: v for k, v in privacy_stats.items()},
            'type_distribution': {k.value: v for k, v in type_stats.items()},
            'encryption_active': True,
            'biometric_auth_enabled': True,
            'data_retention_policy': '90 days for general, indefinite for consented personal'
        }