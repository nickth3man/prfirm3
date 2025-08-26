"""Intelligent caching mechanism for the Virtual PR Firm application.

This module provides multi-tier caching with intelligent invalidation,
performance monitoring, and support for both in-memory and Redis backends.
"""

import hashlib
import json
import time
import pickle
import threading
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from collections import OrderedDict
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    size: int = 0
    max_size: int = 0
    hit_rate: float = 0.0
    avg_access_time: float = 0.0


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats(max_size=max_size)
        self.lock = threading.RLock()
        self.cleanup_thread = None
        self.running = True
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while self.running:
                try:
                    time.sleep(60)  # Run every minute
                    self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self.cache.items():
                if current_time - entry.timestamp > entry.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                self.stats.deletes += 1
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            entry = self.cache[key]
            current_time = time.time()
            
            # Check if expired
            if current_time - entry.timestamp > entry.ttl:
                del self.cache[key]
                self.stats.misses += 1
                self.stats.deletes += 1
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_accessed = current_time
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            
            self.stats.hits += 1
            self._update_stats()
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self.lock:
            ttl = ttl or self.default_ttl
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )
            
            # Remove if key already exists
            if key in self.cache:
                del self.cache[key]
            
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats.deletes += 1
            
            self.cache[key] = entry
            self.stats.sets += 1
            self.stats.size = len(self.cache)
            self._update_stats()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.deletes += 1
                self.stats.size = len(self.cache)
                self._update_stats()
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        with self.lock:
            self.cache.clear()
            self.stats.size = 0
            self._update_stats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self.lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                sets=self.stats.sets,
                deletes=self.stats.deletes,
                size=self.stats.size,
                max_size=self.stats.max_size,
                hit_rate=self.stats.hit_rate,
                avg_access_time=self.stats.avg_access_time
            )
    
    def _update_stats(self) -> None:
        """Update cache statistics."""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests
    
    def shutdown(self) -> None:
        """Shutdown the cache and cleanup thread."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)


class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, redis_url: str, default_ttl: float = 3600):
        try:
            import redis
            self.redis_client = redis.from_url(redis_url)
            self.default_ttl = default_ttl
            self.stats = CacheStats()
            logger.info(f"Redis cache initialized with URL: {redis_url}")
        except ImportError:
            logger.error("Redis not installed. Install with: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            start_time = time.time()
            data = self.redis_client.get(key)
            access_time = (time.time() - start_time) * 1000
            
            if data is None:
                self.stats.misses += 1
                return None
            
            # Update access time
            self.stats.avg_access_time = (
                (self.stats.avg_access_time * (self.stats.hits + self.stats.misses - 1) + access_time) /
                (self.stats.hits + self.stats.misses)
            )
            
            self.stats.hits += 1
            self._update_stats()
            
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in Redis cache."""
        try:
            ttl = ttl or self.default_ttl
            data = pickle.dumps(value)
            self.redis_client.setex(key, int(ttl), data)
            self.stats.sets += 1
            self._update_stats()
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        try:
            result = self.redis_client.delete(key)
            if result:
                self.stats.deletes += 1
                self._update_stats()
            return bool(result)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all entries from Redis cache."""
        try:
            # Note: This clears ALL keys in the Redis database
            # In production, use a specific prefix for this application
            self.redis_client.flushdb()
            self.stats.size = 0
            self._update_stats()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        try:
            info = self.redis_client.info()
            self.stats.size = info.get('db0', {}).get('keys', 0)
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
        
        return CacheStats(
            hits=self.stats.hits,
            misses=self.stats.misses,
            sets=self.stats.sets,
            deletes=self.stats.deletes,
            size=self.stats.size,
            max_size=0,  # Redis doesn't have a fixed max size
            hit_rate=self.stats.hit_rate,
            avg_access_time=self.stats.avg_access_time
        )
    
    def _update_stats(self) -> None:
        """Update cache statistics."""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests


class MultiTierCache:
    """Multi-tier cache implementation with fallback."""
    
    def __init__(self, memory_cache: LRUCache, redis_cache: Optional[RedisCache] = None):
        self.memory_cache = memory_cache
        self.redis_cache = redis_cache
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache tiers."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache if available
        if self.redis_cache:
            value = self.redis_cache.get(key)
            if value is not None:
                # Cache in memory for future access
                self.memory_cache.set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in all cache tiers."""
        # Set in memory cache
        self.memory_cache.set(key, value, ttl)
        
        # Set in Redis cache if available
        if self.redis_cache:
            self.redis_cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from all cache tiers."""
        memory_result = self.memory_cache.delete(key)
        redis_result = False
        
        if self.redis_cache:
            redis_result = self.redis_cache.delete(key)
        
        return memory_result or redis_result
    
    def clear(self) -> None:
        """Clear all cache tiers."""
        self.memory_cache.clear()
        if self.redis_cache:
            self.redis_cache.clear()
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics from all cache tiers."""
        stats = {
            'memory': self.memory_cache.get_stats(),
            'combined': self.stats
        }
        
        if self.redis_cache:
            stats['redis'] = self.redis_cache.get_stats()
        
        return stats


class CacheKeyGenerator:
    """Generates cache keys for flow requests."""
    
    @staticmethod
    def generate_key(shared: Dict[str, Any]) -> str:
        """Generate a cache key from shared store data."""
        # Extract relevant data
        task_req = shared.get('task_requirements', {})
        platforms = sorted(task_req.get('platforms', []))
        topic = task_req.get('topic_or_goal', '').strip().lower()
        
        # Create brand bible hash
        brand_bible = shared.get('brand_bible', {})
        brand_bible_str = json.dumps(brand_bible, sort_keys=True)
        brand_bible_hash = hashlib.md5(brand_bible_str.encode()).hexdigest()[:8]
        
        # Create topic hash
        topic_hash = hashlib.md5(topic.encode()).hexdigest()[:8]
        
        # Combine into cache key
        key_data = {
            'platforms': platforms,
            'topic_hash': topic_hash,
            'brand_bible_hash': brand_bible_hash
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @staticmethod
    def generate_similarity_key(topic: str, platforms: List[str]) -> str:
        """Generate a key for similarity-based cache invalidation."""
        normalized_topic = topic.strip().lower()
        sorted_platforms = sorted(platforms)
        
        key_data = {
            'topic': normalized_topic,
            'platforms': sorted_platforms
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()


class CacheManager:
    """Manages the application's caching system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enable_caching', True)
        self.default_ttl = config.get('cache_ttl', 3600)
        self.memory_size = config.get('cache_size', 1000)
        
        # Initialize cache tiers
        self.memory_cache = LRUCache(
            max_size=self.memory_size,
            default_ttl=self.default_ttl
        )
        
        self.redis_cache = None
        redis_url = config.get('redis_url')
        if redis_url:
            try:
                self.redis_cache = RedisCache(redis_url, self.default_ttl)
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
        
        self.multi_tier_cache = MultiTierCache(self.memory_cache, self.redis_cache)
        self.key_generator = CacheKeyGenerator()
        
        # Similarity cache for invalidation
        self.similarity_cache: Dict[str, List[str]] = {}
    
    def get(self, shared: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached result for a flow request."""
        if not self.enabled:
            return None
        
        try:
            key = self.key_generator.generate_key(shared)
            result = self.multi_tier_cache.get(key)
            
            if result is not None:
                logger.info(f"Cache hit for key: {key[:8]}...")
                return result
            
            logger.debug(f"Cache miss for key: {key[:8]}...")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, shared: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Cache result for a flow request."""
        if not self.enabled:
            return
        
        try:
            key = self.key_generator.generate_key(shared)
            self.multi_tier_cache.set(key, result, self.default_ttl)
            
            # Update similarity cache
            task_req = shared.get('task_requirements', {})
            topic = task_req.get('topic_or_goal', '')
            platforms = task_req.get('platforms', [])
            similarity_key = self.key_generator.generate_similarity_key(topic, platforms)
            
            if similarity_key not in self.similarity_cache:
                self.similarity_cache[similarity_key] = []
            self.similarity_cache[similarity_key].append(key)
            
            logger.debug(f"Cached result for key: {key[:8]}...")
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def invalidate_similar(self, topic: str, platforms: List[str]) -> int:
        """Invalidate cache entries for similar topics."""
        if not self.enabled:
            return 0
        
        try:
            similarity_key = self.key_generator.generate_similarity_key(topic, platforms)
            keys_to_invalidate = self.similarity_cache.get(similarity_key, [])
            
            invalidated_count = 0
            for key in keys_to_invalidate:
                if self.multi_tier_cache.delete(key):
                    invalidated_count += 1
            
            # Remove from similarity cache
            if similarity_key in self.similarity_cache:
                del self.similarity_cache[similarity_key]
            
            logger.info(f"Invalidated {invalidated_count} similar cache entries")
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self.enabled:
            return {'enabled': False}
        
        stats = self.multi_tier_cache.get_stats()
        
        return {
            'enabled': True,
            'tiers': stats,
            'similarity_cache_size': len(self.similarity_cache),
            'config': {
                'default_ttl': self.default_ttl,
                'memory_size': self.memory_size,
                'redis_enabled': self.redis_cache is not None
            }
        }
    
    def clear(self) -> None:
        """Clear all cache tiers."""
        if self.enabled:
            self.multi_tier_cache.clear()
            self.similarity_cache.clear()
            logger.info("All caches cleared")
    
    def shutdown(self) -> None:
        """Shutdown the cache manager."""
        if self.enabled:
            self.memory_cache.shutdown()
            logger.info("Cache manager shutdown complete")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def initialize_cache(config: Dict[str, Any]) -> None:
    """Initialize the global cache manager."""
    global _cache_manager
    _cache_manager = CacheManager(config)
    logger.info("Cache manager initialized")


def get_cache_manager() -> Optional[CacheManager]:
    """Get the global cache manager instance."""
    return _cache_manager


def cache_result(ttl: Optional[float] = None):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            if not cache_manager or not cache_manager.enabled:
                return func(*args, **kwargs)
            
            # Generate cache key from function name and arguments
            key_data = {
                'function': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            key = hashlib.md5(json.dumps(key_data, default=str, sort_keys=True).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = cache_manager.multi_tier_cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.multi_tier_cache.set(key, result, ttl or cache_manager.default_ttl)
            
            return result
        
        return wrapper
    return decorator


def with_cache(shared: Dict[str, Any], ttl: Optional[float] = None):
    """Context manager for caching flow results."""
    cache_manager = get_cache_manager()
    
    if not cache_manager or not cache_manager.enabled:
        return None
    
    # Try to get from cache
    cached_result = cache_manager.get(shared)
    if cached_result is not None:
        return cached_result
    
    return cache_manager, ttl or cache_manager.default_ttl


@contextmanager
def cache_context(shared: Dict[str, Any], ttl: Optional[float] = None):
    """Context manager for caching flow results."""
    cache_manager = get_cache_manager()
    
    if not cache_manager or not cache_manager.enabled:
        yield None
        return
    
    # Try to get from cache
    cached_result = cache_manager.get(shared)
    if cached_result is not None:
        yield cached_result
        return
    
    # Execute flow and cache result
    result = None
    try:
        yield None  # Let the caller execute the flow
    finally:
        # The result should be in shared['content_pieces']
        if 'content_pieces' in shared:
            cache_manager.set(shared, shared['content_pieces'], ttl or cache_manager.default_ttl)