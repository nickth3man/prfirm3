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
        """
        Initialize the LRU in-memory cache.
        
        Parameters:
            max_size (int): Maximum number of entries the cache will hold before evicting least-recently-used items.
            default_ttl (float): Default time-to-live for entries in seconds when no per-entry TTL is provided.
        
        Notes:
            - Creates internal structures (OrderedDict) and a reentrant lock for thread-safe access.
            - Starts a background cleanup thread that removes expired entries while the cache is running.
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats(max_size=max_size)
        self.lock = threading.RLock()
        self.cleanup_thread = None
        self.running = True
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """
        Start a background daemon thread that periodically removes expired cache entries.
        
        The thread runs until the instance's `self.running` flag is cleared. Every 60 seconds it calls `self._cleanup_expired()` to purge expired entries. Exceptions raised during cleanup are caught and logged; they do not stop the thread. The thread object is stored on `self.cleanup_thread`.
        """
        def cleanup_worker():
            """
            Background worker loop that periodically triggers cache expiration cleanup.
            
            Runs while the cache's `running` flag is True: sleeps for 60 seconds, then calls
            the instance method `_cleanup_expired()` to remove expired entries. Exceptions
            raised by the cleanup call are caught and logged; the worker continues running
            until `running` is cleared.
            """
            while self.running:
                try:
                    time.sleep(60)  # Run every minute
                    self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_expired(self):
        """
        Remove entries whose TTL has expired from the in-memory cache.
        
        This method acquires the cache lock, iterates existing entries, and deletes any
        whose age (current time minus entry.timestamp) exceeds their entry.ttl. For each
        deleted entry the cache statistics counter `stats.deletes` is incremented. If at
        least one entry is removed, a debug log is emitted.
        
        Intended for internal use by the periodic cleanup worker.
        """
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
        """
        Retrieve a value from the in-memory LRU cache for the given key.
        
        If the key is present and not expired this updates the entry's access metadata (access_count, last_accessed),
        moves the entry to the most-recently-used position, and updates cache statistics. If the key is absent or expired
        it is treated as a miss (expired entries are removed and delete statistics incremented).
        
        Parameters:
            key: Cache key string.
        
        Returns:
            The cached value if present and not expired, otherwise None.
        """
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
        """
        Store a value in the in-memory LRU cache.
        
        If ttl is None the cache's default_ttl is used. If the key already exists it is replaced (moved to most-recently-used). When the cache is at capacity the oldest entry is evicted to make room. This operation is thread-safe and updates cache statistics.
        
        Parameters:
            key: Cache key.
            value: Value to store under the key.
            ttl: Time-to-live in seconds for this entry; if omitted the cache default is applied.
        """
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
        """
        Remove a key from the in-memory cache if present.
        
        This is a thread-safe deletion that, when the key exists, removes the entry,
        increments the delete counter, updates the recorded cache size and derived stats.
        
        Parameters:
            key (str): Cache key to remove.
        
        Returns:
            bool: True if the key was present and deleted, False if the key was not found.
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.deletes += 1
                self.stats.size = len(self.cache)
                self._update_stats()
                return True
            return False
    
    def clear(self) -> None:
        """
        Clear all entries from the in-memory cache and reset related statistics.
        
        This method is thread-safe: it removes every CacheEntry, sets the recorded size to zero,
        and updates aggregate statistics (e.g., hit rate) to reflect the cleared state.
        """
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
        """
        Recalculate and update the cache hit rate.
        
        Computes hit_rate as hits / (hits + misses) and updates the stats object.
        If there are no recorded requests (hits + misses == 0), the hit_rate is left unchanged.
        """
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests
    
    def shutdown(self) -> None:
        """
        Stop the cache's background cleanup thread and wait briefly for it to finish.
        
        Sets the internal running flag to False to signal the cleanup thread to exit, then joins the thread (waiting up to 5 seconds) if it exists. This method does not raise on timeout; the thread may still be running after return if it failed to stop within the join timeout.
        """
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)


class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, redis_url: str, default_ttl: float = 3600):
        """
        Initialize the Redis-backed cache.
        
        Attempts to import the `redis` package and create a client using `redis.from_url(redis_url)`.
        On success stores the client, sets the default TTL for entries, and initializes cache statistics.
        
        Parameters:
            redis_url (str): Redis connection URL accepted by `redis.from_url` (e.g. "redis://[:password]@host:port/db").
            default_ttl (float): Default time-to-live for cache entries in seconds (defaults to 3600).
        
        Raises:
            ImportError: If the `redis` package is not installed.
            Exception: Propagates other exceptions raised while constructing the Redis client.
        """
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
        """
        Retrieve and return the Python object stored under `key` in Redis, or None if the key is not present or an error occurs.
        
        If present, the stored bytes are unpickled and returned. This method updates Redis cache statistics (hits, misses, and average access time in milliseconds). Errors during retrieval or deserialization are logged and counted as a miss, and the method returns None.
        """
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
        """
        Store a Python object in Redis under the given key with an expiration.
        
        The value is serialized with pickle and written to Redis using SETEX with a TTL (in seconds).
        If ttl is None the cache's configured default_ttl is used. On success the cache's set counter
        is incremented; failures are logged and not raised.
        """
        try:
            ttl = ttl or self.default_ttl
            data = pickle.dumps(value)
            self.redis_client.setex(key, int(ttl), data)
            self.stats.sets += 1
            self._update_stats()
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the Redis cache.
        
        Parameters:
            key (str): Redis key to remove.
        
        Returns:
            bool: True if Redis reported the key was deleted, False if the key did not exist or an error occurred.
        
        Notes:
            On success increments the cache delete counter and updates aggregated stats. Exceptions are logged and result in False.
        """
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
        """
        Clear all entries in the connected Redis database and reset internal statistics.
        
        This performs a full database flush (Redis FLUSHDB) on the configured client, sets the cached size counter to zero, and refreshes hit/miss statistics. This method swallows and logs exceptions raised by the Redis client rather than propagating them.
        
        Warning: This clears every key in the selected Redis database. In production, prefer using a namespaced key prefix and deleting only matching keys to avoid removing unrelated data.
        """
        try:
            # Note: This clears ALL keys in the Redis database
            # In production, use a specific prefix for this application
            self.redis_client.flushdb()
            self.stats.size = 0
            self._update_stats()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def get_stats(self) -> CacheStats:
        """
        Return a snapshot of Redis-backed cache statistics.
        
        Attempts to update the reported `size` from Redis INFO (reads `db0.keys`) and then returns a CacheStats
        dataclass with hits, misses, sets, deletes, size, max_size, hit_rate, and avg_access_time.
        Note: `max_size` is set to 0 because Redis does not expose a fixed in-process capacity. Errors
        while reading Redis INFO are logged and do not raise.
        """
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
        """
        Recalculate and update the cache hit rate.
        
        Computes hit_rate as hits / (hits + misses) and updates the stats object.
        If there are no recorded requests (hits + misses == 0), the hit_rate is left unchanged.
        """
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests


class MultiTierCache:
    """Multi-tier cache implementation with fallback."""
    
    def __init__(self, memory_cache: LRUCache, redis_cache: Optional[RedisCache] = None):
        """
        Initialize the multi-tier cache coordinator.
        
        Creates a MultiTierCache that routes reads/writes to an in-memory LRUCache and optionally a Redis-backed cache. Initializes an empty CacheStats object used to aggregate per-tier statistics.
        """
        self.memory_cache = memory_cache
        self.redis_cache = redis_cache
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value for `key` from the multi-tier cache.
        
        Checks the in-memory cache first; on a miss and if Redis is enabled, attempts Redis.
        If Redis returns a value it is promoted into the in-memory cache for faster subsequent access.
        
        Returns:
            The cached value if present in either tier, otherwise None.
        """
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
        """
        Set a value in all configured cache tiers.
        
        Stores `value` under `key` in the in-memory cache and, if a Redis tier is configured, in Redis as well.
        
        Parameters:
            key (str): Cache key.
            value (Any): Value to store.
            ttl (float | None): Time-to-live in seconds for this entry. If None, each tier's default TTL is used.
        """
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
        """
        Return a snapshot of cache statistics for each tier.
        
        The returned dictionary contains per-tier CacheStats objects and an aggregated
        'combined' entry. Keys:
        - 'memory': statistics for the in-memory LRU cache.
        - 'redis' (optional): statistics for the Redis cache, present only if Redis is enabled.
        - 'combined': aggregated CacheStats maintained by the multi-tier cache.
        
        Returns:
            Dict[str, CacheStats]: Mapping of tier name to its CacheStats snapshot.
        """
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
        """
        Generate a deterministic cache key for a flow based on the provided shared state.
        
        This function extracts the platforms, topic/goal, and brand bible from the `shared` mapping, normalizes them (platforms are sorted; topic is trimmed and lowercased), and builds a compact representation consisting of:
        - sorted platforms,
        - an 8-character MD5 digest of the topic,
        - an 8-character MD5 digest of the brand bible JSON (sorted keys).
        
        It returns a 32-character MD5 hex string computed over the JSON-encoded representation of those three fields.
        
        Parameters:
            shared (Dict[str, Any]): Flow shared state; expected to include:
                - 'task_requirements' (optional): mapping that may contain 'platforms' (iterable of platform identifiers)
                  and 'topic_or_goal' (string).
                - 'brand_bible' (optional): mapping describing brand settings; it will be JSON-serialized with sorted keys.
        
        Returns:
            str: A 32-character MD5 hex string representing the cache key.
        """
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
        """
        Generate a deterministic similarity key for grouping cache entries by topic and target platforms.
        
        This returns a stable 32-character MD5 hex string derived from a normalized representation of
        the provided topic and platforms. The topic is stripped and lowercased and the platforms list
        is sorted before encoding, so equivalent inputs that differ only by case, surrounding whitespace,
        or platform order produce the same key.
        
        Parameters:
            topic (str): The topic/goal string to normalize and include in the key.
            platforms (List[str]): A list of platform identifiers; order does not affect the result.
        
        Returns:
            str: A 32-character MD5 hexadecimal hash representing the normalized (topic, platforms) pair.
        """
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
        """
        Initialize the CacheManager.
        
        Parameters:
            config (dict): Configuration mapping. Recognized keys:
                - 'enable_caching' (bool): Whether caching is enabled (default True).
                - 'cache_ttl' (int|float): Default time-to-live for entries in seconds (default 3600).
                - 'cache_size' (int): Maximum entries for in-memory LRU cache (default 1000).
                - 'redis_url' (str, optional): URL to initialize a Redis-backed cache; if provided, the manager will attempt to create a RedisCache and log a warning on failure.
        
        Behavior:
            - Creates a thread-safe in-memory LRUCache using the configured size and TTL.
            - Optionally attempts to initialize a RedisCache when 'redis_url' is provided; failure to connect is logged and Redis is disabled.
            - Constructs a MultiTierCache that coordinates memory and optional Redis tiers.
            - Builds a CacheKeyGenerator and an empty similarity_cache mapping used for group invalidation.
        
        Side effects:
            - May log a warning if Redis initialization fails.
        """
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
        """
        Retrieve a cached flow result for the given shared flow state.
        
        Given the flow's shared state (the structure produced/used by the flow runner), this method generates the cache key for that state and returns the cached result if present. If caching is disabled, the entry is not found, or an internal error occurs, None is returned.
        
        Parameters:
            shared (Dict[str, Any]): Flow's shared state used to derive the cache key (includes platforms, topic/goal, brand bible, etc.).
        
        Returns:
            Optional[Dict[str, Any]]: The cached flow result dictionary when available; otherwise None.
        """
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
        """
        Cache the flow result using the configured multi-tier cache and record it for similarity-based invalidation.
        
        Generates a cache key from the provided `shared` flow state and stores `result` in the multi-tier cache with the manager's default TTL. If caching is disabled this is a no-op. Also updates the internal similarity index (mapping from a topic+platforms similarity key to a list of cache keys) so related entries can be invalidated together. Errors during caching are caught and logged but not raised.
        
        Parameters:
            shared (Dict[str, Any]): Flow request state used to derive the cache key (expects keys like 'task_requirements' containing 'topic_or_goal' and 'platforms').
            result (Dict[str, Any]): The flow result payload to be stored in the cache.
        """
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
        """
        Invalidate cache entries associated with topics similar to the given topic and platforms.
        
        Generates a similarity key from `topic` and `platforms`, looks up all cache keys grouped under that similarity key, deletes each key from the multi-tier cache, and removes the similarity mapping. If caching is disabled or an error occurs, no entries are invalidated.
        
        Parameters:
            topic (str): The topic/goal string used to compute the similarity group.
            platforms (List[str]): List of platform identifiers used when computing the similarity group.
        
        Returns:
            int: The number of cache entries successfully invalidated (0 if caching is disabled or on error).
        """
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
        """
        Return a snapshot of the cache manager's status and metrics.
        
        When caching is disabled this returns {'enabled': False}. When enabled the returned dictionary contains:
        - 'enabled' (bool): True.
        - 'tiers' (dict): Per-tier statistics returned by the underlying multi-tier cache.
        - 'similarity_cache_size' (int): Number of entries tracked for similarity-based invalidation.
        - 'config' (dict): Active cache configuration with keys:
            - 'default_ttl' (float): Default time-to-live for entries.
            - 'memory_size' (int): Configured memory cache capacity.
            - 'redis_enabled' (bool): True if a Redis backend is configured.
        
        Returns:
            Dict[str, Any]: Structured cache status and metrics as described above.
        """
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
        """
        Clear all caches managed by this CacheManager.
        
        If caching is enabled, clears the multi-tier cache (memory and optional Redis) and removes all entries from the similarity invalidation mapping. If caching is disabled, this method is a no-op.
        """
        if self.enabled:
            self.multi_tier_cache.clear()
            self.similarity_cache.clear()
            logger.info("All caches cleared")
    
    def shutdown(self) -> None:
        """
        Stop the cache manager and release in-process cache resources.
        
        If caching is enabled, shuts down the in-memory cache (stops its background cleanup thread and cleans up related resources) and logs completion. This does not explicitly flush or close external Redis connections.
        """
        if self.enabled:
            self.memory_cache.shutdown()
            logger.info("Cache manager shutdown complete")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def initialize_cache(config: Dict[str, Any]) -> None:
    """
    Create and assign the module-level CacheManager singleton from the given configuration.
    
    The provided `config` is passed to CacheManager(...) and the resulting manager is stored in the module-global `_cache_manager`, replacing any existing instance. This initializes in-memory and optional Redis tiers according to the configuration and prepares the caching subsystem for use.
    Parameters:
        config (Dict[str, Any]): Configuration dictionary used to construct the CacheManager (e.g., enable_caching, cache_ttl, cache_size, redis_url).
    """
    global _cache_manager
    _cache_manager = CacheManager(config)
    logger.info("Cache manager initialized")


def get_cache_manager() -> Optional[CacheManager]:
    """Get the global cache manager instance."""
    return _cache_manager


def cache_result(ttl: Optional[float] = None):
    """
    Decorator factory that caches a decorated function's return value in the global multi-tier cache.
    
    If a CacheManager is initialized and enabled, the wrapper will:
    - Build a cache key from the function name and its arguments (JSON-serialized with sensible defaults and stable key ordering).
    - Return a cached result when present.
    - On cache miss, call the function, store the result in the multi-tier cache, and return it.
    
    If no cache manager is available or caching is disabled, the wrapped function is executed normally.
    
    Parameters:
        ttl (Optional[float]): Time-to-live for the cached entry in seconds. If omitted, the cache manager's default_ttl is used.
    
    Returns:
        Callable: A decorator that caches the wrapped function's results.
    """
    def decorator(func: Callable) -> Callable:
        """
        Wraps a function to cache its return value in the global multi-tier cache.
        
        The wrapper builds a cache key from the function name and the passed positional and keyword arguments (JSON-serialized, with non-serializable objects converted via str), then attempts to return a cached value from the global cache manager's multi-tier cache. If no cache manager exists or caching is disabled, the wrapped function is executed normally. On a cache miss the function is executed and its result is stored in the multi-tier cache using the provided `ttl` from the enclosing scope (or the cache manager's `default_ttl` when `ttl` is falsy).
        
        Returns:
            Callable: The wrapped function that transparently uses the cache.
        """
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
    """
    Retrieve a cached result for a flow or return information needed to cache the result.
    
    If a global CacheManager is not initialized or caching is disabled, returns None.
    If a cached value exists for the provided shared state, returns that cached value directly.
    Otherwise returns a tuple (cache_manager, ttl) where `cache_manager` is the active CacheManager
    and `ttl` is the time-to-live (seconds) to use when storing the result (uses the manager's
    default_ttl if `ttl` is None).
    
    Parameters:
        shared (Dict[str, Any]): Shared flow state used to generate the cache key (must contain
            the fields the CacheKeyGenerator expects, e.g., platforms/topic/brand data).
        ttl (Optional[float]): Optional TTL (in seconds) to use when caching a new result. If None,
            the CacheManager's default TTL will be used when caching.
    
    Returns:
        Optional[Union[Any, Tuple[CacheManager, float]]]:
            - None if no cache manager is available or caching is disabled.
            - A cached result (arbitrary type) if a hit occurred.
            - A (cache_manager, ttl) tuple to indicate where and how to cache the result on miss.
    """
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
    """
    Context manager that returns a cached flow result (if available) or allows the flow to run and then caches its output.
    
    This generator-based context manager checks the global cache manager for a cached value for the given shared state. If a cached result exists it is yielded to the caller and the flow body should not be executed. If no cached result exists, the context yields None so the caller can run the flow; on exit, if the flow stored output in shared['content_pieces'], that value will be cached using the provided ttl (or the cache manager's default_ttl).
    
    Parameters:
        shared (Dict[str, Any]): Flow state used to generate the cache key. The function expects the flow to place result data into shared['content_pieces'] when caching a new result.
        ttl (Optional[float]): Time-to-live in seconds for a newly cached result. If omitted, the cache manager's default_ttl is used.
    
    Yields:
        The cached result if found, otherwise None.
    
    Side effects:
        May store shared['content_pieces'] in the cache on exit when no cached value was present.
    
    Behavior notes:
        - If no global cache manager is initialized or caching is disabled, the context yields None and no caching occurs.
    """
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