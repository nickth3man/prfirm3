"""Caching mechanism for the Virtual PR Firm.

This module provides a comprehensive caching system for the Virtual PR Firm
application, supporting both in-memory and persistent caching with TTL support.
"""

import hashlib
import json
import time
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
from functools import wraps
import logging
from threading import Lock

logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a cache entry with metadata."""
    
    def __init__(self, value: Any, ttl: int = 3600, created_at: Optional[float] = None):
        self.value = value
        self.ttl = ttl
        self.created_at = created_at or time.time()
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.created_at > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary for serialization."""
        return {
            "value": self.value,
            "ttl": self.ttl,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create cache entry from dictionary."""
        return cls(
            value=data["value"],
            ttl=data["ttl"],
            created_at=data["created_at"]
        )


class MemoryCache:
    """In-memory cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in cache."""
        with self._lock:
            # Remove expired entries
            self._cleanup()
            
            # Check if we need to evict entries
            if len(self._cache) >= self._max_size:
                self._evict_oldest()
            
            self._cache[key] = CacheEntry(value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            return self._cache.pop(key, None) is not None
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def _cleanup(self) -> None:
        """Remove expired entries."""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self._cache[key]
    
    def _evict_oldest(self) -> None:
        """Evict the oldest entry when cache is full."""
        if not self._cache:
            return
        
        oldest_key = min(self._cache.keys(), 
                        key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            self._cleanup()
            return len(self._cache)
    
    def keys(self) -> list:
        """Get all cache keys."""
        with self._lock:
            self._cleanup()
            return list(self._cache.keys())


class FileCache:
    """File-based persistent cache implementation."""
    
    def __init__(self, cache_dir: Union[str, Path] = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Create a safe filename from the key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with self._lock:
                with open(cache_path, 'rb') as f:
                    entry_data = pickle.load(f)
                    entry = CacheEntry.from_dict(entry_data)
            
            if entry.is_expired():
                cache_path.unlink(missing_ok=True)
                return None
            
            return entry.value
        except Exception as e:
            logger.warning(f"Error reading cache file {cache_path}: {e}")
            cache_path.unlink(missing_ok=True)
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in file cache."""
        cache_path = self._get_cache_path(key)
        entry = CacheEntry(value, ttl)
        
        try:
            with self._lock:
                with open(cache_path, 'wb') as f:
                    pickle.dump(entry.to_dict(), f)
        except Exception as e:
            logger.error(f"Error writing cache file {cache_path}: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete value from file cache."""
        cache_path = self._get_cache_path(key)
        try:
            with self._lock:
                if cache_path.exists():
                    cache_path.unlink()
                    return True
                return False
        except Exception as e:
            logger.warning(f"Error deleting cache file {cache_path}: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache files."""
        try:
            with self._lock:
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error clearing cache directory: {e}")
    
    def cleanup_expired(self) -> int:
        """Remove expired cache files and return count of removed files."""
        removed_count = 0
        try:
            with self._lock:
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        with open(cache_file, 'rb') as f:
                            entry_data = pickle.load(f)
                            entry = CacheEntry.from_dict(entry_data)
                        
                        if entry.is_expired():
                            cache_file.unlink()
                            removed_count += 1
                    except Exception:
                        # If we can't read the file, delete it
                        cache_file.unlink(missing_ok=True)
                        removed_count += 1
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
        
        return removed_count


class CacheManager:
    """Main cache manager that coordinates different cache backends."""
    
    def __init__(self, memory_cache_size: int = 1000, 
                 file_cache_dir: Union[str, Path] = "./cache",
                 enable_memory_cache: bool = True,
                 enable_file_cache: bool = True):
        self.memory_cache = MemoryCache(memory_cache_size) if enable_memory_cache else None
        self.file_cache = FileCache(file_cache_dir) if enable_file_cache else None
        self._lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, trying memory first, then file."""
        # Try memory cache first
        if self.memory_cache:
            value = self.memory_cache.get(key)
            if value is not None:
                return value
        
        # Try file cache
        if self.file_cache:
            value = self.file_cache.get(key)
            if value is not None:
                # Store in memory cache for faster access
                if self.memory_cache:
                    self.memory_cache.set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in both memory and file cache."""
        if self.memory_cache:
            self.memory_cache.set(key, value, ttl)
        
        if self.file_cache:
            self.file_cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete value from both caches."""
        memory_deleted = self.memory_cache.delete(key) if self.memory_cache else False
        file_deleted = self.file_cache.delete(key) if self.file_cache else False
        return memory_deleted or file_deleted
    
    def clear(self) -> None:
        """Clear both caches."""
        if self.memory_cache:
            self.memory_cache.clear()
        if self.file_cache:
            self.file_cache.clear()
    
    def cleanup(self) -> int:
        """Clean up expired entries and return count of removed items."""
        removed_count = 0
        
        if self.memory_cache:
            # Memory cache cleanup is automatic
            pass
        
        if self.file_cache:
            removed_count = self.file_cache.cleanup_expired()
        
        return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory_cache_enabled": self.memory_cache is not None,
            "file_cache_enabled": self.file_cache is not None,
        }
        
        if self.memory_cache:
            stats["memory_cache_size"] = self.memory_cache.size()
            stats["memory_cache_keys"] = len(self.memory_cache.keys())
        
        if self.file_cache:
            cache_files = list(self.file_cache.cache_dir.glob("*.cache"))
            stats["file_cache_files"] = len(cache_files)
        
        return stats


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def set_cache_manager(cache_manager: CacheManager) -> None:
    """Set the global cache manager instance."""
    global _cache_manager
    _cache_manager = cache_manager


def cache_result(ttl: int = 3600, key_prefix: str = "", 
                cache_manager: Optional[CacheManager] = None) -> Callable:
    """Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        cache_manager: Cache manager to use (uses global if None)
    
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache manager
            cm = cache_manager or get_cache_manager()
            
            # Generate cache key
            key_data = {
                "function": func.__name__,
                "args": args,
                "kwargs": kwargs
            }
            key_string = json.dumps(key_data, sort_keys=True, default=str)
            cache_key = f"{key_prefix}:{hashlib.md5(key_string.encode()).hexdigest()}"
            
            # Try to get from cache
            cached_result = cm.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cm.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache(pattern: str = "", cache_manager: Optional[CacheManager] = None) -> int:
    """Invalidate cache entries matching a pattern.
    
    Args:
        pattern: Pattern to match cache keys (empty string matches all)
        cache_manager: Cache manager to use (uses global if None)
    
    Returns:
        Number of invalidated entries
    """
    cm = cache_manager or get_cache_manager()
    invalidated_count = 0
    
    # Invalidate memory cache
    if cm.memory_cache:
        keys_to_delete = []
        for key in cm.memory_cache.keys():
            if not pattern or pattern in key:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            if cm.memory_cache.delete(key):
                invalidated_count += 1
    
    # Invalidate file cache
    if cm.file_cache:
        for cache_file in cm.file_cache.cache_dir.glob("*.cache"):
            try:
                # We can't easily check the key without loading the file,
                # so we'll delete all files if pattern is empty
                if not pattern:
                    if cm.file_cache.delete(cache_file.stem):
                        invalidated_count += 1
            except Exception:
                pass
    
    return invalidated_count