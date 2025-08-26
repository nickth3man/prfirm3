"""
Performance monitoring and metrics collection.

This module provides performance tracking, timing, and metrics collection
for the Virtual PR Firm pipeline to help identify bottlenecks and optimize performance.
"""

import logging
import time
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Represents a performance metric."""
    name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NodePerformance:
    """Performance data for a specific node."""
    node_name: str
    execution_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_execution: Optional[float] = None
    error_count: int = 0
    
    def update(self, execution_time: float, success: bool = True):
        """Update performance data with new execution."""
        self.execution_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.execution_count
        self.last_execution = time.time()
        
        if not success:
            self.error_count += 1

class PerformanceMonitor:
    """Monitors and tracks performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: deque = deque(maxlen=max_history)
        self.node_performance: Dict[str, NodePerformance] = defaultdict(lambda: NodePerformance(""))
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, unit: str = "", metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self.metrics.append(metric)
    
    def record_node_execution(self, node_name: str, execution_time: float, success: bool = True):
        """Record node execution performance."""
        with self._lock:
            if node_name not in self.node_performance:
                self.node_performance[node_name] = NodePerformance(node_name)
            
            self.node_performance[node_name].update(execution_time, success)
    
    def get_node_performance(self, node_name: str) -> Optional[NodePerformance]:
        """Get performance data for a specific node."""
        with self._lock:
            return self.node_performance.get(node_name)
    
    def get_all_node_performance(self) -> Dict[str, NodePerformance]:
        """Get performance data for all nodes."""
        with self._lock:
            return dict(self.node_performance)
    
    def get_metrics(self, name: Optional[str] = None, limit: Optional[int] = None) -> List[PerformanceMetric]:
        """Get metrics with optional filtering."""
        with self._lock:
            metrics = list(self.metrics)
        
        if name:
            metrics = [m for m in metrics if m.name == name]
        
        if limit:
            metrics = metrics[-limit:]
        
        return metrics
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics."""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, system metrics disabled")
            return {}
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }
        except Exception as e:
            logger.warning(f"Could not get system metrics: {e}")
            return {}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all performance data."""
        with self._lock:
            total_nodes = len(self.node_performance)
            total_executions = sum(node.execution_count for node in self.node_performance.values())
            total_errors = sum(node.error_count for node in self.node_performance.values())
            
            # Find slowest and fastest nodes
            if self.node_performance:
                slowest_node = max(self.node_performance.values(), key=lambda x: x.avg_time)
                fastest_node = min(self.node_performance.values(), key=lambda x: x.avg_time)
            else:
                slowest_node = fastest_node = None
        
        return {
            "uptime_seconds": time.time() - self.start_time,
            "total_nodes": total_nodes,
            "total_executions": total_executions,
            "total_errors": total_errors,
            "error_rate": total_errors / total_executions if total_executions > 0 else 0,
            "slowest_node": slowest_node.node_name if slowest_node else None,
            "slowest_avg_time": slowest_node.avg_time if slowest_node else 0,
            "fastest_node": fastest_node.node_name if fastest_node else None,
            "fastest_avg_time": fastest_node.avg_time if fastest_node else 0,
            "system_metrics": self.get_system_metrics()
        }
    
    def reset(self):
        """Reset all performance data."""
        with self._lock:
            self.metrics.clear()
            self.node_performance.clear()
            self.start_time = time.time()

@contextmanager
def monitor_node_execution(monitor: PerformanceMonitor, node_name: str):
    """Context manager to monitor node execution time."""
    start_time = time.time()
    success = False
    
    try:
        yield
        success = True
    except Exception as e:
        logger.error(f"Error in node {node_name}: {e}")
        raise
    finally:
        execution_time = time.time() - start_time
        monitor.record_node_execution(node_name, execution_time, success)

class MetricsCollector:
    """Collects and aggregates metrics from multiple sources."""
    
    def __init__(self):
        self.collectors: Dict[str, Callable] = {}
        self.collection_interval = 60  # seconds
        self.last_collection = 0
    
    def register_collector(self, name: str, collector: Callable[[], Dict[str, Any]]):
        """Register a metrics collector."""
        self.collectors[name] = collector
    
    def collect_metrics(self, force: bool = False) -> Dict[str, Any]:
        """Collect metrics from all registered collectors."""
        current_time = time.time()
        
        if not force and current_time - self.last_collection < self.collection_interval:
            return {}
        
        metrics = {}
        for name, collector in self.collectors.items():
            try:
                metrics[name] = collector()
            except Exception as e:
                logger.error(f"Error collecting metrics from {name}: {e}")
                metrics[name] = {"error": str(e)}
        
        self.last_collection = current_time
        return metrics

# Global performance monitor instance
_global_monitor = PerformanceMonitor()

def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _global_monitor

def record_metric(name: str, value: float, unit: str = "", metadata: Optional[Dict[str, Any]] = None):
    """Record a metric using the global monitor."""
    _global_monitor.record_metric(name, value, unit, metadata)

def record_node_execution(node_name: str, execution_time: float, success: bool = True):
    """Record node execution using the global monitor."""
    _global_monitor.record_node_execution(node_name, execution_time, success)

@contextmanager
def monitor_execution(node_name: str):
    """Context manager to monitor execution using the global monitor."""
    with monitor_node_execution(_global_monitor, node_name):
        yield

# Test function for development
if __name__ == "__main__":
    # Test performance monitoring
    monitor = PerformanceMonitor()
    
    # Simulate some node executions
    with monitor_node_execution(monitor, "TestNode1"):
        time.sleep(0.1)
    
    with monitor_node_execution(monitor, "TestNode2"):
        time.sleep(0.2)
    
    # Record some metrics
    monitor.record_metric("memory_usage", 512.5, "MB")
    monitor.record_metric("api_calls", 10, "count")
    
    # Get performance summary
    summary = monitor.get_summary()
    print("Performance summary:", summary)
    
    # Get node performance
    node_perf = monitor.get_node_performance("TestNode1")
    if node_perf:
        print(f"TestNode1 performance: {node_perf}")
    
    # Test metrics collector
    collector = MetricsCollector()
    
    def test_collector():
        return {"test_metric": 42}
    
    collector.register_collector("test", test_collector)
    metrics = collector.collect_metrics(force=True)
    print("Collected metrics:", metrics)