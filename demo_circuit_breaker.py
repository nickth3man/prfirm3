#!/usr/bin/env python3
"""Demo script showing Circuit Breaker functionality with the Virtual PR Firm nodes.

This script demonstrates how the circuit breaker protects against cascading failures
when external LLM services become unavailable, with automatic fallback to safe alternatives.
"""

import logging
import time
from typing import Dict, Any

# Enable detailed logging to see circuit breaker behavior
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

# Import circuit breaker utilities
from utils.circuit_breaker import (
    circuit_breaker, CircuitBreakerError, get_circuit_breaker,
    list_circuit_breakers, reset_all_circuit_breakers
)

def simulate_llm_service_outage():
    """Demonstrate circuit breaker behavior during a simulated LLM service outage."""
    
    print("🔧 Circuit Breaker Demo: Simulating LLM Service Outage")
    print("=" * 60)
    
    # Create a mock LLM service that will fail
    failure_count = 0
    
    @circuit_breaker(name="demo_llm_service", failure_threshold=3, reset_timeout=2.0)
    def mock_llm_service(prompt: str) -> str:
        nonlocal failure_count
        failure_count += 1
        
        # Simulate service degradation and failure
        if failure_count <= 2:
            print(f"🟢 LLM Call #{failure_count}: Success")
            return f"Generated content for: {prompt}"
        elif failure_count <= 5:
            print(f"🔴 LLM Call #{failure_count}: Service failure!")
            raise Exception("Service temporarily unavailable")
        else:
            print(f"🟢 LLM Call #{failure_count}: Service recovered")
            return f"Generated content for: {prompt}"
    
    def fallback_content_generation(prompt: str) -> str:
        """Fallback when LLM service is unavailable."""
        return f"Fallback template content for: {prompt}"
    
    def generate_content_with_protection(prompt: str) -> str:
        """Content generation with circuit breaker protection."""
        try:
            return mock_llm_service(prompt)
        except CircuitBreakerError:
            print("⚡ Circuit breaker is OPEN - using fallback")
            return fallback_content_generation(prompt)
        except Exception as e:
            print(f"🔴 Service error: {e}")
            return fallback_content_generation(prompt)
    
    # Test scenario
    prompts = [
        "Create a Twitter post about innovation",
        "Write a LinkedIn article about leadership", 
        "Generate Instagram content about creativity",
        "Create a blog post about technology trends",
        "Write content about sustainable business",
        "Generate a product announcement",
        "Create content about team collaboration"
    ]
    
    print("\n📝 Starting content generation requests...")
    print("   (Circuit breaker will open after 3 consecutive failures)")
    print()
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Request #{i}: {prompt}")
        result = generate_content_with_protection(prompt)
        print(f"Result: {result}")
        
        # Show circuit breaker status
        breaker = mock_llm_service._circuit_breaker
        status = breaker.get_status()
        print(f"Circuit Status: {status['state'].upper()} "
              f"(failures: {status['consecutive_failures']}, "
              f"total calls: {status['metrics']['total_calls']})")
        
        print()
        
        # Add delay to demonstrate reset timeout behavior
        if i == 5:
            print("⏰ Waiting for circuit breaker reset timeout (2 seconds)...")
            time.sleep(2.5)
            print("✅ Timeout elapsed - circuit breaker will attempt recovery")
            print()
    
    # Show final statistics
    print("📊 Final Circuit Breaker Statistics:")
    print("-" * 40)
    breakers = list_circuit_breakers()
    for name, status in breakers.items():
        if "demo" in name:
            metrics = status['metrics']
            print(f"Service: {name}")
            print(f"  State: {status['state'].upper()}")
            print(f"  Total calls: {metrics['total_calls']}")
            print(f"  Successful: {metrics['successful_calls']}")
            print(f"  Failed: {metrics['failed_calls']}")
            print(f"  Blocked: {metrics['blocked_calls']}")
            print(f"  Success rate: {metrics['success_rate']:.1%}")
            print(f"  State changes: {metrics['state_changes']}")

def demonstrate_node_integration():
    """Show how circuit breaker protects actual Virtual PR Firm nodes."""
    
    print("\n🏗️  Node Integration Demo: ContentCraftsmanNode Protection")
    print("=" * 60)
    
    # Simulate node execution with circuit breaker protection
    print("Simulating ContentCraftsmanNode with LLM circuit breaker protection...")
    print("(This would normally call real LLM APIs)")
    
    # Show that circuit breakers are already integrated
    from utils.call_llm import call_llm
    
    print("\nTesting protected LLM utility:")
    try:
        result = call_llm("Generate content about virtual assistants")
        print(f"✅ LLM Response: {result[:100]}...")
    except Exception as e:
        print(f"ℹ️  LLM unavailable (expected): {type(e).__name__}")
    
    # Show circuit breaker registry
    print("\n🗂️  Active Circuit Breakers in System:")
    breakers = list_circuit_breakers()
    if breakers:
        for name, status in breakers.items():
            print(f"  - {name}: {status['state']} ({status['metrics']['total_calls']} calls)")
    else:
        print("  - No active circuit breakers (LLM services not called yet)")

def show_configuration_examples():
    """Show different circuit breaker configurations for different scenarios."""
    
    print("\n⚙️  Configuration Examples for Different Services")
    print("=" * 60)
    
    configs = [
        {
            "name": "High-Cost LLM APIs (GPT-4, Claude)",
            "config": {"failure_threshold": 2, "reset_timeout": 120, "success_threshold": 2},
            "reason": "Expensive - fail fast to avoid costs"
        },
        {
            "name": "Standard LLM APIs (GPT-3.5, Llama)",
            "config": {"failure_threshold": 3, "reset_timeout": 60, "success_threshold": 2},
            "reason": "Balanced protection and functionality"
        },
        {
            "name": "Real-time Services (Streaming, Chat)",
            "config": {"failure_threshold": 5, "reset_timeout": 30, "success_threshold": 3},
            "reason": "Allow more failures for real-time tolerance"
        },
        {
            "name": "Database Operations",
            "config": {"failure_threshold": 2, "reset_timeout": 180, "success_threshold": 2},
            "reason": "Conservative - DB issues may take time to resolve"
        },
        {
            "name": "External APIs (non-critical)",
            "config": {"failure_threshold": 10, "reset_timeout": 300, "success_threshold": 5},
            "reason": "Lenient - not critical to core functionality"
        }
    ]
    
    for service in configs:
        print(f"🔧 {service['name']}:")
        config = service['config']
        print(f"   failure_threshold={config['failure_threshold']}, "
              f"reset_timeout={config['reset_timeout']}s, "
              f"success_threshold={config['success_threshold']}")
        print(f"   Rationale: {service['reason']}")
        print()

if __name__ == "__main__":
    print("🎯 Virtual PR Firm - Circuit Breaker Protection Demo")
    print("This demo shows how circuit breakers prevent cascading failures")
    print()
    
    try:
        # Run demonstrations
        simulate_llm_service_outage()
        demonstrate_node_integration()
        show_configuration_examples()
        
        print("\n✅ Demo completed successfully!")
        print("\nKey Benefits Demonstrated:")
        print("  🛡️  Automatic failure detection and protection")
        print("  🔄 Graceful fallback to alternative implementations")
        print("  📊 Comprehensive metrics and monitoring")
        print("  ⚡ Fast failure detection to prevent resource waste")
        print("  🏥 Automatic recovery testing and circuit restoration")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up for demo purposes
        reset_all_circuit_breakers()
        print("\n🧹 Circuit breakers reset for next demo")