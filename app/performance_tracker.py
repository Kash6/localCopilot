"""Performance monitoring and metrics tracking."""
import time
import json
from datetime import datetime
from typing import Dict, List
from pathlib import Path


class PerformanceTracker:
    """Track and analyze performance metrics for the coding assistant."""
    
    def __init__(self, metrics_file: str = "metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.current_session = {
            "queries": [],
            "start_time": datetime.now().isoformat()
        }
        self.baseline_latency = None
    
    def start_query(self) -> float:
        """Start timing a query."""
        return time.time()
    
    def end_query(self, start_time: float, query: str, tokens_generated: int = 0) -> Dict:
        """End timing and record metrics."""
        latency = time.time() - start_time
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # Truncate for storage
            "latency_seconds": latency,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_generated / latency if latency > 0 else 0
        }
        
        self.current_session["queries"].append(metric)
        return metric
    
    def calculate_improvement(self) -> Dict:
        """Calculate performance improvements."""
        if len(self.current_session["queries"]) < 2:
            return {"improvement": 0, "message": "Insufficient data"}
        
        latencies = [q["latency_seconds"] for q in self.current_session["queries"]]
        
        # Use first query as baseline if not set
        if self.baseline_latency is None:
            self.baseline_latency = latencies[0]
        
        avg_current = sum(latencies[1:]) / len(latencies[1:])
        improvement = ((self.baseline_latency - avg_current) / self.baseline_latency) * 100
        
        return {
            "baseline_latency": self.baseline_latency,
            "current_avg_latency": avg_current,
            "improvement_percentage": improvement,
            "total_queries": len(latencies)
        }
    
    def save_metrics(self):
        """Save metrics to file."""
        self.current_session["end_time"] = datetime.now().isoformat()
        self.current_session["summary"] = self.calculate_improvement()
        
        # Load existing metrics
        existing = []
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                existing = json.load(f)
        
        existing.append(self.current_session)
        
        with open(self.metrics_file, 'w') as f:
            json.dump(existing, f, indent=2)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        if not self.current_session["queries"]:
            return {}
        
        latencies = [q["latency_seconds"] for q in self.current_session["queries"]]
        tokens_per_sec = [q["tokens_per_second"] for q in self.current_session["queries"]]
        
        return {
            "total_queries": len(latencies),
            "avg_latency": sum(latencies) / len(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "avg_tokens_per_sec": sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0,
            "improvement": self.calculate_improvement()
        }
