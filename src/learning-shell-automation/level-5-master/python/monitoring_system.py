#!/usr/bin/env python3
"""
Level 5 - Master: Monitoring and Alerting System
Production-ready monitoring with alerting, metrics, and dashboards
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"

@dataclass
class Metric:
    """Represents a metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str]
    
    def to_dict(self):
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels
        }

@dataclass
class Alert:
    """Represents an alert"""
    name: str
    level: AlertLevel
    message: str
    timestamp: datetime
    metric: Optional[Metric] = None
    
    def to_dict(self):
        data = {
            'name': self.name,
            'level': self.level.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat()
        }
        if self.metric:
            data['metric'] = self.metric.to_dict()
        return data

class AlertRule:
    """Defines an alerting rule"""
    
    def __init__(self, name: str, metric_name: str, 
                 threshold: float, level: AlertLevel, 
                 comparison: str = "greater"):
        self.name = name
        self.metric_name = metric_name
        self.threshold = threshold
        self.level = level
        self.comparison = comparison
    
    def evaluate(self, metric: Metric) -> Optional[Alert]:
        """Evaluate if metric triggers this rule"""
        if metric.name != self.metric_name:
            return None
        
        triggered = False
        if self.comparison == "greater" and metric.value > self.threshold:
            triggered = True
        elif self.comparison == "less" and metric.value < self.threshold:
            triggered = True
        
        if triggered:
            message = (f"{metric.name} is {metric.value:.2f}, "
                      f"threshold: {self.threshold}")
            return Alert(
                name=self.name,
                level=self.level,
                message=message,
                timestamp=datetime.now(),
                metric=metric
            )
        
        return None

class MonitoringSystem:
    """Complete monitoring and alerting system"""
    
    def __init__(self):
        self.metrics: List[Metric] = []
        self.alerts: List[Alert] = []
        self.alert_rules: List[AlertRule] = []
        self.dashboards: Dict[str, Any] = {}
        
        self.setup_default_rules()
    
    def setup_default_rules(self):
        """Setup default alerting rules"""
        self.alert_rules = [
            AlertRule("high_cpu", "cpu_usage", 80.0, AlertLevel.WARNING),
            AlertRule("critical_cpu", "cpu_usage", 95.0, AlertLevel.CRITICAL),
            AlertRule("high_memory", "memory_usage", 85.0, AlertLevel.WARNING),
            AlertRule("critical_memory", "memory_usage", 95.0, AlertLevel.CRITICAL),
            AlertRule("high_disk", "disk_usage", 90.0, AlertLevel.WARNING),
            AlertRule("error_rate", "error_rate", 5.0, AlertLevel.WARNING),
            AlertRule("slow_response", "response_time_ms", 1000.0, AlertLevel.WARNING),
        ]
    
    def record_metric(self, name: str, value: float, 
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Dict[str, str] = None):
        """Record a new metric"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        
        self.metrics.append(metric)
        logger.info(f"Metric recorded: {name}={value}")
        
        # Evaluate alert rules
        self.evaluate_alerts(metric)
    
    def evaluate_alerts(self, metric: Metric):
        """Evaluate all alert rules for a metric"""
        for rule in self.alert_rules:
            alert = rule.evaluate(metric)
            if alert:
                self.trigger_alert(alert)
    
    def trigger_alert(self, alert: Alert):
        """Trigger an alert"""
        self.alerts.append(alert)
        
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(f"ALERT: {alert.message}")
            self.send_notification(alert)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(f"ALERT: {alert.message}")
        else:
            logger.info(f"ALERT: {alert.message}")
    
    def send_notification(self, alert: Alert):
        """Send alert notification"""
        # Simulate sending notifications
        logger.info(f"Sending notification for: {alert.name}")
        
        # In production, integrate with:
        # - Slack
        # - PagerDuty
        # - Email
        # - SMS
        
        notification = {
            'alert': alert.to_dict(),
            'channels': ['slack', 'email'],
            'sent_at': datetime.now().isoformat()
        }
        
        logger.info(f"Notification: {json.dumps(notification, indent=2)}")
    
    def get_metrics_summary(self, time_range: timedelta = None) -> Dict:
        """Get summary of metrics"""
        if time_range:
            cutoff = datetime.now() - time_range
            recent_metrics = [m for m in self.metrics if m.timestamp > cutoff]
        else:
            recent_metrics = self.metrics
        
        summary = {}
        for metric in recent_metrics:
            if metric.name not in summary:
                summary[metric.name] = {
                    'count': 0,
                    'sum': 0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'latest': None
                }
            
            s = summary[metric.name]
            s['count'] += 1
            s['sum'] += metric.value
            s['min'] = min(s['min'], metric.value)
            s['max'] = max(s['max'], metric.value)
            s['latest'] = metric.value
        
        # Calculate averages
        for name, data in summary.items():
            data['average'] = data['sum'] / data['count'] if data['count'] > 0 else 0
        
        return summary
    
    def get_active_alerts(self, time_range: timedelta = timedelta(hours=1)) -> List[Alert]:
        """Get active alerts within time range"""
        cutoff = datetime.now() - time_range
        return [a for a in self.alerts if a.timestamp > cutoff]
    
    def export_metrics(self, filename: str = "metrics.json"):
        """Export metrics to file"""
        data = {
            'metrics': [m.to_dict() for m in self.metrics],
            'alerts': [a.to_dict() for a in self.alerts],
            'exported_at': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported to {filename}")
    
    def create_dashboard(self, name: str) -> Dict:
        """Create a monitoring dashboard"""
        summary = self.get_metrics_summary(timedelta(minutes=5))
        active_alerts = self.get_active_alerts()
        
        dashboard = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'metrics_summary': summary,
            'active_alerts': [a.to_dict() for a in active_alerts],
            'total_metrics': len(self.metrics),
            'total_alerts': len(self.alerts)
        }
        
        self.dashboards[name] = dashboard
        return dashboard

def simulate_monitoring():
    """Simulate a monitoring session"""
    system = MonitoringSystem()
    
    logger.info("="*60)
    logger.info("Production Monitoring System")
    logger.info("="*60)
    
    # Simulate collecting metrics over time
    for i in range(10):
        # CPU usage
        cpu = 70 + (i * 5)  # Gradually increasing
        system.record_metric("cpu_usage", cpu, labels={'host': 'web-01'})
        
        # Memory usage
        memory = 75 + (i * 3)
        system.record_metric("memory_usage", memory, labels={'host': 'web-01'})
        
        # Response time
        response_time = 500 + (i * 100)
        system.record_metric("response_time_ms", response_time, labels={'endpoint': '/api/users'})
        
        # Error rate
        error_rate = 1 + (i * 0.5)
        system.record_metric("error_rate", error_rate, labels={'service': 'api'})
        
        time.sleep(0.5)
    
    # Generate dashboard
    logger.info("\n" + "="*60)
    logger.info("Generating Dashboard")
    logger.info("="*60)
    
    dashboard = system.create_dashboard("Production Overview")
    print(json.dumps(dashboard, indent=2))
    
    # Export metrics
    system.export_metrics()
    
    logger.info("\n" + "="*60)
    logger.info("Monitoring Summary")
    logger.info("="*60)
    logger.info(f"Total metrics collected: {len(system.metrics)}")
    logger.info(f"Total alerts triggered: {len(system.alerts)}")
    
    active_alerts = system.get_active_alerts()
    logger.info(f"Active alerts: {len(active_alerts)}")
    
    for alert in active_alerts:
        logger.info(f"  - [{alert.level.value.upper()}] {alert.message}")

def main():
    """Main execution"""
    simulate_monitoring()

if __name__ == "__main__":
    main()
