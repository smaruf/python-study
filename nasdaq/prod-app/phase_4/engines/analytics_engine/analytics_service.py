"""Analytics Engine for Phase 4."""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    total_orders: int
    filled_orders: int
    cancelled_orders: int
    rejected_orders: int
    total_volume: float
    total_value: float
    average_fill_price: float
    fill_rate: float


@dataclass
class PerformanceReport:
    """Performance analysis report."""
    report_id: str
    symbol: str
    period_start: datetime
    period_end: datetime
    metrics: TradingMetrics
    generated_at: datetime


class AnalyticsEngine:
    """Analytics Engine for Phase 4."""
    
    def __init__(self):
        self.trade_data: List[Dict] = []
        self.order_data: List[Dict] = []
        self.reports: Dict[str, PerformanceReport] = {}
    
    async def record_trade(self, trade_data: Dict):
        """Record trade data for analytics."""
        trade_data['timestamp'] = datetime.utcnow()
        self.trade_data.append(trade_data)
        
        # Keep only recent data
        if len(self.trade_data) > 10000:
            self.trade_data = self.trade_data[-5000:]
    
    async def record_order(self, order_data: Dict):
        """Record order data for analytics."""
        order_data['timestamp'] = datetime.utcnow()
        self.order_data.append(order_data)
        
        # Keep only recent data
        if len(self.order_data) > 10000:
            self.order_data = self.order_data[-5000:]
    
    async def generate_trading_metrics(self, symbol: str = None, 
                                     period_hours: int = 24) -> TradingMetrics:
        """Generate trading metrics for a period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=period_hours)
        
        # Filter orders by symbol and time
        filtered_orders = [
            order for order in self.order_data
            if order['timestamp'] > cutoff_time and (
                symbol is None or order.get('symbol') == symbol
            )
        ]
        
        # Calculate metrics
        total_orders = len(filtered_orders)
        filled_orders = len([o for o in filtered_orders if o.get('status') == 'FILLED'])
        cancelled_orders = len([o for o in filtered_orders if o.get('status') == 'CANCELLED'])
        rejected_orders = len([o for o in filtered_orders if o.get('status') == 'REJECTED'])
        
        total_volume = sum(o.get('quantity', 0) for o in filtered_orders if o.get('status') == 'FILLED')
        total_value = sum(
            o.get('quantity', 0) * o.get('price', 0) 
            for o in filtered_orders 
            if o.get('status') == 'FILLED' and o.get('price')
        )
        
        average_fill_price = total_value / total_volume if total_volume > 0 else 0
        fill_rate = filled_orders / total_orders if total_orders > 0 else 0
        
        return TradingMetrics(
            total_orders=total_orders,
            filled_orders=filled_orders,
            cancelled_orders=cancelled_orders,
            rejected_orders=rejected_orders,
            total_volume=total_volume,
            total_value=total_value,
            average_fill_price=average_fill_price,
            fill_rate=fill_rate
        )
    
    async def generate_performance_report(self, symbol: str, period_hours: int = 24) -> PerformanceReport:
        """Generate comprehensive performance report."""
        period_end = datetime.utcnow()
        period_start = period_end - timedelta(hours=period_hours)
        
        metrics = await self.generate_trading_metrics(symbol, period_hours)
        
        report = PerformanceReport(
            report_id=f"RPT_{datetime.utcnow().timestamp()}",
            symbol=symbol,
            period_start=period_start,
            period_end=period_end,
            metrics=metrics,
            generated_at=datetime.utcnow()
        )
        
        self.reports[report.report_id] = report
        return report
    
    async def get_market_summary(self) -> Dict:
        """Get overall market summary."""
        symbols = set(order.get('symbol') for order in self.order_data if order.get('symbol'))
        
        summary = {}
        for symbol in symbols:
            metrics = await self.generate_trading_metrics(symbol, 24)
            summary[symbol] = {
                "total_orders": metrics.total_orders,
                "fill_rate": round(metrics.fill_rate * 100, 2),
                "total_volume": metrics.total_volume,
                "average_price": round(metrics.average_fill_price, 2)
            }
        
        return summary
    
    async def export_report(self, report_id: str) -> Optional[str]:
        """Export report as JSON."""
        if report_id not in self.reports:
            return None
        
        report = self.reports[report_id]
        
        report_data = {
            "report_id": report.report_id,
            "symbol": report.symbol,
            "period": {
                "start": report.period_start.isoformat(),
                "end": report.period_end.isoformat()
            },
            "metrics": {
                "total_orders": report.metrics.total_orders,
                "filled_orders": report.metrics.filled_orders,
                "cancelled_orders": report.metrics.cancelled_orders,
                "rejected_orders": report.metrics.rejected_orders,
                "total_volume": report.metrics.total_volume,
                "total_value": report.metrics.total_value,
                "average_fill_price": report.metrics.average_fill_price,
                "fill_rate": report.metrics.fill_rate
            },
            "generated_at": report.generated_at.isoformat()
        }
        
        return json.dumps(report_data, indent=2)
