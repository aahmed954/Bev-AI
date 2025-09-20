"""
Comprehensive Logging and Alerting System for BEV Auto-Recovery
==============================================================

Advanced logging, monitoring, and alerting infrastructure with
structured logging, distributed tracing, and multi-channel alerting.

Features:
- Structured logging with contextual metadata
- Distributed tracing across service boundaries
- Multi-level alerting with escalation policies
- Real-time metrics collection and dashboards
- Integration with external monitoring systems
- Compliance logging and audit trails

Author: BEV Infrastructure Team
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager
import structlog
import aiohttp
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import redis
import psycopg2
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import asyncio_mqtt
import opentelemetry
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.asyncio import AsyncIOInstrumentor


class AlertLevel(Enum):
    """Alert severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Available alert channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"
    DISCORD = "discord"


class LogEvent(Enum):
    """Standard log event types."""
    SERVICE_START = "service_start"
    SERVICE_STOP = "service_stop"
    HEALTH_CHECK = "health_check"
    RECOVERY_START = "recovery_start"
    RECOVERY_SUCCESS = "recovery_success"
    RECOVERY_FAILURE = "recovery_failure"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    CIRCUIT_BREAKER_CLOSE = "circuit_breaker_close"
    SNAPSHOT_CREATED = "snapshot_created"
    ROLLBACK_PERFORMED = "rollback_performed"
    ALERT_SENT = "alert_sent"
    METRIC_THRESHOLD_EXCEEDED = "metric_threshold_exceeded"
    CONFIG_CHANGE = "config_change"
    SECURITY_EVENT = "security_event"
    AUDIT_EVENT = "audit_event"


@dataclass
class Alert:
    """Alert message structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    level: AlertLevel = AlertLevel.INFO
    title: str = ""
    message: str = ""
    service: str = ""
    component: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    channels: List[AlertChannel] = field(default_factory=list)
    escalation_level: int = 0
    acknowledged: bool = False
    resolved: bool = False
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    level: str = "INFO"
    event: LogEvent = LogEvent.SERVICE_START
    service: str = ""
    component: str = ""
    message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class MetricPoint:
    """Metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram


class AlertManager:
    """
    Advanced alert management with escalation policies and channel routing.
    """

    def __init__(self,
                 redis_client: Optional[redis.Redis] = None,
                 postgres_engine = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize alert manager.

        Args:
            redis_client: Redis client for state management
            postgres_engine: PostgreSQL engine for persistence
            config: Alert configuration
        """
        self.redis_client = redis_client
        self.postgres_engine = postgres_engine
        self.config = config or {}

        # Alert state management
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.escalation_policies: Dict[str, Dict[str, Any]] = {}

        # Channel configurations
        self.channel_configs = {
            AlertChannel.EMAIL: self.config.get('email', {}),
            AlertChannel.SLACK: self.config.get('slack', {}),
            AlertChannel.WEBHOOK: self.config.get('webhook', {}),
            AlertChannel.SMS: self.config.get('sms', {}),
            AlertChannel.PAGERDUTY: self.config.get('pagerduty', {}),
        }

        # Metrics
        self.alerts_sent = Counter('alerts_sent_total', 'Total alerts sent', ['level', 'channel'])
        self.alert_processing_time = Histogram('alert_processing_seconds', 'Alert processing time')
        self.active_alerts_gauge = Gauge('active_alerts', 'Number of active alerts', ['level'])

        # Background tasks
        self.escalation_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None

        self.logger = structlog.get_logger("alert_manager")

    async def send_alert(self, alert: Alert) -> bool:
        """
        Send alert through configured channels.

        Args:
            alert: Alert to send

        Returns:
            bool: Success status
        """
        start_time = time.time()

        try:
            # Store alert
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)

            # Update metrics
            self.active_alerts_gauge.labels(level=alert.level.value).inc()

            # Send through channels
            send_tasks = []
            for channel in alert.channels:
                task = asyncio.create_task(
                    self._send_to_channel(alert, channel)
                )
                send_tasks.append(task)

            # Wait for all channels
            results = await asyncio.gather(*send_tasks, return_exceptions=True)

            # Check if any channel succeeded
            success = any(result is True for result in results if not isinstance(result, Exception))

            # Log failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(
                        "Alert channel failed",
                        alert_id=alert.id,
                        channel=alert.channels[i].value,
                        error=str(result)
                    )

            # Save to database
            if self.postgres_engine:
                await self._save_alert_to_db(alert)

            # Save to Redis for state management
            if self.redis_client:
                await self._save_alert_to_redis(alert)

            # Update metrics
            processing_time = time.time() - start_time
            self.alert_processing_time.observe(processing_time)

            if success:
                for channel in alert.channels:
                    self.alerts_sent.labels(
                        level=alert.level.value,
                        channel=channel.value
                    ).inc()

            self.logger.info(
                "Alert sent",
                alert_id=alert.id,
                level=alert.level.value,
                channels=[c.value for c in alert.channels],
                success=success,
                processing_time=processing_time
            )

            return success

        except Exception as e:
            self.logger.error(
                "Alert sending failed",
                alert_id=alert.id,
                error=str(e)
            )
            return False

    async def _send_to_channel(self, alert: Alert, channel: AlertChannel) -> bool:
        """Send alert to specific channel."""
        try:
            if channel == AlertChannel.EMAIL:
                return await self._send_email_alert(alert)
            elif channel == AlertChannel.SLACK:
                return await self._send_slack_alert(alert)
            elif channel == AlertChannel.WEBHOOK:
                return await self._send_webhook_alert(alert)
            elif channel == AlertChannel.SMS:
                return await self._send_sms_alert(alert)
            elif channel == AlertChannel.PAGERDUTY:
                return await self._send_pagerduty_alert(alert)
            else:
                self.logger.warning(f"Unknown alert channel: {channel}")
                return False

        except Exception as e:
            self.logger.error(f"Channel {channel.value} failed: {e}")
            return False

    async def _send_email_alert(self, alert: Alert) -> bool:
        """Send email alert."""
        config = self.channel_configs.get(AlertChannel.EMAIL, {})
        if not config.get('enabled', False):
            return False

        try:
            msg = MimeMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"

            # Create HTML body
            html_body = f"""
            <html>
            <body>
                <h2>Alert: {alert.title}</h2>
                <table border="1" cellpadding="5">
                    <tr><td><b>Level:</b></td><td>{alert.level.value}</td></tr>
                    <tr><td><b>Service:</b></td><td>{alert.service}</td></tr>
                    <tr><td><b>Component:</b></td><td>{alert.component}</td></tr>
                    <tr><td><b>Time:</b></td><td>{alert.timestamp.isoformat()}</td></tr>
                    <tr><td><b>Message:</b></td><td>{alert.message}</td></tr>
                </table>

                <h3>Tags:</h3>
                <ul>
                    {''.join(f'<li><b>{k}:</b> {v}</li>' for k, v in alert.tags.items())}
                </ul>

                <h3>Metadata:</h3>
                <pre>{json.dumps(alert.metadata, indent=2)}</pre>

                <p><small>Alert ID: {alert.id}</small></p>
            </body>
            </html>
            """

            msg.attach(MimeText(html_body, 'html'))

            # Send email
            with smtplib.SMTP(config['smtp_host'], config['smtp_port']) as server:
                if config.get('use_tls', True):
                    server.starttls()
                if config.get('username'):
                    server.login(config['username'], config['password'])
                server.send_message(msg)

            return True

        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")
            return False

    async def _send_slack_alert(self, alert: Alert) -> bool:
        """Send Slack alert."""
        config = self.channel_configs.get(AlertChannel.SLACK, {})
        if not config.get('enabled', False):
            return False

        try:
            # Map alert levels to Slack colors
            color_map = {
                AlertLevel.DEBUG: "#808080",
                AlertLevel.INFO: "#36a64f",
                AlertLevel.WARNING: "#ff9500",
                AlertLevel.ERROR: "#ff0000",
                AlertLevel.CRITICAL: "#8b0000",
                AlertLevel.EMERGENCY: "#ff1493"
            }

            payload = {
                "text": f"Alert: {alert.title}",
                "attachments": [{
                    "color": color_map.get(alert.level, "#808080"),
                    "fields": [
                        {"title": "Level", "value": alert.level.value.upper(), "short": True},
                        {"title": "Service", "value": alert.service, "short": True},
                        {"title": "Component", "value": alert.component, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True},
                        {"title": "Message", "value": alert.message, "short": False}
                    ],
                    "footer": f"Alert ID: {alert.id}",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }

            # Add tags if present
            if alert.tags:
                payload["attachments"][0]["fields"].append({
                    "title": "Tags",
                    "value": ", ".join(f"{k}={v}" for k, v in alert.tags.items()),
                    "short": False
                })

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config['webhook_url'],
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    return response.status == 200

        except Exception as e:
            self.logger.error(f"Slack alert failed: {e}")
            return False

    async def _send_webhook_alert(self, alert: Alert) -> bool:
        """Send webhook alert."""
        config = self.channel_configs.get(AlertChannel.WEBHOOK, {})
        if not config.get('enabled', False):
            return False

        try:
            payload = asdict(alert)
            # Convert datetime to ISO format
            payload['timestamp'] = alert.timestamp.isoformat()

            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'BEV-AutoRecovery/1.0'
            }

            # Add authentication headers if configured
            if config.get('auth_header'):
                headers['Authorization'] = config['auth_header']

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config['url'],
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    return response.status in [200, 201, 202]

        except Exception as e:
            self.logger.error(f"Webhook alert failed: {e}")
            return False

    async def _send_sms_alert(self, alert: Alert) -> bool:
        """Send SMS alert."""
        config = self.channel_configs.get(AlertChannel.SMS, {})
        if not config.get('enabled', False):
            return False

        try:
            # SMS providers like Twilio, AWS SNS, etc.
            # Implementation depends on provider
            self.logger.info("SMS alert would be sent here")
            return True

        except Exception as e:
            self.logger.error(f"SMS alert failed: {e}")
            return False

    async def _send_pagerduty_alert(self, alert: Alert) -> bool:
        """Send PagerDuty alert."""
        config = self.channel_configs.get(AlertChannel.PAGERDUTY, {})
        if not config.get('enabled', False):
            return False

        try:
            # PagerDuty Events API v2
            payload = {
                "routing_key": config['routing_key'],
                "event_action": "trigger",
                "dedup_key": f"{alert.service}_{alert.component}_{alert.level.value}",
                "payload": {
                    "summary": alert.title,
                    "source": alert.service,
                    "severity": alert.level.value,
                    "timestamp": alert.timestamp.isoformat(),
                    "component": alert.component,
                    "custom_details": {
                        "message": alert.message,
                        "tags": alert.tags,
                        "metadata": alert.metadata,
                        "alert_id": alert.id
                    }
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    return response.status == 202

        except Exception as e:
            self.logger.error(f"PagerDuty alert failed: {e}")
            return False

    async def _save_alert_to_db(self, alert: Alert):
        """Save alert to PostgreSQL database."""
        if not self.postgres_engine:
            return

        try:
            with self.postgres_engine.connect() as conn:
                conn.execute("""
                    INSERT INTO alerts
                    (id, timestamp, level, title, message, service, component,
                     tags, metadata, channels, correlation_id, trace_id)
                    VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    alert.id,
                    alert.timestamp,
                    alert.level.value,
                    alert.title,
                    alert.message,
                    alert.service,
                    alert.component,
                    json.dumps(alert.tags),
                    json.dumps(alert.metadata),
                    json.dumps([c.value for c in alert.channels]),
                    alert.correlation_id,
                    alert.trace_id
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save alert to database: {e}")

    async def _save_alert_to_redis(self, alert: Alert):
        """Save alert to Redis for state management."""
        if not self.redis_client:
            return

        try:
            alert_data = asdict(alert)
            alert_data['timestamp'] = alert.timestamp.isoformat()
            alert_data['channels'] = [c.value for c in alert.channels]
            alert_data['level'] = alert.level.value

            # Store active alert
            self.redis_client.setex(
                f"alert:active:{alert.id}",
                3600,  # 1 hour TTL
                json.dumps(alert_data)
            )

            # Add to alert queue for processing
            self.redis_client.lpush("alert:queue", alert.id)

        except Exception as e:
            self.logger.error(f"Failed to save alert to Redis: {e}")

    async def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.acknowledged = True
                alert.metadata['acknowledged_by'] = user_id
                alert.metadata['acknowledged_at'] = datetime.utcnow().isoformat()

                # Update metrics
                self.active_alerts_gauge.labels(level=alert.level.value).dec()

                self.logger.info(
                    "Alert acknowledged",
                    alert_id=alert_id,
                    user_id=user_id
                )

                return True

        except Exception as e:
            self.logger.error(f"Failed to acknowledge alert: {e}")

        return False

    async def resolve_alert(self, alert_id: str, user_id: str, resolution: str) -> bool:
        """Resolve an alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.metadata['resolved_by'] = user_id
                alert.metadata['resolved_at'] = datetime.utcnow().isoformat()
                alert.metadata['resolution'] = resolution

                # Remove from active alerts
                del self.active_alerts[alert_id]

                # Update metrics
                self.active_alerts_gauge.labels(level=alert.level.value).dec()

                self.logger.info(
                    "Alert resolved",
                    alert_id=alert_id,
                    user_id=user_id,
                    resolution=resolution
                )

                return True

        except Exception as e:
            self.logger.error(f"Failed to resolve alert: {e}")

        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]


class StructuredLogger:
    """
    Advanced structured logging with tracing integration.
    """

    def __init__(self,
                 service_name: str,
                 redis_client: Optional[redis.Redis] = None,
                 postgres_engine = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize structured logger.

        Args:
            service_name: Name of the service
            redis_client: Redis client for log streaming
            postgres_engine: PostgreSQL engine for log persistence
            config: Logger configuration
        """
        self.service_name = service_name
        self.redis_client = redis_client
        self.postgres_engine = postgres_engine
        self.config = config or {}

        # Initialize structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        self.logger = structlog.get_logger(service_name)

        # Tracing setup
        self._setup_tracing()

        # Log streaming
        self.log_queue = asyncio.Queue(maxsize=10000)
        self.streaming_task: Optional[asyncio.Task] = None

    def _setup_tracing(self):
        """Setup distributed tracing."""
        try:
            # Initialize OpenTelemetry
            trace.set_tracer_provider(TracerProvider())
            tracer = trace.get_tracer(__name__)

            # Configure Jaeger exporter if enabled
            if self.config.get('jaeger', {}).get('enabled', False):
                jaeger_exporter = JaegerExporter(
                    agent_host_name=self.config['jaeger'].get('host', 'jaeger'),
                    agent_port=self.config['jaeger'].get('port', 6831),
                )

                span_processor = BatchSpanProcessor(jaeger_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)

            # Auto-instrument asyncio
            AsyncIOInstrumentor().instrument()

            self.tracer = tracer

        except Exception as e:
            print(f"Failed to setup tracing: {e}")
            self.tracer = None

    @asynccontextmanager
    async def trace_span(self, operation_name: str, **kwargs):
        """Create a traced span."""
        if self.tracer:
            with self.tracer.start_as_current_span(operation_name) as span:
                for key, value in kwargs.items():
                    span.set_attribute(key, str(value))
                yield span
        else:
            yield None

    async def log_event(self,
                       level: str,
                       event: LogEvent,
                       message: str,
                       component: str = "",
                       **context):
        """Log a structured event."""
        # Get trace context
        trace_id = None
        span_id = None

        if self.tracer:
            current_span = trace.get_current_span()
            if current_span:
                span_context = current_span.get_span_context()
                trace_id = f"{span_context.trace_id:032x}"
                span_id = f"{span_context.span_id:016x}"

        # Create log entry
        log_entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            event=event,
            service=self.service_name,
            component=component,
            message=message,
            context=context,
            trace_id=trace_id,
            span_id=span_id,
            correlation_id=context.get('correlation_id'),
            user_id=context.get('user_id'),
            session_id=context.get('session_id'),
            request_id=context.get('request_id')
        )

        # Log using structlog
        self.logger.log(
            getattr(logging, level.upper()),
            message,
            event=event.value,
            service=self.service_name,
            component=component,
            trace_id=trace_id,
            span_id=span_id,
            **context
        )

        # Queue for streaming and persistence
        try:
            self.log_queue.put_nowait(log_entry)
        except asyncio.QueueFull:
            # Drop oldest log if queue is full
            try:
                self.log_queue.get_nowait()
                self.log_queue.put_nowait(log_entry)
            except asyncio.QueueEmpty:
                pass

    async def start_streaming(self):
        """Start log streaming and persistence."""
        if self.streaming_task:
            return

        self.streaming_task = asyncio.create_task(self._process_log_queue())

    async def stop_streaming(self):
        """Stop log streaming."""
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass

    async def _process_log_queue(self):
        """Process log queue for streaming and persistence."""
        while True:
            try:
                # Get log entry from queue
                log_entry = await self.log_queue.get()

                # Stream to Redis if enabled
                if self.redis_client:
                    await self._stream_to_redis(log_entry)

                # Persist to database if enabled
                if self.postgres_engine:
                    await self._persist_to_db(log_entry)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Log processing error: {e}")

    async def _stream_to_redis(self, log_entry: LogEntry):
        """Stream log entry to Redis."""
        try:
            log_data = asdict(log_entry)
            log_data['timestamp'] = log_entry.timestamp.isoformat()
            log_data['event'] = log_entry.event.value

            # Stream to service-specific channel
            channel = f"logs:{self.service_name}"
            self.redis_client.lpush(channel, json.dumps(log_data))

            # Keep only last 10000 logs per service
            self.redis_client.ltrim(channel, 0, 9999)

            # Stream to global log channel
            self.redis_client.lpush("logs:global", json.dumps(log_data))
            self.redis_client.ltrim("logs:global", 0, 99999)

        except Exception as e:
            print(f"Redis streaming error: {e}")

    async def _persist_to_db(self, log_entry: LogEntry):
        """Persist log entry to database."""
        try:
            with self.postgres_engine.connect() as conn:
                conn.execute("""
                    INSERT INTO logs
                    (timestamp, level, event, service, component, message,
                     context, trace_id, span_id, correlation_id, user_id, session_id, request_id)
                    VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    log_entry.timestamp,
                    log_entry.level,
                    log_entry.event.value,
                    log_entry.service,
                    log_entry.component,
                    log_entry.message,
                    json.dumps(log_entry.context),
                    log_entry.trace_id,
                    log_entry.span_id,
                    log_entry.correlation_id,
                    log_entry.user_id,
                    log_entry.session_id,
                    log_entry.request_id
                ))
                conn.commit()

        except Exception as e:
            print(f"Database persistence error: {e}")


class MetricsCollector:
    """
    Prometheus metrics collector for auto-recovery system.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector."""
        self.registry = registry or CollectorRegistry()

        # Auto-recovery metrics
        self.recovery_attempts = Counter(
            'recovery_attempts_total',
            'Total recovery attempts',
            ['service', 'strategy', 'result'],
            registry=self.registry
        )

        self.recovery_duration = Histogram(
            'recovery_duration_seconds',
            'Recovery operation duration',
            ['service', 'strategy'],
            registry=self.registry
        )

        self.service_health = Gauge(
            'service_health_status',
            'Service health status (1=healthy, 0=unhealthy)',
            ['service'],
            registry=self.registry
        )

        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=half-open, 2=open)',
            ['service'],
            registry=self.registry
        )

        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            ['service'],
            registry=self.registry
        )

        self.resource_usage = Gauge(
            'resource_usage_percent',
            'Resource usage percentage',
            ['service', 'resource'],
            registry=self.registry
        )

        self.uptime = Counter(
            'uptime_seconds_total',
            'Service uptime in seconds',
            ['service'],
            registry=self.registry
        )

        self.error_rate = Gauge(
            'error_rate',
            'Error rate percentage',
            ['service'],
            registry=self.registry
        )

    def record_recovery_attempt(self, service: str, strategy: str, result: str, duration: float):
        """Record a recovery attempt."""
        self.recovery_attempts.labels(service=service, strategy=strategy, result=result).inc()
        self.recovery_duration.labels(service=service, strategy=strategy).observe(duration)

    def update_service_health(self, service: str, healthy: bool):
        """Update service health status."""
        self.service_health.labels(service=service).set(1.0 if healthy else 0.0)

    def update_circuit_breaker_state(self, service: str, state: str):
        """Update circuit breaker state."""
        state_map = {'closed': 0, 'half_open': 1, 'open': 2}
        self.circuit_breaker_state.labels(service=service).set(state_map.get(state, 0))

    def update_resource_usage(self, service: str, resource: str, percentage: float):
        """Update resource usage."""
        self.resource_usage.labels(service=service, resource=resource).set(percentage)

    def update_error_rate(self, service: str, rate: float):
        """Update error rate."""
        self.error_rate.labels(service=service).set(rate)

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')


# Integration class
class LoggingAlertingSystem:
    """
    Integrated logging and alerting system for auto-recovery.
    """

    def __init__(self,
                 service_name: str,
                 redis_url: str = "redis://redis:6379/12",
                 postgres_url: str = None,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize logging and alerting system."""
        self.service_name = service_name
        self.config = config or {}

        # Initialize connections
        self.redis_client = redis.from_url(redis_url) if redis_url else None
        self.postgres_engine = None
        if postgres_url:
            from sqlalchemy import create_engine
            self.postgres_engine = create_engine(postgres_url)

        # Initialize components
        self.alert_manager = AlertManager(
            redis_client=self.redis_client,
            postgres_engine=self.postgres_engine,
            config=self.config.get('alerting', {})
        )

        self.logger = StructuredLogger(
            service_name=service_name,
            redis_client=self.redis_client,
            postgres_engine=self.postgres_engine,
            config=self.config.get('logging', {})
        )

        self.metrics = MetricsCollector()

    async def start(self):
        """Start logging and alerting system."""
        await self.logger.start_streaming()

    async def stop(self):
        """Stop logging and alerting system."""
        await self.logger.stop_streaming()

    async def log_and_alert(self,
                           level: AlertLevel,
                           event: LogEvent,
                           title: str,
                           message: str,
                           component: str = "",
                           channels: List[AlertChannel] = None,
                           **context):
        """Log an event and send alert if necessary."""
        # Log the event
        await self.logger.log_event(
            level=level.value,
            event=event,
            message=message,
            component=component,
            **context
        )

        # Send alert for warning and above
        if level.value in ['warning', 'error', 'critical', 'emergency'] and channels:
            alert = Alert(
                level=level,
                title=title,
                message=message,
                service=self.service_name,
                component=component,
                channels=channels,
                tags=context.get('tags', {}),
                metadata=context
            )

            await self.alert_manager.send_alert(alert)

    def get_metrics(self) -> str:
        """Get Prometheus metrics."""
        return self.metrics.get_metrics()