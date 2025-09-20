import os
"""
BEV OSINT Framework - Advanced Alert System
Configurable alerting with multiple severity levels and notification channels.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import aioredis
import asyncpg
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml
import jinja2
from prometheus_client import Counter, Gauge


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertState(Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    TELEGRAM = "telegram"
    SMS = "sms"


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne", "contains"
    threshold: Union[float, int, str]
    severity: AlertSeverity
    duration: int  # Duration in seconds before triggering
    labels: Dict[str, str]
    annotations: Dict[str, str]
    enabled: bool = True
    notification_channels: List[NotificationChannel] = None
    escalation_rules: List[Dict[str, Any]] = None
    inhibit_rules: List[str] = None  # Rules that suppress this alert


@dataclass
class Alert:
    """Active alert instance."""
    id: str
    rule_name: str
    service_name: str
    metric_name: str
    current_value: Union[float, int, str]
    threshold: Union[float, int, str]
    severity: AlertSeverity
    state: AlertState
    started_at: datetime
    last_updated: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    message: str = ""
    labels: Dict[str, str] = None
    annotations: Dict[str, str] = None
    notification_count: int = 0
    last_notification: Optional[datetime] = None


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    channel: NotificationChannel
    enabled: bool
    config: Dict[str, Any]
    rate_limit: Optional[int] = None  # Minutes between notifications
    severity_filter: List[AlertSeverity] = None


class AlertSystem:
    """
    Advanced alert system with configurable thresholds, multiple notification
    channels, and intelligent escalation.
    """

    def __init__(self, config_path: str = "/app/config/alert_system.yml"):
        self.config = self._load_config(config_path)
        self.alert_rules = self._load_alert_rules()
        self.notification_configs = self._load_notification_configs()

        # Alert state
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.suppressed_alerts: Dict[str, datetime] = {}

        # External connections
        self.redis_client: Optional[aioredis.Redis] = None
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self.session: Optional[aiohttp.ClientSession] = None

        # Notification templates
        self.template_env = jinja2.Environment(
            loader=jinja2.DictLoader(self._get_default_templates())
        )

        # Prometheus metrics
        self._init_prometheus_metrics()

        # Background tasks
        self.alert_tasks: Dict[str, asyncio.Task] = {}

        self.logger = self._setup_logging()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load alert system configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for alert system."""
        return {
            "global": {
                "evaluation_interval": 30,
                "notification_rate_limit": 5,  # minutes
                "alert_retention_days": 30,
                "enable_inhibition": True,
                "enable_escalation": True
            },
            "templates": {
                "email_subject": "BEV Alert: {{ alert.severity.value.upper() }} - {{ alert.service_name }}",
                "email_body": "alert_email.html",
                "webhook_payload": "webhook.json",
                "slack_message": "slack.json"
            },
            "escalation": {
                "levels": [
                    {"duration": 300, "channels": ["email"]},  # 5 minutes
                    {"duration": 900, "channels": ["email", "slack"]},  # 15 minutes
                    {"duration": 1800, "channels": ["email", "slack", "webhook"]}  # 30 minutes
                ]
            }
        }

    def _load_alert_rules(self) -> Dict[str, AlertRule]:
        """Load alert rules configuration."""
        rules = {}

        # Default alert rules
        default_rules = [
            AlertRule(
                name="high_response_time",
                metric_name="service_response_time",
                condition="gt",
                threshold=5.0,
                severity=AlertSeverity.WARNING,
                duration=120,
                labels={"component": "performance"},
                annotations={
                    "summary": "High response time detected",
                    "description": "Service {{ $labels.service }} response time is {{ $value }}s"
                }
            ),
            AlertRule(
                name="critical_response_time",
                metric_name="service_response_time",
                condition="gt",
                threshold=10.0,
                severity=AlertSeverity.CRITICAL,
                duration=60,
                labels={"component": "performance"},
                annotations={
                    "summary": "Critical response time detected",
                    "description": "Service {{ $labels.service }} response time is {{ $value }}s"
                }
            ),
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu_usage_percent",
                condition="gt",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration=300,
                labels={"component": "resources"},
                annotations={
                    "summary": "High CPU usage detected",
                    "description": "Service {{ $labels.service }} CPU usage is {{ $value }}%"
                }
            ),
            AlertRule(
                name="critical_cpu_usage",
                metric_name="cpu_usage_percent",
                condition="gt",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                duration=120,
                labels={"component": "resources"},
                annotations={
                    "summary": "Critical CPU usage detected",
                    "description": "Service {{ $labels.service }} CPU usage is {{ $value }}%"
                }
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="memory_usage_percent",
                condition="gt",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
                duration=300,
                labels={"component": "resources"},
                annotations={
                    "summary": "High memory usage detected",
                    "description": "Service {{ $labels.service }} memory usage is {{ $value }}%"
                }
            ),
            AlertRule(
                name="critical_memory_usage",
                metric_name="memory_usage_percent",
                condition="gt",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                duration=120,
                labels={"component": "resources"},
                annotations={
                    "summary": "Critical memory usage detected",
                    "description": "Service {{ $labels.service }} memory usage is {{ $value }}%"
                }
            ),
            AlertRule(
                name="service_down",
                metric_name="service_up",
                condition="eq",
                threshold=0,
                severity=AlertSeverity.CRITICAL,
                duration=60,
                labels={"component": "availability"},
                annotations={
                    "summary": "Service is down",
                    "description": "Service {{ $labels.service }} is not responding"
                }
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="error_rate",
                condition="gt",
                threshold=5.0,
                severity=AlertSeverity.WARNING,
                duration=180,
                labels={"component": "reliability"},
                annotations={
                    "summary": "High error rate detected",
                    "description": "Service {{ $labels.service }} error rate is {{ $value }}%"
                }
            ),
            AlertRule(
                name="critical_error_rate",
                metric_name="error_rate",
                condition="gt",
                threshold=15.0,
                severity=AlertSeverity.CRITICAL,
                duration=120,
                labels={"component": "reliability"},
                annotations={
                    "summary": "Critical error rate detected",
                    "description": "Service {{ $labels.service }} error rate is {{ $value }}%"
                }
            ),
            AlertRule(
                name="disk_space_low",
                metric_name="disk_usage_percent",
                condition="gt",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration=300,
                labels={"component": "storage"},
                annotations={
                    "summary": "Low disk space",
                    "description": "Service {{ $labels.service }} disk usage is {{ $value }}%"
                }
            ),
            AlertRule(
                name="disk_space_critical",
                metric_name="disk_usage_percent",
                condition="gt",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                duration=120,
                labels={"component": "storage"},
                annotations={
                    "summary": "Critical disk space",
                    "description": "Service {{ $labels.service }} disk usage is {{ $value }}%"
                }
            )
        ]

        for rule in default_rules:
            rules[rule.name] = rule

        return rules

    def _load_notification_configs(self) -> Dict[str, NotificationConfig]:
        """Load notification channel configurations."""
        configs = {}

        # Email configuration
        configs["email"] = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            enabled=True,
            config={
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "${EMAIL_USERNAME}",
                "password": "${EMAIL_PASSWORD}",
                "from_email": "bev-alerts@company.com",
                "to_emails": ["admin@company.com", "ops@company.com"],
                "use_tls": True
            },
            rate_limit=5,
            severity_filter=[AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        )

        # Webhook configuration
        configs["webhook"] = NotificationConfig(
            channel=NotificationChannel.WEBHOOK,
            enabled=True,
            config={
                "url": "${WEBHOOK_URL}",
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer ${WEBHOOK_TOKEN}"
                },
                "timeout": 10
            },
            rate_limit=1,
            severity_filter=[AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        )

        # Slack configuration
        configs["slack"] = NotificationConfig(
            channel=NotificationChannel.SLACK,
            enabled=True,
            config={
                "webhook_url": "${SLACK_WEBHOOK_URL}",
                "channel": "#bev-alerts",
                "username": "BEV Alert Bot",
                "icon_emoji": ":warning:"
            },
            rate_limit=2,
            severity_filter=[AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        )

        return configs

    def _get_default_templates(self) -> Dict[str, str]:
        """Get default notification templates."""
        return {
            "alert_email.html": """
            <html>
            <body>
                <h2 style="color: {% if alert.severity == 'critical' %}red{% elif alert.severity == 'warning' %}orange{% else %}blue{% endif %};">
                    BEV Alert: {{ alert.severity.value.upper() }}
                </h2>

                <table style="border-collapse: collapse; width: 100%;">
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Service:</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{{ alert.service_name }}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Metric:</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{{ alert.metric_name }}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Current Value:</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{{ alert.current_value }}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Threshold:</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{{ alert.threshold }}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Started:</strong></td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{{ alert.started_at.strftime('%Y-%m-%d %H:%M:%S UTC') }}</td>
                    </tr>
                </table>

                <p><strong>Message:</strong> {{ alert.message }}</p>

                {% if alert.labels %}
                <p><strong>Labels:</strong></p>
                <ul>
                {% for key, value in alert.labels.items() %}
                    <li>{{ key }}: {{ value }}</li>
                {% endfor %}
                </ul>
                {% endif %}

                <p style="color: #666; font-size: 12px;">
                    Alert ID: {{ alert.id }}<br>
                    Generated by BEV OSINT Framework
                </p>
            </body>
            </html>
            """,

            "webhook.json": """
            {
                "alert_id": "{{ alert.id }}",
                "rule_name": "{{ alert.rule_name }}",
                "service_name": "{{ alert.service_name }}",
                "metric_name": "{{ alert.metric_name }}",
                "severity": "{{ alert.severity.value }}",
                "state": "{{ alert.state.value }}",
                "current_value": {{ alert.current_value }},
                "threshold": {{ alert.threshold }},
                "started_at": "{{ alert.started_at.isoformat() }}",
                "message": "{{ alert.message }}",
                "labels": {{ alert.labels | tojson if alert.labels else {} }},
                "annotations": {{ alert.annotations | tojson if alert.annotations else {} }}
            }
            """,

            "slack.json": """
            {
                "channel": "{{ config.channel }}",
                "username": "{{ config.username }}",
                "icon_emoji": "{{ config.icon_emoji }}",
                "attachments": [
                    {
                        "color": "{% if alert.severity.value == 'critical' %}danger{% elif alert.severity.value == 'warning' %}warning{% else %}good{% endif %}",
                        "title": "{{ alert.severity.value.upper() }}: {{ alert.service_name }}",
                        "text": "{{ alert.message }}",
                        "fields": [
                            {
                                "title": "Metric",
                                "value": "{{ alert.metric_name }}",
                                "short": true
                            },
                            {
                                "title": "Current Value",
                                "value": "{{ alert.current_value }}",
                                "short": true
                            },
                            {
                                "title": "Threshold",
                                "value": "{{ alert.threshold }}",
                                "short": true
                            },
                            {
                                "title": "Started",
                                "value": "{{ alert.started_at.strftime('%Y-%m-%d %H:%M:%S UTC') }}",
                                "short": true
                            }
                        ],
                        "footer": "BEV OSINT Framework",
                        "ts": {{ alert.started_at.timestamp() | int }}
                    }
                ]
            }
            """
        }

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics for alert system."""
        self.prom_alerts_total = Counter(
            'bev_alerts_total',
            'Total number of alerts triggered',
            ['service', 'severity', 'rule']
        )
        self.prom_active_alerts = Gauge(
            'bev_active_alerts',
            'Number of active alerts',
            ['service', 'severity']
        )
        self.prom_notifications_sent = Counter(
            'bev_notifications_sent_total',
            'Total notifications sent',
            ['channel', 'severity']
        )
        self.prom_notification_failures = Counter(
            'bev_notification_failures_total',
            'Total notification failures',
            ['channel', 'error_type']
        )

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for alert system."""
        logger = logging.getLogger('alert_system')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    async def initialize(self):
        """Initialize external connections and alert system."""
        try:
            # Initialize Redis connection
            self.redis_client = await aioredis.from_url(
                "redis://redis:6379",
                db=3,
                decode_responses=True
            )

            # Initialize PostgreSQL connection pool
            self.postgres_pool = await asyncpg.create_pool(
                host="postgres",
                port=5432,
                database="osint",
                user="bev",
                password=os.getenv('DB_PASSWORD', 'dev_password'),
                min_size=2,
                max_size=5
            )

            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )

            # Create database tables
            await self._ensure_database_schema()

            # Load existing alerts from storage
            await self._load_alert_state()

            self.logger.info("Alert system initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize alert system: {e}")
            raise

    async def _ensure_database_schema(self):
        """Ensure database schema exists for alert storage."""
        if self.postgres_pool:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id VARCHAR(255) PRIMARY KEY,
                        rule_name VARCHAR(255) NOT NULL,
                        service_name VARCHAR(255) NOT NULL,
                        metric_name VARCHAR(255) NOT NULL,
                        current_value TEXT NOT NULL,
                        threshold TEXT NOT NULL,
                        severity VARCHAR(50) NOT NULL,
                        state VARCHAR(50) NOT NULL,
                        started_at TIMESTAMPTZ NOT NULL,
                        last_updated TIMESTAMPTZ NOT NULL,
                        resolved_at TIMESTAMPTZ,
                        acknowledged_at TIMESTAMPTZ,
                        acknowledged_by VARCHAR(255),
                        message TEXT,
                        labels JSONB,
                        annotations JSONB,
                        notification_count INTEGER DEFAULT 0,
                        last_notification TIMESTAMPTZ,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alerts_service_severity
                    ON alerts(service_name, severity, state)
                """)

                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alerts_started_at
                    ON alerts(started_at)
                """)

    async def _load_alert_state(self):
        """Load existing alert state from storage."""
        if self.postgres_pool:
            try:
                async with self.postgres_pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT * FROM alerts WHERE state = 'active'
                    """)

                    for row in rows:
                        alert = Alert(
                            id=row['id'],
                            rule_name=row['rule_name'],
                            service_name=row['service_name'],
                            metric_name=row['metric_name'],
                            current_value=json.loads(row['current_value']),
                            threshold=json.loads(row['threshold']),
                            severity=AlertSeverity(row['severity']),
                            state=AlertState(row['state']),
                            started_at=row['started_at'],
                            last_updated=row['last_updated'],
                            resolved_at=row['resolved_at'],
                            acknowledged_at=row['acknowledged_at'],
                            acknowledged_by=row['acknowledged_by'],
                            message=row['message'],
                            labels=json.loads(row['labels']) if row['labels'] else {},
                            annotations=json.loads(row['annotations']) if row['annotations'] else {},
                            notification_count=row['notification_count'],
                            last_notification=row['last_notification']
                        )

                        self.active_alerts[alert.id] = alert

                self.logger.info(f"Loaded {len(self.active_alerts)} active alerts from storage")

            except Exception as e:
                self.logger.error(f"Error loading alert state: {e}")

    async def evaluate_metric(self, metric_name: str, value: Union[float, int, str],
                            labels: Dict[str, str], service_name: str) -> List[Alert]:
        """
        Evaluate metric against alert rules and trigger alerts if conditions are met.

        Args:
            metric_name: Name of the metric
            value: Current metric value
            labels: Metric labels
            service_name: Source service name

        Returns:
            List[Alert]: New alerts triggered
        """
        triggered_alerts = []

        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled or rule.metric_name != metric_name:
                continue

            # Check if rule condition is met
            if self._evaluate_condition(value, rule.condition, rule.threshold):
                alert_id = self._generate_alert_id(rule, service_name, labels)

                # Check if alert already exists
                if alert_id in self.active_alerts:
                    # Update existing alert
                    alert = self.active_alerts[alert_id]
                    alert.current_value = value
                    alert.last_updated = datetime.now(timezone.utc)
                    await self._update_alert_in_storage(alert)
                else:
                    # Create new alert
                    alert = Alert(
                        id=alert_id,
                        rule_name=rule_name,
                        service_name=service_name,
                        metric_name=metric_name,
                        current_value=value,
                        threshold=rule.threshold,
                        severity=rule.severity,
                        state=AlertState.ACTIVE,
                        started_at=datetime.now(timezone.utc),
                        last_updated=datetime.now(timezone.utc),
                        message=self._generate_alert_message(rule, value, labels),
                        labels=labels,
                        annotations=rule.annotations
                    )

                    self.active_alerts[alert_id] = alert
                    triggered_alerts.append(alert)

                    # Store in database
                    await self._store_alert(alert)

                    # Update Prometheus metrics
                    self.prom_alerts_total.labels(
                        service=service_name,
                        severity=rule.severity.value,
                        rule=rule_name
                    ).inc()

                    self.prom_active_alerts.labels(
                        service=service_name,
                        severity=rule.severity.value
                    ).inc()

                    # Send notifications
                    await self._send_alert_notifications(alert)

                    self.logger.info(
                        f"Alert triggered: {rule_name} for {service_name} "
                        f"({metric_name}={value} > {rule.threshold})"
                    )

            else:
                # Check if we should resolve existing alert
                alert_id = self._generate_alert_id(rule, service_name, labels)
                if alert_id in self.active_alerts:
                    await self._resolve_alert(alert_id)

        return triggered_alerts

    def _evaluate_condition(self, value: Union[float, int, str], condition: str,
                          threshold: Union[float, int, str]) -> bool:
        """Evaluate alert condition."""
        try:
            if condition == "gt":
                return float(value) > float(threshold)
            elif condition == "lt":
                return float(value) < float(threshold)
            elif condition == "eq":
                return value == threshold
            elif condition == "ne":
                return value != threshold
            elif condition == "contains":
                return str(threshold).lower() in str(value).lower()
            else:
                self.logger.warning(f"Unknown condition: {condition}")
                return False
        except (ValueError, TypeError) as e:
            self.logger.error(f"Error evaluating condition {condition}: {e}")
            return False

    def _generate_alert_id(self, rule: AlertRule, service_name: str,
                          labels: Dict[str, str]) -> str:
        """Generate unique alert ID."""
        label_str = "_".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{rule.name}_{service_name}_{label_str}".replace(" ", "_")

    def _generate_alert_message(self, rule: AlertRule, value: Union[float, int, str],
                              labels: Dict[str, str]) -> str:
        """Generate alert message from rule template."""
        template_str = rule.annotations.get("description", rule.annotations.get("summary", "Alert triggered"))

        # Simple template substitution
        message = template_str.replace("{{ $value }}", str(value))
        for key, val in labels.items():
            message = message.replace(f"{{{{ $labels.{key} }}}}", str(val))

        return message

    async def _send_alert_notifications(self, alert: Alert):
        """Send notifications for alert through configured channels."""
        rule = self.alert_rules.get(alert.rule_name)
        if not rule or not rule.notification_channels:
            # Use default channels based on severity
            channels = self._get_default_channels_for_severity(alert.severity)
        else:
            channels = rule.notification_channels

        for channel in channels:
            if channel.value in self.notification_configs:
                config = self.notification_configs[channel.value]

                # Check if channel is enabled and severity filter
                if not config.enabled:
                    continue

                if config.severity_filter and alert.severity not in config.severity_filter:
                    continue

                # Check rate limiting
                if await self._is_rate_limited(alert, config):
                    continue

                try:
                    success = await self._send_notification(alert, config)
                    if success:
                        alert.notification_count += 1
                        alert.last_notification = datetime.now(timezone.utc)

                        self.prom_notifications_sent.labels(
                            channel=channel.value,
                            severity=alert.severity.value
                        ).inc()

                        self.logger.info(f"Notification sent via {channel.value} for alert {alert.id}")
                    else:
                        self.prom_notification_failures.labels(
                            channel=channel.value,
                            error_type="send_failed"
                        ).inc()

                except Exception as e:
                    self.logger.error(f"Error sending notification via {channel.value}: {e}")
                    self.prom_notification_failures.labels(
                        channel=channel.value,
                        error_type="exception"
                    ).inc()

    def _get_default_channels_for_severity(self, severity: AlertSeverity) -> List[NotificationChannel]:
        """Get default notification channels based on severity."""
        if severity == AlertSeverity.EMERGENCY:
            return [NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.WEBHOOK]
        elif severity == AlertSeverity.CRITICAL:
            return [NotificationChannel.EMAIL, NotificationChannel.SLACK]
        elif severity == AlertSeverity.WARNING:
            return [NotificationChannel.EMAIL]
        else:
            return []

    async def _is_rate_limited(self, alert: Alert, config: NotificationConfig) -> bool:
        """Check if notification is rate limited."""
        if not config.rate_limit:
            return False

        if not alert.last_notification:
            return False

        time_since_last = datetime.now(timezone.utc) - alert.last_notification
        return time_since_last.total_seconds() < (config.rate_limit * 60)

    async def _send_notification(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send notification through specific channel."""
        try:
            if config.channel == NotificationChannel.EMAIL:
                return await self._send_email_notification(alert, config)
            elif config.channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook_notification(alert, config)
            elif config.channel == NotificationChannel.SLACK:
                return await self._send_slack_notification(alert, config)
            else:
                self.logger.warning(f"Notification channel {config.channel} not implemented")
                return False

        except Exception as e:
            self.logger.error(f"Error in notification channel {config.channel}: {e}")
            return False

    async def _send_email_notification(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send email notification."""
        try:
            smtp_config = config.config

            # Render email template
            subject_template = self.template_env.from_string(
                self.config["templates"]["email_subject"]
            )
            subject = subject_template.render(alert=alert)

            body_template = self.template_env.get_template("alert_email.html")
            body = body_template.render(alert=alert)

            # Create message
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = ', '.join(smtp_config['to_emails'])
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'html'))

            # Send email
            server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
            if smtp_config.get('use_tls'):
                server.starttls()

            if smtp_config.get('username') and smtp_config.get('password'):
                server.login(smtp_config['username'], smtp_config['password'])

            server.send_message(msg)
            server.quit()

            return True

        except Exception as e:
            self.logger.error(f"Error sending email notification: {e}")
            return False

    async def _send_webhook_notification(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send webhook notification."""
        try:
            webhook_config = config.config

            # Render webhook payload
            payload_template = self.template_env.get_template("webhook.json")
            payload_str = payload_template.render(alert=alert)
            payload = json.loads(payload_str)

            # Send webhook
            async with self.session.post(
                webhook_config['url'],
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=webhook_config.get('timeout', 10)
            ) as response:
                return response.status < 400

        except Exception as e:
            self.logger.error(f"Error sending webhook notification: {e}")
            return False

    async def _send_slack_notification(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send Slack notification."""
        try:
            slack_config = config.config

            # Render Slack message
            message_template = self.template_env.get_template("slack.json")
            message_str = message_template.render(alert=alert, config=slack_config)
            message = json.loads(message_str)

            # Send to Slack
            async with self.session.post(
                slack_config['webhook_url'],
                json=message
            ) as response:
                return response.status < 400

        except Exception as e:
            self.logger.error(f"Error sending Slack notification: {e}")
            return False

    async def _store_alert(self, alert: Alert):
        """Store alert in database."""
        if self.postgres_pool:
            try:
                async with self.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO alerts (
                            id, rule_name, service_name, metric_name, current_value,
                            threshold, severity, state, started_at, last_updated,
                            message, labels, annotations, notification_count
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    """,
                    alert.id, alert.rule_name, alert.service_name, alert.metric_name,
                    json.dumps(alert.current_value), json.dumps(alert.threshold),
                    alert.severity.value, alert.state.value, alert.started_at,
                    alert.last_updated, alert.message, json.dumps(alert.labels),
                    json.dumps(alert.annotations), alert.notification_count
                    )

            except Exception as e:
                self.logger.error(f"Error storing alert: {e}")

    async def _update_alert_in_storage(self, alert: Alert):
        """Update existing alert in database."""
        if self.postgres_pool:
            try:
                async with self.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE alerts SET
                            current_value = $1,
                            state = $2,
                            last_updated = $3,
                            resolved_at = $4,
                            acknowledged_at = $5,
                            acknowledged_by = $6,
                            notification_count = $7,
                            last_notification = $8
                        WHERE id = $9
                    """,
                    json.dumps(alert.current_value), alert.state.value,
                    alert.last_updated, alert.resolved_at, alert.acknowledged_at,
                    alert.acknowledged_by, alert.notification_count,
                    alert.last_notification, alert.id
                    )

            except Exception as e:
                self.logger.error(f"Error updating alert: {e}")

    async def _resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.state = AlertState.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc)
            alert.last_updated = datetime.now(timezone.utc)

            # Update storage
            await self._update_alert_in_storage(alert)

            # Update Prometheus metrics
            self.prom_active_alerts.labels(
                service=alert.service_name,
                severity=alert.severity.value
            ).dec()

            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]

            self.logger.info(f"Alert resolved: {alert.rule_name} for {alert.service_name}")

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.state = AlertState.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now(timezone.utc)
            alert.acknowledged_by = acknowledged_by
            alert.last_updated = datetime.now(timezone.utc)

            await self._update_alert_in_storage(alert)

            self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True

        return False

    async def get_active_alerts(self, service_name: Optional[str] = None,
                              severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts with optional filtering."""
        alerts = list(self.active_alerts.values())

        if service_name:
            alerts = [a for a in alerts if a.service_name == service_name]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda x: x.started_at, reverse=True)

    async def get_alert_history(self, hours: int = 24, service_name: Optional[str] = None) -> List[Alert]:
        """Get alert history."""
        if self.postgres_pool:
            try:
                async with self.postgres_pool.acquire() as conn:
                    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

                    query = "SELECT * FROM alerts WHERE started_at >= $1"
                    params = [cutoff_time]

                    if service_name:
                        query += " AND service_name = $2"
                        params.append(service_name)

                    query += " ORDER BY started_at DESC LIMIT 1000"

                    rows = await conn.fetch(query, *params)

                    return [
                        Alert(
                            id=row['id'],
                            rule_name=row['rule_name'],
                            service_name=row['service_name'],
                            metric_name=row['metric_name'],
                            current_value=json.loads(row['current_value']),
                            threshold=json.loads(row['threshold']),
                            severity=AlertSeverity(row['severity']),
                            state=AlertState(row['state']),
                            started_at=row['started_at'],
                            last_updated=row['last_updated'],
                            resolved_at=row['resolved_at'],
                            acknowledged_at=row['acknowledged_at'],
                            acknowledged_by=row['acknowledged_by'],
                            message=row['message'],
                            labels=json.loads(row['labels']) if row['labels'] else {},
                            annotations=json.loads(row['annotations']) if row['annotations'] else {},
                            notification_count=row['notification_count'],
                            last_notification=row['last_notification']
                        )
                        for row in rows
                    ]

            except Exception as e:
                self.logger.error(f"Error getting alert history: {e}")

        return []

    async def cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        try:
            if self.postgres_pool:
                retention_days = self.config["global"]["alert_retention_days"]
                cutoff_time = datetime.now(timezone.utc) - timedelta(days=retention_days)

                async with self.postgres_pool.acquire() as conn:
                    result = await conn.execute("""
                        DELETE FROM alerts
                        WHERE state IN ('resolved', 'acknowledged')
                        AND (resolved_at < $1 OR acknowledged_at < $1)
                    """, cutoff_time)

                    self.logger.info(f"Cleaned up old alerts: {result}")

        except Exception as e:
            self.logger.error(f"Error cleaning up old alerts: {e}")

    async def start_alert_evaluation_loop(self):
        """Start background alert evaluation loop."""
        self.alert_tasks["evaluation"] = asyncio.create_task(
            self._alert_evaluation_loop()
        )

        self.alert_tasks["cleanup"] = asyncio.create_task(
            self._cleanup_loop()
        )

        self.logger.info("Alert evaluation loops started")

    async def _alert_evaluation_loop(self):
        """Background loop for alert evaluation."""
        while True:
            try:
                # This would typically be called by the metrics collector
                # when new metrics are received. For now, just sleep.
                await asyncio.sleep(self.config["global"]["evaluation_interval"])

            except Exception as e:
                self.logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(60)

    async def _cleanup_loop(self):
        """Background loop for cleanup tasks."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_old_alerts()

            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)

    async def shutdown(self):
        """Graceful shutdown of alert system."""
        # Cancel all tasks
        for task_name, task in self.alert_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                self.logger.info(f"Cancelled {task_name} task")

        # Close connections
        if self.session:
            await self.session.close()
        if self.redis_client:
            await self.redis_client.close()
        if self.postgres_pool:
            await self.postgres_pool.close()

        self.logger.info("Alert system shutdown completed")


async def main():
    """Main entry point for alert system."""
    alert_system = AlertSystem()

    try:
        await alert_system.initialize()
        await alert_system.start_alert_evaluation_loop()

        # Keep running
        while True:
            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        await alert_system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())