# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import atexit
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from relax.utils.logging_utils import get_logger
from relax.utils.misc import SingletonMeta


logger = get_logger(__name__)


class _AppriseAdapter(metaclass=SingletonMeta):
    """Apprise notification adapter for training metrics and alerts.

    This adapter sends training metrics and alerts to various notification services
    using the Apprise library. It supports:
    - Startup notifications
    - Periodic metric updates with delta tracking
    - Completion notifications
    - Failure alerts (via atexit handler)
    - Custom alerts

    Example:
        args.notify_urls = "redcity://webhook_key?msgtype=markdown&freq=10"
        adapter = _AppriseAdapter(args)
        adapter.log({"rollout/raw_reward": 0.75}, step=100)
        adapter.finish()
    """

    def __init__(self, args):
        self.notify_urls_raw = self._parse_notify_urls(args)
        if not self.notify_urls_raw:
            logger.warning("No notify URLs provided, Apprise adapter will not send notifications")
            self.apprise = None
            return

        # Parse frequency parameter and clean URLs
        self.notify_freq, self.notify_urls_clean = self._parse_freq_and_clean_urls(self.notify_urls_raw)

        # Track log notification count for frequency control
        self.log_count = 0

        try:
            from apprise import Apprise

            self.apprise = Apprise()

            # Add cleaned notification URLs (without freq parameter)
            for url in self.notify_urls_clean:
                if self.apprise.add(url):
                    logger.info(f"Added notification URL: {url}")
                else:
                    logger.warning(f"Failed to add notification URL: {url}")

            if self.notify_freq > 1:
                logger.info(
                    f"Notification frequency control enabled: send 1 out of every {self.notify_freq} log events"
                )

            # Track previous metrics for delta calculation
            self.prev_metrics: Dict[str, float] = {}

            # Register exit handler for failure alerts
            atexit.register(self._on_exit)
            self._normal_exit = False

            # Send startup notification (not affected by freq)
            self._send_startup_notification(args)

        except ImportError:
            logger.error("Apprise is not installed. Please install it with: pip install apprise")
            self.apprise = None
        except Exception as e:
            logger.error(f"Failed to initialize Apprise: {e}")
            self.apprise = None

    def _parse_notify_urls(self, args) -> list:
        """Parse notify URLs from arguments."""
        notify_urls_str = args.notify_urls
        if not notify_urls_str:
            return []

        # Split by comma and strip whitespace
        urls = [url.strip() for url in notify_urls_str.split(",") if url.strip()]
        return urls

    def _parse_freq_and_clean_urls(self, urls: list) -> tuple[int, list]:
        """Parse freq parameter from URLs and return cleaned URLs.

        Args:
            urls: List of notification URLs

        Returns:
            Tuple of (frequency, cleaned_urls)
            - frequency: The freq value from the first URL that has it, default 1
            - cleaned_urls: URLs with freq parameter removed
        """
        freq = 1
        cleaned_urls = []

        for url in urls:
            try:
                # Parse URL
                parsed = urlparse(url)

                # Parse query parameters
                query_params = parse_qs(parsed.query)

                # Extract freq parameter if present
                if "freq" in query_params:
                    try:
                        url_freq = int(query_params["freq"][0])
                        if url_freq > 0:
                            # Use the first valid freq value found
                            if freq == 1:
                                freq = url_freq
                            # Remove freq from query params
                            del query_params["freq"]
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid freq parameter in URL: {url}")

                # Rebuild query string without freq
                if query_params:
                    query_parts = []
                    for key, values in query_params.items():
                        for value in values:
                            query_parts.append(f"{key}={value}")
                    new_query = "&".join(query_parts)
                else:
                    new_query = ""

                # Rebuild URL
                if new_query:
                    cleaned_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"
                else:
                    cleaned_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

                # Add fragment if present
                if parsed.fragment:
                    cleaned_url += f"#{parsed.fragment}"

                cleaned_urls.append(cleaned_url)

            except Exception as e:
                logger.warning(f"Failed to parse URL {url}: {e}, using as-is")
                cleaned_urls.append(url)

        return freq, cleaned_urls

    def _send_startup_notification(self, args):
        """Send a notification when training starts."""
        if not self.apprise:
            return

        project_name = args.tb_project_name
        experiment_name = args.tb_experiment_name

        markdown = f"""# 🚀 训练启动通知

**项目**: {project_name}
**实验**: {experiment_name}
**状态**: 已启动

---
*训练指标将定期更新*
"""

        try:
            self.apprise.notify(body=markdown)
            logger.info("Sent startup notification")
        except Exception as e:
            logger.error(f"Failed to send startup notification: {e}")

    def log(self, data: Dict[str, Any], step: int):
        """Log metrics and send notification with markdown report.

        Frequency control is applied to log notifications only.

        Args:
            data: Dictionary of metrics to log
            step: Current training step
        """
        if not self.apprise:
            return

        # Increment log count
        self.log_count += 1

        # Apply frequency control: only send notification every N calls
        if self.log_count % self.notify_freq != 0:
            logger.debug(
                f"Skipping notification for step {step} (log_count={self.log_count}, freq={self.notify_freq})"
            )
            # Still update metrics for delta calculation
            self._update_metrics(data)
            return

        # Define key metrics to track
        key_metrics = [
            "rollout/raw_reward",
            "rollout/reward",
            "train/grad_norm",
            "train/entropy_loss",
            "train/policy_loss",
            "train/value_loss",
            "train/learning_rate",
            "train/kl_divergence",
        ]

        # Extract and format metrics
        metrics_data = {}
        for key in key_metrics:
            if key in data:
                value = data[key]
                # Handle numeric types
                if isinstance(value, (int, float)):
                    metrics_data[key] = float(value)

        if not metrics_data:
            # No key metrics to report
            return

        # Build markdown report
        markdown = self._build_markdown_report(step, metrics_data)

        # Send notification
        try:
            self.apprise.notify(body=markdown)
            logger.debug(f"Sent metrics notification for step {step} (log_count={self.log_count})")
        except Exception as e:
            logger.error(f"Failed to send metrics notification: {e}")

        # Update previous metrics
        self.prev_metrics = metrics_data.copy()

    def _update_metrics(self, data: Dict[str, Any]):
        """Update metrics without sending notification (for frequency control).

        Args:
            data: Dictionary of metrics to update
        """
        key_metrics = [
            "rollout/raw_reward",
            "rollout/reward",
            "train/grad_norm",
            "train/entropy_loss",
            "train/policy_loss",
            "train/value_loss",
            "train/learning_rate",
            "train/kl_divergence",
        ]

        metrics_data = {}
        for key in key_metrics:
            if key in data:
                value = data[key]
                if isinstance(value, (int, float)):
                    metrics_data[key] = float(value)

        if metrics_data:
            self.prev_metrics = metrics_data.copy()

    def _build_markdown_report(self, step: int, metrics: Dict[str, float]) -> str:
        """Build a markdown report with metrics and deltas."""
        lines = [f"# 📊 训练报告 - Step {step}", ""]

        for key, value in metrics.items():
            # Format metric name
            metric_name = key.split("/")[-1].replace("_", " ").title()

            # Format value
            if abs(value) < 0.01:
                value_str = f"{value:.6f}"
            elif abs(value) < 1:
                value_str = f"{value:.4f}"
            else:
                value_str = f"{value:.2f}"

            # Calculate delta
            if key in self.prev_metrics:
                delta = value - self.prev_metrics[key]

                # Format delta
                if abs(delta) < 1e-8:
                    delta_str = ""
                else:
                    # Format delta value
                    if abs(delta) < 0.01:
                        delta_formatted = f"{delta:+.6f}"
                    elif abs(delta) < 1:
                        delta_formatted = f"{delta:+.4f}"
                    else:
                        delta_formatted = f"{delta:+.2f}"

                    delta_str = f" ({delta_formatted})"
            else:
                delta_str = ""

            lines.append(f"- **{metric_name}**: {value_str}{delta_str}")

        return "\n".join(lines)

    def _on_exit(self):
        """Exit handler to send failure alert if training didn't finish
        normally."""
        if self._normal_exit or not self.apprise:
            return

        # Training ended abnormally
        markdown = """# ⚠️ 训练异常终止

**状态**: 训练进程异常退出

请检查日志以了解详细信息。
"""

        try:
            self.apprise.notify(body=markdown)
            logger.info("Sent failure alert notification")
        except Exception as e:
            logger.error(f"Failed to send failure alert: {e}")

    def finish(self):
        """Mark training as finished normally and send completion
        notification."""
        self._normal_exit = True

        if not self.apprise:
            return

        markdown = f"""# ✅ 训练完成

**状态**: 训练正常完成

{self._format_final_metrics()}
"""

        try:
            self.apprise.notify(body=markdown)
            logger.info("Sent completion notification")
        except Exception as e:
            logger.error(f"Failed to send completion notification: {e}")

    def _format_final_metrics(self) -> str:
        """Format final metrics summary."""
        if not self.prev_metrics:
            return ""

        lines = ["**最终指标**:", ""]
        for key, value in self.prev_metrics.items():
            metric_name = key.split("/")[-1].replace("_", " ").title()
            if abs(value) < 0.01:
                value_str = f"{value:.6f}"
            elif abs(value) < 1:
                value_str = f"{value:.4f}"
            else:
                value_str = f"{value:.2f}"
            lines.append(f"- {metric_name}: {value_str}")

        return "\n".join(lines)

    def send_alert(self, title: str, message: str, level: str = "warning"):
        """Send a custom alert notification.

        Args:
            title: Alert title
            message: Alert message
            level: Alert level (info, warning, error)
        """
        if not self.apprise:
            return

        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
        }

        emoji = emoji_map.get(level, "📢")

        markdown = f"""# {emoji} {title}

{message}
"""

        try:
            self.apprise.notify(body=markdown)
            logger.info(f"Sent {level} alert: {title}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def log_error(self, error_message: str, error_traceback: str = None):
        """Send error notification with traceback.

        This method is used to report runtime errors during training/inference.
        It sends a detailed error notification including the error message and
        full traceback for debugging.

        Args:
            error_message: The error message or exception string
            error_traceback: Full traceback string (optional)
        """
        if not self.apprise:
            return

        # Build markdown report
        markdown_lines = [
            "# ❌ 运行时错误",
            "",
            "**错误信息**:",
            "```",
            error_message,
            "```",
        ]

        # Add traceback if provided
        if error_traceback:
            markdown_lines.extend(
                [
                    "",
                    "**调用栈**:",
                    "```",
                    error_traceback,
                    "```",
                ]
            )

        markdown = "\n".join(markdown_lines)

        try:
            self.apprise.notify(body=markdown)
            logger.info("Sent error notification with traceback")
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")
