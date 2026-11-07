#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Script for deploying and using the Metrics Service.

This script demonstrates how to:
1. Deploy the Metrics Service using Ray Serve
2. Use the Metrics Client to log metrics
3. Report metrics at the end of steps
"""

import argparse

import ray
from ray import serve

from relax.utils.metrics.client import MetricsClient
from relax.utils.metrics.service import MetricsService
from relax.utils.misc import create_namespace


def create_demo_args() -> argparse.Namespace:
    """Create demo arguments for metrics service configuration."""
    args_dict = {
        "use_wandb": False,  # Set to True if you have W&B configured
        "use_tensorboard": True,
        "use_clearml": False,  # Set to True if you have ClearML configured
        "tb_project_name": "demo-project",
        "tb_experiment_name": "demo-experiment",
        "wandb_project": "demo-project",
        "wandb_team": "demo-team",
        "wandb_group": "demo-group",
        "use_metrics_service": True,  # Enable metrics service
        # TensorBoardAdapter reads args.save directly after Task 13 cleanup.
        "save": None,
    }
    return create_namespace(args_dict)


def deploy_metrics_service(args: argparse.Namespace):
    """Deploy the metrics service using Ray Serve."""
    print("Deploying Metrics Service...")

    # Start Ray if not already started
    if not ray.is_initialized():
        ray.init()

    # Deploy the metrics service
    deployment = MetricsService.bind(
        healthy=None,  # Health manager (optional)
        pg=None,  # Placement group (optional)
        config=args,
        role="metrics",
    )

    serve.run(deployment, name="metrics", route_prefix="/metrics")
    print("Metrics Service deployed successfully at http://localhost:8000/metrics")

    return deployment


def demo_metrics_collection(args: argparse.Namespace):
    """Demonstrate metrics collection and reporting."""
    print("\n=== Demo: Metrics Collection ===")

    from relax.utils.utils import get_serve_url

    # Create metrics client using get_serve_url
    service_url = get_serve_url(route_prefix="/metrics")
    client = MetricsClient(service_url)

    # Check service health
    health = client.health_check()
    print(f"Service health: {health}")

    # Demo 1: Log individual metrics
    print("\n1. Logging individual metrics...")
    for step in range(3):
        # Log some metrics for this step
        client.log_metric(step, "train/loss", 0.5 + step * 0.1)
        client.log_metric(step, "train/accuracy", 0.8 - step * 0.05)
        client.log_metric(step, "train/lr", 0.001)

        # Add some performance metrics
        client.log_metric(step, "perf/throughput", 100 + step * 10)
        client.log_metric(step, "perf/latency", 50 - step * 5)

        print(f"  Step {step}: Logged 5 metrics")

    # Demo 2: Log batch metrics
    print("\n2. Logging batch metrics...")
    step = 3
    batch_metrics = {
        "train/loss": 0.42,
        "train/accuracy": 0.92,
        "train/lr": 0.0005,
        "perf/throughput": 150,
        "perf/latency": 35,
        "custom/metric1": 123.45,
        "custom/metric2": "some_value",
    }

    client.log_metrics_batch(step, batch_metrics)
    print(f"  Step {step}: Logged {len(batch_metrics)} metrics in batch")

    # Demo 3: Report metrics for each step
    print("\n3. Reporting metrics at step end...")
    for step in range(4):
        result = client.report_step(step)
        status = result.get("status", "unknown")
        message = result.get("message", "No message")
        print(f"  Step {step}: {status} - {message}")

        if "results" in result:
            for backend, backend_result in result["results"].items():
                print(f"    {backend}: {backend_result}")

    # Demo 4: Check buffer status
    print(f"\n4. Buffered metrics count: {client.get_buffered_metrics_count()}")

    # Demo 5: Clear buffer
    client.clear_buffer()
    print("5. Cleared metrics buffer")
    print(f"   Buffered metrics count after clear: {client.get_buffered_metrics_count()}")


def demo_direct_api_calls():
    """Demonstrate direct API calls to the metrics service."""
    print("\n=== Demo: Direct API Calls ===")

    import json

    import requests

    base_url = "http://localhost:8000/metrics"

    # 1. Health check
    print("1. Health check:")
    response = requests.get(f"{base_url}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")

    # 2. Log a metric via API
    print("\n2. Logging a metric via API:")
    payload = {
        "step": 100,
        "metric_name": "api/demo_metric",
        "metric_value": 42.0,
        "tags": {"source": "api_demo", "type": "test"},
    }
    response = requests.post(f"{base_url}/log_metric", json=payload)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")

    # 3. Report step via API
    print("\n3. Reporting step via API:")
    payload = {"step": 100}
    response = requests.post(f"{base_url}/report_step", json=payload)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")


def main():
    parser = argparse.ArgumentParser(description="Deploy and demo Metrics Service")
    parser.add_argument("--deploy-only", action="store_true", help="Only deploy the service")
    parser.add_argument("--demo-only", action="store_true", help="Only run demos (assumes service is deployed)")
    parser.add_argument("--api-demo", action="store_true", help="Run API demo")
    args = parser.parse_args()

    # Create demo configuration
    demo_args = create_demo_args()

    if not args.demo_only and not args.api_demo:
        # Deploy the service
        deploy_metrics_service(demo_args)

    if not args.deploy_only:
        if args.api_demo:
            demo_direct_api_calls()
        else:
            # Run metrics collection demo
            demo_metrics_collection(demo_args)

    if not args.deploy_only and not args.demo_only and not args.api_demo:
        print("\n=== Summary ===")
        print("Metrics Service deployment and demo completed successfully!")
        print("\nTo use the metrics service in your code:")
        print("1. Set `use_metrics_service=True` in your config")
        print("2. The service URL is automatically determined using get_serve_url()")
        print("3. Call `tracking_utils.log(args, metrics, 'step')` as before")
        print("4. Call `tracking_utils.flush_metrics(args)` at step end")
        print("\nOr use the MetricsClient directly:")
        print("  from relax.utils.metrics.client import MetricsClient")
        print("  from relax.utils.utils import get_serve_url")
        print("  service_url = get_serve_url(route_prefix='/metrics')")
        print("  client = MetricsClient(service_url)")
        print("  client.log_metric(step, 'metric/name', value)")
        print("  client.report_step(step)")


if __name__ == "__main__":
    main()
