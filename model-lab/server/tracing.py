"""
OpenTelemetry tracing setup for Model Lab.

This module provides distributed tracing capabilities for the production API.

Usage:
    1. Set environment variables:
       OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
       OTEL_SERVICE_NAME=model-lab-api

    2. Import and initialize in deploy_api.py:
       from server.tracing import init_tracing
       init_tracing(app)

    3. Use tracer in code:
       from server.tracing import get_tracer
       tracer = get_tracer()
       with tracer.start_as_current_span("operation") as span:
           span.set_attribute("key", "value")
           # ... do work
"""

import os

# Lazy imports to avoid dependency issues if opentelemetry not installed
_tracer = None
_initialized = False


def init_tracing(app=None) -> bool:
    """
    Initialize OpenTelemetry tracing.

    Args:
        app: Optional FastAPI app to instrument

    Returns:
        True if tracing was initialized, False otherwise
    """
    global _tracer, _initialized

    if _initialized:
        return True

    # Check if tracing is enabled
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Create resource with service name
        service_name = os.environ.get("OTEL_SERVICE_NAME", "model-lab-api")
        resource = Resource.create({"service.name": service_name})

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Add OTLP exporter
        exporter = OTLPSpanExporter(endpoint=endpoint)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        # Get tracer
        _tracer = trace.get_tracer(__name__)

        # Instrument FastAPI if provided
        if app is not None:
            FastAPIInstrumentor.instrument_app(app)

        _initialized = True
        return True

    except ImportError as e:
        print(f"OpenTelemetry not available: {e}")
        print(
            "Install with: uv pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-instrumentation-fastapi"
        )
        return False


def get_tracer():
    """
    Get the OpenTelemetry tracer.

    Returns a no-op tracer if tracing is not initialized.
    """
    global _tracer

    if _tracer is not None:
        return _tracer

    # Return no-op tracer if not initialized
    try:
        from opentelemetry import trace

        return trace.get_tracer(__name__)
    except ImportError:
        # Create a minimal no-op tracer
        class NoOpSpan:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def set_attribute(self, key, value):
                pass

            def add_event(self, name, attributes=None):
                pass

        class NoOpTracer:
            def start_as_current_span(self, name, **kwargs):
                return NoOpSpan()

        return NoOpTracer()


def trace_function(name: str | None = None):
    """
    Decorator to trace a function.

    Usage:
        @trace_function("my_operation")
        def my_function():
            pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
