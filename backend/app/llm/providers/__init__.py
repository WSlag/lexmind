"""Concrete LLM providers used by the pipeline.

Each provider is a thin adapter over the base :class:`LLMProvider` interface.
The mock provider is deterministic and offline so tests and demos run without
network access or API keys.
"""
