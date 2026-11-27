"""Prediction and robot/model conversion interfaces without model dependencies.

This module defines the abstract boundary between robot observation snapshots and
machine learning model inference:
1. `Policy[RequestT, ResultT]`: Generic synchronous predictor interface. Decoupled
   from any specific inference engine, GPU framework, or network transport.
2. `PolicyAdapter[RequestT, ResultT]`: Adapter converting physical robot Observation
   snapshots into model-native request types, and model-native results back into
   physical ActionChunks.
3. `EnginePolicy`: Lightweight wrapper around an engine exposing a `.step()` method
   (such as PhyAI's `Engine`), allowing direct embedding without static import dependencies.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar, Protocol

from .types import ActionChunk, Observation
from .scheduling import InferenceContext

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class Policy(Protocol[RequestT, ResultT]):
    """Generic synchronous model policy predictor interface.

    Binds model input (RequestT) and output (ResultT) types without prescribing
    their internal tensor structures or layout conventions.

    Implementations can be in-process local engines (like EnginePolicy), CPU test doubles
    (like StepPolicy in tests), or remote network clients (like RemotePolicy over ZMQ).
    """

    def predict(self, request: RequestT) -> ResultT:
        """Execute one synchronous inference forward pass on the prepared request.

        Args:
            request: Model-native input prepared by the paired PolicyAdapter.to_request().

        Returns:
            Model-native prediction output to be consumed by PolicyAdapter.to_actions().
        """
        ...


class PolicyAdapter(Protocol[RequestT, ResultT]):
    """Bridges physical robot observations and model-specific prediction types.

    Responsibilities:
    - Feature extraction & formatting: Selects required cameras/sensors and constructs
      the model input structure RequestT.
    - Scheduling context integration: Receives InferenceContext, allowing Real-Time
      Chunking (RTC) adapters to inform the model of pending actions still executing.
    - Coordinate & unit restoration: Converts model predictions back into physical
      engineering units (radians, meters) and maps relative deltas to absolute targets
      anchored strictly to the observation snapshot paired with the request.
    """

    def to_request(
        self, observation: Observation, context: InferenceContext
    ) -> RequestT:
        """Convert a physical robot observation and scheduling context into a model request.

        Args:
            observation: Complete, fresh sensor snapshot from the robot.
            context: InferenceContext describing actions that will execute during this
                inference step (empty for StopAndRefill; populated for RTC).

        Returns:
            Model-native request object (RequestT).
        """
        ...

    def to_actions(self, result: ResultT, observation: Observation) -> ActionChunk:
        """Translate model output into an ActionChunk of physical actuator targets.

        Invariance:
            Relative delta decoding must add predictions strictly to the `observation`
            snapshot that generated the request. Never reread hardware state here, as
            that would alter the policy's intended trajectory anchor.

        Args:
            result: Raw model inference output (ResultT).
            observation: The identical Observation snapshot previously passed to to_request().

        Returns:
            ActionChunk containing ordered single-step physical command mappings.
        """
        ...


class EnginePolicy(Generic[RequestT, ResultT]):
    """Wraps an engine exposing a step(request) method as a Policy.

    Designed for seamless integration with the PhyAI Engine while keeping phyai-robot
    importable in environments without PyTorch or CUDA runtime dependencies.

    Lifecycle:
        The application owns the underlying engine instance and remains responsible
        for its lifecycle and GPU resource release upon shutdown.
    """

    def __init__(self, engine: Any) -> None:
        """Initialize wrapper verifying that the engine provides a callable step method.

        Args:
            engine: Object providing a callable step(request) -> result method.

        Raises:
            TypeError: If engine does not provide a callable step attribute.
        """
        if not callable(getattr(engine, "step", None)):
            raise TypeError("engine must provide step(request)")
        self._engine: Any = engine  # Underlying inference engine instance.

    def predict(self, request: RequestT) -> ResultT:
        """Delegate prediction directly to engine.step(request)."""
        return self._engine.step(request)
