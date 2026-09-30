"""Lifecycle coverage for the CUDA graph wrapper."""

from phyai.runtime.cuda_graph_manager import CudaGraph


class _FakeTorchGraph:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


def test_cuda_graph_reset_releases_captured_state():
    graph = CudaGraph()
    fake = _FakeTorchGraph()
    graph._captured = True
    graph._graph = fake
    graph._input_buffers = {"input": object()}
    graph._output = object()

    graph.reset()

    assert fake.reset_calls == 1
    assert not graph.is_captured
    assert graph._graph is None
    assert graph._input_buffers == {}
    assert graph._output is None
