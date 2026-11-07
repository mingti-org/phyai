# Eager Cosmos3 policy options

These options apply to the `cosmos3_policy` engine entry. They default to
`False`. Configure the engine runtime with `use_cuda_graph=False` and use one
rank for both tensor parallelism and classifier-free guidance to enable them.
Graph and parallel execution keep the configured reference paths.

Pass the options through `Cosmos3PolicyArgs`, as shown below. The existing
[policy example](run_cosmos3_policy.py) shows engine construction and input
preprocessing. These options do not change the sampler, checkpoint or request
format. Language token lengths and observation shapes can vary between requests;
each implementation checks its supported contract before dispatch.

## Conditioning prefixes

Set `conditioning_prefix=True` to encode only the causal VAE prefix needed by
the requested conditioning latent indices. Public VAE encoding still returns its
full result. Unsupported VAE types, selections, modes and batches retain the
original path.

## Attention preparation

Set `fuse_eager_attention=True` to use the runner's compiled RoPE and KV
preparation during eager inference. Compilation uses
`torch.compile(dynamic=True)` and preserves intermediate precision casts.
Supported Dynamo compilation errors fall back to eager preparation. Warm up the
shapes used by the workload before measuring steady inference latency.

## Configuration example

```python
from phyai.models.cosmos3.main_cosmos3_policy import Cosmos3PolicyArgs

args = Cosmos3PolicyArgs(
    checkpoint_dir=checkpoint_dir,
    conditioning_prefix=True,
    fuse_eager_attention=True,
)
```

Here `checkpoint_dir` is the checkpoint supplied by the application. Set
`EngineConfig.runtime.use_cuda_graph=False` when constructing the engine.
