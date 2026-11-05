# vLLM collective kernels

Source: <https://github.com/vllm-project/vllm>, commit
`8977b13a1c38ece61034ccb8916e76064f348919`, files
`csrc/custom_all_reduce.cuh` and `csrc/custom_collective_common.cuh`.
The original Apache-2.0 license is retained here.

The only header change increases `kMaxBlocks` from 36 to 128 so the SM90
deployment can launch 96 blocks. Arithmetic and synchronization are unchanged.
`../../tp_sum.cu` calls the one/two-stage kernels directly with a single fixed
peer-input table. It does not instantiate vLLM's `CustomAllreduce` class or use
its graph pointer registry. The dispatcher copies each
input into a lifetime-owned Torch tensor on the current stream. This refreshes
the shared input on every invocation, including CUDA Graph replay. Weights, residuals,
norms and the solver do not pass through this adapter.

The Python owner allocates two bounded CUDA IPC regions per rank: signal/scratch
and a Torch-owned input tensor. It owns the peer mappings and the single 64-byte pointer
table until all graph launches finish. Serving is serialized; concurrent
requests or independent CUDA streams cannot share this workspace.
