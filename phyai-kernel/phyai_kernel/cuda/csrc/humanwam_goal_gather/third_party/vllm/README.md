# vLLM collective synchronization

Source: <https://github.com/vllm-project/vllm>, commit
`8977b13a1c38ece61034ccb8916e76064f348919`, file
`csrc/custom_collective_common.cuh`. The Apache-2.0 license is retained here.
The header increases `kMaxBlocks` from 36 to 128; synchronization is unchanged.

`../../tp_gather.cu` uses the peer signal barriers and pointer table to copy BF16
values across GPUs. It performs no arithmetic reduction. The runner owns the
CUDA IPC buffers until all CUDA Graph launches finish. Each workspace serves
serialized requests on one CUDA stream.
