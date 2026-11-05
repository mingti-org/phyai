// Fixed-workspace adapter for vLLM's Apache-2.0 collective kernels.
// No graph pointer registry: the dispatcher refreshes a fixed input buffer
// with a stream-ordered device copy, including on every graph replay.
#include "third_party/vllm/custom_all_reduce.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;

void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct Workspace {
  vllm::RankSignals signals{};
  vllm::Signal* self;
  vllm::RankData* inputs;
  void* staging;
  size_t max_bytes;
  int rank, size;
};

template <typename T>
void sum(Workspace* w, const void* input, void* output, int count,
         cudaStream_t stream, int block_limit) {
  constexpr int pack = vllm::packed_t<T>::P::size;
  if (count <= 0 || count % pack || size_t(count) * sizeof(T) > w->max_bytes ||
      block_limit <= 0 || block_limit > vllm::kMaxBlocks)
    throw std::runtime_error("Unsupported fixed SUM launch shape");
  if (input != w->staging)
    check(cudaMemcpyAsync(w->staging, input, size_t(count) * sizeof(T),
                          cudaMemcpyDeviceToDevice, stream));
  int n = count / pack;
  int blocks = std::min(block_limit, (n + 511) / 512);
  // Same stage selection as vLLM for a fully connected 2/4-GPU group.
#define CASE(RANKS)                                                            \
  case RANKS:                                                                 \
    if (RANKS == 2 || size_t(count) * sizeof(T) < 512 * 1024)                    \
      vllm::cross_device_reduce_1stage<T, RANKS><<<blocks, 512, 0, stream>>>(     \
          w->inputs, w->signals, w->self, static_cast<T*>(output), w->rank, n); \
    else                                                                      \
      vllm::cross_device_reduce_2stage<T, RANKS><<<blocks, 512, 0, stream>>>(     \
          w->inputs, w->signals, w->self, static_cast<T*>(output), w->rank, n); \
    break;
  switch (w->size) {
    CASE(2)
    CASE(4)
    default: throw std::runtime_error("Fixed SUM requires 2/4 ranks");
  }
#undef CASE
  check(cudaGetLastError());
}
}  // namespace

#define BEGIN try {
#define END                                   \
  return 0;                                   \
  } catch (const std::exception& error) {      \
    last_error = error.what();                \
    return -1;                                \
  }

extern "C" const char* error_message() { return last_error.c_str(); }
extern "C" size_t signal_size() { return sizeof(vllm::Signal); }
extern "C" size_t table_size() { return sizeof(vllm::RankData); }

extern "C" int alloc_shared(size_t bytes, void** pointer, char* handle) {
  BEGIN
  *pointer = nullptr;
  check(cudaMalloc(pointer, bytes));
  try {
    check(cudaMemset(*pointer, 0, bytes));
    check(cudaIpcGetMemHandle(reinterpret_cast<cudaIpcMemHandle_t*>(handle), *pointer));
  } catch (...) {
    cudaFree(*pointer);
    *pointer = nullptr;
    throw;
  }
  END
}

extern "C" int open_shared(const char* handle, void** pointer) {
  BEGIN
  check(cudaIpcOpenMemHandle(pointer, *reinterpret_cast<const cudaIpcMemHandle_t*>(handle),
                             cudaIpcMemLazyEnablePeerAccess));
  END
}
extern "C" int export_shared(void* pointer, char* handle, size_t* offset) {
  BEGIN
  CUdeviceptr base;
  auto result = cuPointerGetAttribute(&base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                      reinterpret_cast<CUdeviceptr>(pointer));
  if (result != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(message ? message : "Could not find CUDA allocation base");
  }
  check(cudaIpcGetMemHandle(reinterpret_cast<cudaIpcMemHandle_t*>(handle),
                            reinterpret_cast<void*>(base)));
  *offset = reinterpret_cast<CUdeviceptr>(pointer) - base;
  END
}
extern "C" int close_shared(void* pointer) {
  BEGIN
  check(cudaIpcCloseMemHandle(pointer));
  END
}
extern "C" int free_shared(void* pointer) {
  BEGIN
  check(cudaFree(pointer));
  END
}

extern "C" int create_sum(void** signals, void** inputs, size_t max_bytes,
                           int rank, int size, void** result) {
  BEGIN
  if ((size != 2 && size != 4) || rank < 0 || rank >= size)
    throw std::runtime_error("Invalid fixed SUM group");
  auto* w = new Workspace{};
  w->self = static_cast<vllm::Signal*>(signals[rank]);
  w->staging = inputs[rank];
  w->max_bytes = max_bytes;
  w->rank = rank;
  w->size = size;
  vllm::RankData table{};
  for (int i = 0; i < size; ++i) {
    w->signals.signals[i] = static_cast<vllm::Signal*>(signals[i]);
    table.ptrs[i] = inputs[i];
  }
  try {
    check(cudaMalloc(&w->inputs, sizeof(table)));
    check(cudaMemcpy(w->inputs, &table, sizeof(table), cudaMemcpyHostToDevice));
  } catch (...) {
    if (w->inputs) cudaFree(w->inputs);
    delete w;
    throw;
  }
  *result = w;
  END
}

extern "C" int reduce_sum(void* workspace, const void* input, void* output,
                           int count, void* stream, int blocks, int dtype) {
  BEGIN
  auto* w = static_cast<Workspace*>(workspace);
  auto s = reinterpret_cast<cudaStream_t>(stream);
  if (dtype == 0) sum<float>(w, input, output, count, s, blocks);
  else if (dtype == 1) sum<nv_bfloat16>(w, input, output, count, s, blocks);
  else throw std::runtime_error("Fixed SUM supports FP32/BF16 only");
  END
}
extern "C" int destroy_sum(void* workspace) {
  BEGIN
  auto* w = static_cast<Workspace*>(workspace);
  check(cudaFree(w->inputs));
  delete w;
  END
}
