// Bit-preserving CUDA IPC gathers using vLLM's Apache-2.0 synchronization.
// Peer input buffers remain live across CUDA Graph capture and replay.
#include "third_party/vllm/custom_collective_common.cuh"

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

}  // namespace

#define BEGIN try {
#define END                             \
  return 0;                             \
  }                                     \
  catch (const std::exception& error) { \
    last_error = error.what();          \
    return -1;                          \
  }

extern "C" const char* error_message() { return last_error.c_str(); }
extern "C" size_t signal_size() { return sizeof(vllm::Signal); }

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
  check(cudaIpcOpenMemHandle(pointer, *reinterpret_cast<const cudaIpcMemHandle_t*>(handle), cudaIpcMemLazyEnablePeerAccess));
  END
}
extern "C" int export_shared(void* pointer, char* handle, size_t* offset) {
  BEGIN
  CUdeviceptr base;
  auto result = cuPointerGetAttribute(&base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, reinterpret_cast<CUdeviceptr>(pointer));
  if (result != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(message ? message : "Could not find CUDA allocation base");
  }
  check(cudaIpcGetMemHandle(reinterpret_cast<cudaIpcMemHandle_t*>(handle), reinterpret_cast<void*>(base)));
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

extern "C" int create_gather(void** signals, void** inputs, size_t max_bytes, int rank, int size, void** result) {
  BEGIN
  if ((size != 2 && size != 4) || rank < 0 || rank >= size) throw std::runtime_error("Invalid fixed gather group");
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

extern "C" int destroy_gather(void* workspace) {
  BEGIN
  auto* w = static_cast<Workspace*>(workspace);
  check(cudaFree(w->inputs));
  delete w;
  END
}

namespace {
template<int Ranks, int Columns = 0>
__global__ void __launch_bounds__(512, 1)
    gather_columns(vllm::RankData* pointers, vllm::RankSignals signals, vllm::Signal* self, uint4* output, int rank,
                   int vectors, int width_vectors) {
  auto table = *pointers;
  vllm::barrier_at_start<Ranks>(signals, self, rank);
  const int columns = Columns ? Columns / 8 : width_vectors;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < vectors; index += gridDim.x * blockDim.x) {
    const int row = index / (columns * Ranks);
    const int source_rank = (index / columns) % Ranks;
    const int column = index % columns;
    output[index] = static_cast<const uint4*>(table.ptrs[source_rank])[row * columns + column];
  }
  vllm::barrier_at_end<Ranks, true>(signals, self, rank);
}

template<int Ranks>
void launch_gather(Workspace* w, void* output, int rows, int columns, cudaStream_t stream, int block_limit) {
  int vectors = size_t(rows) * columns * Ranks / 8;
  int blocks = std::min(block_limit, (vectors + 511) / 512);
#define GATHER_WIDTH(C)                                                                                                        \
  case C:                                                                                                                      \
    gather_columns<Ranks, C><<<blocks, 512, 0, stream>>>(w->inputs, w->signals, w->self, static_cast<uint4*>(output), w->rank, \
                                                         vectors, columns / 8);                                                \
    break
  switch (columns) {
    GATHER_WIDTH(1024);
    GATHER_WIDTH(2048);
    GATHER_WIDTH(3072);
    GATHER_WIDTH(4096);
    GATHER_WIDTH(6144);
    GATHER_WIDTH(12288);
    default:
      gather_columns<Ranks><<<blocks, 512, 0, stream>>>(w->inputs, w->signals, w->self, static_cast<uint4*>(output), w->rank,
                                                        vectors, columns / 8);
  }
#undef GATHER_WIDTH
}
}  // namespace

extern "C" int gather_last_dimension(void* workspace, const void* input, void* output, int rows, int columns, void* stream,
                                     int blocks) {
  BEGIN
  auto* w = static_cast<Workspace*>(workspace);
  if (rows <= 0 || columns <= 0 || columns % 8 || size_t(rows) * columns * 2 > w->max_bytes || blocks <= 0
      || blocks > vllm::kMaxBlocks)
    throw std::runtime_error("Unsupported fixed gather shape");
  auto s = reinterpret_cast<cudaStream_t>(stream);
  if (input != w->staging) check(cudaMemcpyAsync(w->staging, input, size_t(rows) * columns * 2, cudaMemcpyDeviceToDevice, s));
  if (w->size == 2)
    launch_gather<2>(w, output, rows, columns, s, blocks);
  else if (w->size == 4)
    launch_gather<4>(w, output, rows, columns, s, blocks);
  else
    throw std::runtime_error("Fixed gather supports 2/4 ranks");
  check(cudaGetLastError());
  END
}

namespace {
template<int Ranks>
__global__ void __launch_bounds__(512, 1)
    gather_kv_columns(vllm::RankData* pointers, vllm::RankSignals signals, vllm::Signal* self, const uint4* text_k,
                      const uint4* text_v, uint4* out_k, uint4* out_v, int rank, int part, int tokens, int text) {
  auto table = *pointers;
  vllm::barrier_at_start<Ranks>(signals, self, rank);
  constexpr int width = 1024 / 8;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (text + tokens) * width; i += gridDim.x * blockDim.x) {
    int token = i / width, col = i % width;
    if (token < text) {
      out_k[i] = text_k[i];
      out_v[i] = text_v[i];
    } else {
      int local = token - text, source_rank = local / part, row = local % part;
      auto source = static_cast<const uint4*>(table.ptrs[source_rank]);
      out_k[i] = source[row * 2 * width + col];
      out_v[i] = source[row * 2 * width + width + col];
    }
  }
  vllm::barrier_at_end<Ranks, true>(signals, self, rank);
}
}  // namespace
extern "C" int gather_sp_kv(void* workspace, const void* text_k, const void* text_v, void* out_k, void* out_v, int part,
                            int tokens, int text, void* stream, int blocks) {
  BEGIN
  auto* w = static_cast<Workspace*>(workspace);
  if ((w->size != 2 && w->size != 4) || part <= 0 || tokens <= 0 || tokens > w->size * part || text < 0
      || size_t(part) * 2048 * 2 > w->max_bytes || blocks <= 0 || blocks > 128)
    throw std::runtime_error("Unsupported sequence KV gather shape");
  int grid = std::min(blocks, ((text + tokens) * 128 + 511) / 512);
#define KV_GATHER(RANKS)                                                                                    \
  gather_kv_columns<RANKS><<<grid, 512, 0, static_cast<cudaStream_t>(stream)>>>(                            \
      w->inputs, w->signals, w->self, static_cast<const uint4*>(text_k), static_cast<const uint4*>(text_v), \
      static_cast<uint4*>(out_k), static_cast<uint4*>(out_v), w->rank, part, tokens, text)
  if (w->size == 2) {
    KV_GATHER(2);
  } else {
    KV_GATHER(4);
  }
#undef KV_GATHER
  check(cudaGetLastError());
  END
}
