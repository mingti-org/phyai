#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <cstdint>
#include <cstring>
struct Plan {
  cublasLtHandle_t h;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  std::vector<cublasLtMatmulHeuristicResult_t> alg;
  int error = 0;
};
extern "C" void* lt_make(int m, int n, int k, int b_row, size_t ws, int limit) {
  Plan* p = new Plan;
  cublasLtCreate(&p->h);
  cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtMatrixLayoutCreate(&p->a, CUDA_R_16BF, m, k, k);
  cublasLtMatrixLayoutCreate(&p->b, CUDA_R_16BF, k, n, b_row ? n : k);
  cublasLtMatrixLayoutCreate(&p->c, CUDA_R_16BF, m, n, n);
  cublasLtOrder_t row = CUBLASLT_ORDER_ROW, col = CUBLASLT_ORDER_COL;
  cublasLtMatrixLayoutSetAttribute(p->a, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
  cublasLtMatrixLayoutSetAttribute(p->b, CUBLASLT_MATRIX_LAYOUT_ORDER, b_row ? &row : &col, sizeof(row));
  cublasLtMatrixLayoutSetAttribute(p->c, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  p->alg.resize(limit);
  int got = 0;
  p->error = cublasLtMatmulAlgoGetHeuristic(p->h, p->op, p->a, p->b, p->c, p->c, pref, limit, p->alg.data(), &got);
  p->alg.resize(got);
  cublasLtMatmulPreferenceDestroy(pref);
  return p;
}
extern "C" int lt_count(void* ptr) { return ((Plan*)ptr)->alg.size(); }
extern "C" int lt_error(void* ptr) { return ((Plan*)ptr)->error; }
extern "C" int lt_attr(void* ptr, int i, int attr) {
  auto* p = (Plan*)ptr;
  int v = -999;
  size_t written = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&p->alg.at(i).algo, (cublasLtMatmulAlgoConfigAttributes_t)attr, &v, sizeof(v), &written);
  return v;
}
extern "C" size_t lt_ws(void* ptr, int i) { return ((Plan*)ptr)->alg.at(i).workspaceSize; }
extern "C" int lt_run(void* ptr, int i, void* a, void* b, void* out, void* workspace, size_t bytes, void* stream) {
  Plan* p = (Plan*)ptr;
  float alpha = 1, beta = 0;
  return cublasLtMatmul(p->h, p->op, &alpha, a, p->a, b, p->b, &beta, out, p->c, out, p->c, &p->alg.at(i).algo, workspace,
                        bytes, (cudaStream_t)stream);
}
extern "C" void lt_free(void* ptr) {
  Plan* p = (Plan*)ptr;
  cublasLtMatrixLayoutDestroy(p->a);
  cublasLtMatrixLayoutDestroy(p->b);
  cublasLtMatrixLayoutDestroy(p->c);
  cublasLtMatmulDescDestroy(p->op);
  cublasLtDestroy(p->h);
  delete p;
}

extern "C" int lt_configure(void* ptr, const int* attrs, size_t ws) {
  auto* p = (Plan*)ptr;
  cublasLtMatmulAlgo_t a;
  auto status = cublasLtMatmulAlgoInit(p->h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                                       attrs[0], &a);
  if (status != CUBLAS_STATUS_SUCCESS) return -(int)status;
  for (int j = 1; j < 9; ++j) {
    if (attrs[j] < 0) continue;
    if (j == 7 || j == 8) {
      uint16_t v = attrs[j];
      status = cublasLtMatmulAlgoConfigSetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)j, &v, sizeof(v));
    } else {
      int v = attrs[j];
      status = cublasLtMatmulAlgoConfigSetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)j, &v, sizeof(v));
    }
    if (status != CUBLAS_STATUS_SUCCESS) return -(int)status;
  }
  cublasLtMatmulHeuristicResult_t r{};
  status = cublasLtMatmulAlgoCheck(p->h, p->op, p->a, p->b, p->c, p->c, &a, &r);
  if (status != CUBLAS_STATUS_SUCCESS || r.workspaceSize > ws) return -100 - (int)status;
  r.algo = a;
  p->alg.push_back(r);
  return p->alg.size() - 1;
}
