#include "rapidllm/runtime/cuda_engine.h"

#if !defined(RAPIDLLM_WITH_CUDA)

namespace rapidllm::cuda_gen {

bool available() { return false; }

std::unique_ptr<Engine> Engine::create(WeightStore&, int) { return nullptr; }

} // namespace rapidllm::cuda_gen

#endif
