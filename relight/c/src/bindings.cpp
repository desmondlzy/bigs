#include "tensors.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("eval_sh_forward", &eval_sh_forward_tensor, "eval sh forward (CUDA)");
	m.def("eval_sh_backward", &eval_sh_backward_tensor, "eval sh backward (CUDA)");
}
