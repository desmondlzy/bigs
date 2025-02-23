#include "tensors.cuh"
#include "spherical_harmonics.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)
 
torch::Tensor eval_sh_forward_tensor(
    const unsigned degree,
    const unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &coeffs
) {
    CHECK_INPUT(viewdirs);
    CHECK_INPUT(coeffs);

    unsigned num_bases = num_sh_bases(degree);
    if (coeffs.ndimension() != 3) { 
        AT_ERROR("coeffs must have dimensions (N, D, C)"); }
    if (coeffs.size(1) != num_bases) { AT_ERROR("coeffs size not consistent with degree"); }

    unsigned num_points = coeffs.size(0);
	unsigned num_channels = coeffs.size(2);

    torch::Tensor colors = torch::empty({num_points, num_channels}, coeffs.options());
    eval_sh_forward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
		num_channels,
        degree,
        degrees_to_use,
        (float3 *)viewdirs.contiguous().data_ptr<float>(),
        coeffs.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>()
    );
    return colors;
}

torch::Tensor eval_sh_backward_tensor(
    const unsigned degree,
    const unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &v_colors
) {
    CHECK_INPUT(viewdirs);
    CHECK_INPUT(v_colors);

    unsigned num_points = v_colors.size(0);
    unsigned num_channels = v_colors.size(1);

    if (viewdirs.ndimension() != 2 || viewdirs.size(0) != num_points) {
        AT_ERROR("viewdirs must have dimensions (N, C)");
    }
    if (v_colors.ndimension() != 2 || v_colors.size(0) != num_points) {
        AT_ERROR("v_colors must have dimensions (N, C)");
    }
    unsigned num_bases = num_sh_bases(degree);

    torch::Tensor v_coeffs =
        torch::zeros({num_points, num_bases, num_channels}, v_colors.options());
    eval_sh_backward_kernel<<<
        (num_points + N_THREADS - 1) / N_THREADS,
        N_THREADS>>>(
        num_points,
		num_channels,
        degree,
        degrees_to_use,
        (float3 *)viewdirs.contiguous().data_ptr<float>(),
        v_colors.contiguous().data_ptr<float>(),
        v_coeffs.contiguous().data_ptr<float>()
    );

    return v_coeffs;
}

