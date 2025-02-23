#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

__device__ __constant__ float SH_C0 = 0.28209479177387814f;
__device__ __constant__ float SH_C1 = 0.4886025119029199f;
__device__ __constant__ float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
__device__ __constant__ float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};
__device__ __constant__ float SH_C4[] = {
    2.5033429417967046f,
    -1.7701307697799304,
    0.9461746957575601f,
    -0.6690465435572892f,
    0.10578554691520431f,
    -0.6690465435572892f,
    0.47308734787878004f,
    -1.7701307697799304f,
    0.6258357354491761f};

// This function is used in both host and device code
__host__ __device__ unsigned num_sh_bases(const unsigned degree) {
    if (degree == 1)
        return 1;
    if (degree == 2)
        return 4;
    if (degree == 3)
        return 9;
    if (degree == 4)
        return 16;
    if (degree == 5)
        return 25;
    if (degree == 6)    
        return 36;
    if (degree == 7)
        return 49;

    return 64;
}

__device__ void eval_sh(
    const unsigned degree,
	const unsigned num_channels,
    const float3 &dir,
    const float *coeffs,
    float *colors
) {
    // Expects v_values to be len num_channels
    // and v_coeffs to be num_bases * num_channels
    for (int c = 0; c < num_channels; ++c) {
        colors[c] = SH_C0 * coeffs[c];
    }
    if (degree <= 1) {
        return;
    }

    float norm = sqrt(
        dir.x * dir.x + dir.y * dir.y + dir.z * dir.z
    );
    float x = dir.x / norm;
    float y = dir.y / norm;
    float z = dir.z / norm;

    float x2 = x * x;
    float xy = x * y;
    float xz = x * z;
    float y2 = y * y;
    float yz = y * z;
    float z2 = z * z;

	float x4 = x2 * x2;
	float y4 = y2 * y2;
	float z4 = z2 * z2;

	float x6 = x4 * x2;
	float y6 = y4 * y2;
	float z6 = z4 * z2;
    // expects num_channels * num_bases coefficients
    // supports up to num_bases = 25
    for (int c = 0; c < num_channels; ++c) {
        colors[c] += SH_C1 * (-y * coeffs[1 * num_channels + c] +
                              z * coeffs[2 * num_channels + c] -
                              x * coeffs[3 * num_channels + c]);
        if (degree <= 2) {
            continue;
        }
        colors[c] +=
            (SH_C2[0] * xy * coeffs[4 * num_channels + c] +
             SH_C2[1] * yz * coeffs[5 * num_channels + c] +
             SH_C2[2] * (2.f * z2 - x2 - y2) * coeffs[6 * num_channels + c] +
             SH_C2[3] * xz * coeffs[7 * num_channels + c] +
             SH_C2[4] * (x2 - y2) * coeffs[8 * num_channels + c]);
        if (degree <= 3) {
            continue;
        }
        colors[c] +=
            (SH_C3[0] * y * (3.f * x2 - y2) * coeffs[9 * num_channels + c] +
             SH_C3[1] * xy * z * coeffs[10 * num_channels + c] +
             SH_C3[2] * y * (4.f * z2 - x2 - y2) * coeffs[11 * num_channels + c] +
             SH_C3[3] * z * (2.f * z2 - 3.f * x2 - 3.f * y2) *
                 coeffs[12 * num_channels + c] +
             SH_C3[4] * x * (4.f * z2 - x2 - y2) * coeffs[13 * num_channels + c] +
             SH_C3[5] * z * (x2 - y2) * coeffs[14 * num_channels + c] +
             SH_C3[6] * x * (x2 - 3.f * y2) * coeffs[15 * num_channels + c]);
        if (degree <= 4) {
            continue;
        }
        colors[c] +=
            (SH_C4[0] * xy * (x2 - y2) * coeffs[16 * num_channels + c] +
             SH_C4[1] * yz * (3.f * x2 - y2) * coeffs[17 * num_channels + c] +
             SH_C4[2] * xy * (7.f * z2 - 1.f) * coeffs[18 * num_channels + c] +
             SH_C4[3] * yz * (7.f * z2 - 3.f) * coeffs[19 * num_channels + c] +
             SH_C4[4] * (z2 * (35.f * z2 - 30.f) + 3.f) *
                 coeffs[20 * num_channels + c] +
             SH_C4[5] * xz * (7.f * z2 - 3.f) * coeffs[21 * num_channels + c] +
             SH_C4[6] * (x2 - y2) * (7.f * z2 - 1.f) *
                 coeffs[22 * num_channels + c] +
             SH_C4[7] * xz * (x2 - 3.f * y2) * coeffs[23 * num_channels + c] +
             SH_C4[8] * (x2 * (x2 - 3.f * y2) - y2 * (3.f * x2 - y2)) *
                 coeffs[24 * num_channels + c]);
	
		if (degree <= 5) {
			continue;
		}
		colors[c] += (
			  coeffs[25 * num_channels + c] * (0.65638205684017015f*y*(10.0f*x2*y2 - 5.0f*x4 - y4))
			+ coeffs[26 * num_channels + c] * (8.3026492595241645f*xy*z*(x2 - y2))
	        + coeffs[27 * num_channels + c] * (-0.48923829943525038f*y*(3.0f*x2 - y2)*(9.0f*z2 - 1.0f))                         // -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
	        + coeffs[28 * num_channels + c] * (4.7935367849733241f*xy*z*(3.0f*z2 - 1.0f))                             // sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
	        + coeffs[29 * num_channels + c] * (0.45294665119569694f*y*(14.0f*z2 - 21.0f*z4 - 1.0f))                            // sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
	        + coeffs[30 * num_channels + c] * (0.1169503224534236f*z*(-70.0f*z2 + 63.0f*z4 + 15.0f))                           // sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
	        + coeffs[31 * num_channels + c] * (0.45294665119569694f*x*(14.0f*z2 - 21.0f*z4 - 1.0f))                           // sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
	        + coeffs[32 * num_channels + c] * (2.3967683924866621f*z*(x2 - y2)*(3.0f*z2 - 1.0f))                              // sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
	        + coeffs[33 * num_channels + c] * (-0.48923829943525038f*x*(x2 - 3.0f*y2)*(9.0f*z2 - 1.0f))                         // -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
	        + coeffs[34 * num_channels + c] * (2.0756623148810411f*z*(-6.0f*x2*y2 + x4 + y4))                        // 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
	        + coeffs[35 * num_channels + c] * (0.65638205684017015f*x*(10.0f*x2*y2 - x4 - 5.0f*y4))                           // 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
		);


		if (degree <= 6) { continue; }
		colors[c] += (
			  coeffs[36 * num_channels + c] * (1.3663682103838286f*xy*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4))                               // sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
			+ coeffs[37 * num_channels + c] * (2.3666191622317521f*yz*(10.0f*x2*y2 - 5.0f*x4 - y4))                            // 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
			+ coeffs[38 * num_channels + c] * (2.0182596029148963f*xy*(x2 - y2)*(11.0f*z2 - 1.0f))                             // 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
			+ coeffs[39 * num_channels + c] * (-0.92120525951492349f*yz*(3.0f*x2 - y2)*(11.0f*z2 - 3.0f))                               // -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
			+ coeffs[40 * num_channels + c] * (0.92120525951492349f*xy*(-18.0f*z2 + 33.0f*z4 + 1.0f))                           // sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
			+ coeffs[41 * num_channels + c] * (0.58262136251873131f*yz*(30.0f*z2 - 33.0f*z4 - 5.0f))                            // sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
			+ coeffs[42 * num_channels + c] * (6.6747662381009842f*z2 - 20.024298714302954f*z4 + 14.684485723822165f*z6 - 0.31784601133814211f)                         // sqrt(13)*(105*z2 - 315*z4 + 231*z6 - 5)/(32*sqrt(pi))
			+ coeffs[43 * num_channels + c] * (0.58262136251873131f*xz*(30.0f*z2 - 33.0f*z4 - 5.0f))                            // sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
			+ coeffs[44 * num_channels + c] * (0.46060262975746175f*(x2 - y2)*(11.0f*z2*(3.0f*z2 - 1.0f) - 7.0f*z2 + 1.0f))                               // sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2 + 1)/(64*sqrt(pi))
			+ coeffs[45 * num_channels + c] * (-0.92120525951492349f*xz*(x2 - 3.0f*y2)*(11.0f*z2 - 3.0f))                               // -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
			+ coeffs[46 * num_channels + c] * (0.50456490072872406f*(11.0f*z2 - 1.0f)*(-6.0f*x2*y2 + x4 + y4))                          // 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
			+ coeffs[47 * num_channels + c] * (2.3666191622317521f*xz*(10.0f*x2*y2 - x4 - 5.0f*y4))                            // 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
			+ coeffs[48 * num_channels + c] * (10.247761577878714f*x2*y4 - 10.247761577878714f*x4*y2 + 0.6831841051919143f*x6 - 0.6831841051919143f*y6)                         // sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
		);

		if (degree <= 7) { continue; }
		colors[c] += (
			  coeffs[49 * num_channels + c] * (0.70716273252459627f*y*(-21.0f*x2*y4 + 35.0f*x4*y2 - 7.0f*x6 + y6))                              // 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 - 7*x6 + y6)/(64*sqrt(pi))
			+ coeffs[50 * num_channels + c] * (5.2919213236038001f*xy*z*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4))                             // 3*sqrt(10010)*xy*z*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
			+ coeffs[51 * num_channels + c] * (-0.51891557872026028f*y*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + 5.0f*x4 + y4))                          // -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 + 5*x4 + y4)/(64*sqrt(pi))
			+ coeffs[52 * num_channels + c] * (4.1513246297620823f*xy*z*(x2 - y2)*(13.0f*z2 - 3.0f))                           // 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
			+ coeffs[53 * num_channels + c] * (-0.15645893386229404f*y*(3.0f*x2 - y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f))                              // -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
			+ coeffs[54 * num_channels + c] * (0.44253269244498261f*xy*z*(-110.0f*z2 + 143.0f*z4 + 15.0f))                              // 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
			+ coeffs[55 * num_channels + c] * (0.090331607582517306f*y*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f))                              // sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
			+ coeffs[56 * num_channels + c] * (0.068284276912004949f*z*(315.0f*z2 - 693.0f*z4 + 429.0f*z6 - 35.0f))                              // sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
			+ coeffs[57 * num_channels + c] * (0.090331607582517306f*x*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f))                              // sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
			+ coeffs[58 * num_channels + c] * (0.07375544874083044f*z*(x2 - y2)*(143.0f*z2*(3.0f*z2 - 1.0f) - 187.0f*z2 + 45.0f))                         // sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2 - 1) - 187*z2 + 45)/(64*sqrt(pi))
			+ coeffs[59 * num_channels + c] * (-0.15645893386229404f*x*(x2 - 3.0f*y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f))                              // -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
			+ coeffs[60 * num_channels + c] * (1.0378311574405206f*z*(13.0f*z2 - 3.0f)*(-6.0f*x2*y2 + x4 + y4))                         // 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
			+ coeffs[61 * num_channels + c] * (-0.51891557872026028f*x*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + x4 + 5.0f*y4))                          // -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 + x4 + 5*y4)/(64*sqrt(pi))
			+ coeffs[62 * num_channels + c] * (2.6459606618019f*z*(15.0f*x2*y4 - 15.0f*x4*y2 + x6 - y6))                               // 3*sqrt(10010)*z*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
			+ coeffs[63 * num_channels + c] * (0.70716273252459627f*x*(-35.0f*x2*y4 + 21.0f*x4*y2 - x6 + 7.0f*y6))                              // 3*sqrt(715)*x*(-35*x2*y4 + 21*x4*y2 - x6 + 7*y6)/(64*sqrt(pi))
		);
    }
}

__device__ void eval_sh_grad(
    const unsigned degree,
	const unsigned num_channels,
    const float3 &dir,
    const float *v_values,
    float *v_coeffs
) {
    // Expects v_values to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    #pragma unroll
    for (int c = 0; c < num_channels; ++c) {
        v_coeffs[c] = SH_C0 * v_values[c];
    }
    if (degree <= 1) {
        return;
    }

    float norm = sqrt(
        dir.x * dir.x + dir.y * dir.y + dir.z * dir.z
    );
    float x = dir.x / norm;
    float y = dir.y / norm;
    float z = dir.z / norm;

    float x2 = x * x;
    float xy = x * y;
    float xz = x * z;
    float y2 = y * y;
    float yz = y * z;
    float z2 = z * z;

	float x4 = x2 * x2;
	float y4 = y2 * y2;
	float z4 = z2 * z2;

	float x6 = x4 * x2;
	float y6 = y4 * y2;
	float z6 = z4 * z2;
    
    #pragma unroll
    for (int c = 0; c < num_channels; ++c) {
        float v1 = -SH_C1 * y;
        float v2 = SH_C1 * z;
        float v3 = -SH_C1 * x;
        v_coeffs[1 * num_channels + c] = v1 * v_values[c];
        v_coeffs[2 * num_channels + c] = v2 * v_values[c];
        v_coeffs[3 * num_channels + c] = v3 * v_values[c];

        if (degree <= 2) { continue; }

        float v4 = SH_C2[0] * xy;
        float v5 = SH_C2[1] * yz;
        float v6 = SH_C2[2] * (2.f * z2 - x2 - y2);
        float v7 = SH_C2[3] * xz;
        float v8 = SH_C2[4] * (x2 - y2);
        v_coeffs[4 * num_channels + c] = v4 * v_values[c];
        v_coeffs[5 * num_channels + c] = v5 * v_values[c];
        v_coeffs[6 * num_channels + c] = v6 * v_values[c];
        v_coeffs[7 * num_channels + c] = v7 * v_values[c];
        v_coeffs[8 * num_channels + c] = v8 * v_values[c];

        if (degree <= 3) { continue; }

        float v9 = SH_C3[0] * y * (3.f * x2 - y2);
        float v10 = SH_C3[1] * xy * z;
        float v11 = SH_C3[2] * y * (4.f * z2 - x2 - y2);
        float v12 = SH_C3[3] * z * (2.f * z2 - 3.f * x2 - 3.f * y2);
        float v13 = SH_C3[4] * x * (4.f * z2 - x2 - y2);
        float v14 = SH_C3[5] * z * (x2 - y2);
        float v15 = SH_C3[6] * x * (x2 - 3.f * y2);
        v_coeffs[9 * num_channels + c] = v9 * v_values[c];
        v_coeffs[10 * num_channels + c] = v10 * v_values[c];
        v_coeffs[11 * num_channels + c] = v11 * v_values[c];
        v_coeffs[12 * num_channels + c] = v12 * v_values[c];
        v_coeffs[13 * num_channels + c] = v13 * v_values[c];
        v_coeffs[14 * num_channels + c] = v14 * v_values[c];
        v_coeffs[15 * num_channels + c] = v15 * v_values[c];

        if (degree <= 4) { continue; }

        float v16 = SH_C4[0] * xy * (x2 - y2);
        float v17 = SH_C4[1] * yz * (3.f * x2 - y2);
        float v18 = SH_C4[2] * xy * (7.f * z2 - 1.f);
        float v19 = SH_C4[3] * yz * (7.f * z2 - 3.f);
        float v20 = SH_C4[4] * (z2 * (35.f * z2 - 30.f) + 3.f);
        float v21 = SH_C4[5] * xz * (7.f * z2 - 3.f);
        float v22 = SH_C4[6] * (x2 - y2) * (7.f * z2 - 1.f);
        float v23 = SH_C4[7] * xz * (x2 - 3.f * y2);
        float v24 = SH_C4[8] * (x2 * (x2 - 3.f * y2) - y2 * (3.f * x2 - y2));
        v_coeffs[16 * num_channels + c] = v16 * v_values[c];
        v_coeffs[17 * num_channels + c] = v17 * v_values[c];
        v_coeffs[18 * num_channels + c] = v18 * v_values[c];
        v_coeffs[19 * num_channels + c] = v19 * v_values[c];
        v_coeffs[20 * num_channels + c] = v20 * v_values[c];
        v_coeffs[21 * num_channels + c] = v21 * v_values[c];
        v_coeffs[22 * num_channels + c] = v22 * v_values[c];
        v_coeffs[23 * num_channels + c] = v23 * v_values[c];
        v_coeffs[24 * num_channels + c] = v24 * v_values[c];

		if (degree <= 5) { continue; }

		v_coeffs[25 * num_channels + c] = v_values[c] * (0.65638205684017015f*y*(10.0f*x2*y2 - 5.0f*x4 - y4));
		v_coeffs[26 * num_channels + c] = v_values[c] * (8.3026492595241645f*xy*z*(x2 - y2));
		v_coeffs[27 * num_channels + c] = v_values[c] * (-0.48923829943525038f*y*(3.0f*x2 - y2)*(9.0f*z2 - 1.0f));                         // -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
		v_coeffs[28 * num_channels + c] = v_values[c] * (4.7935367849733241f*xy*z*(3.0f*z2 - 1.0f));                             // sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
		v_coeffs[29 * num_channels + c] = v_values[c] * (0.45294665119569694f*y*(14.0f*z2 - 21.0f*z4 - 1.0f));                            // sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
		v_coeffs[30 * num_channels + c] = v_values[c] * (0.1169503224534236f*z*(-70.0f*z2 + 63.0f*z4 + 15.0f));                          // sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
		v_coeffs[31 * num_channels + c] = v_values[c] * (0.45294665119569694f*x*(14.0f*z2 - 21.0f*z4 - 1.0f));                           // sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
		v_coeffs[32 * num_channels + c] = v_values[c] * (2.3967683924866621f*z*(x2 - y2)*(3.0f*z2 - 1.0f));                             // sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
		v_coeffs[33 * num_channels + c] = v_values[c] * (-0.48923829943525038f*x*(x2 - 3.0f*y2)*(9.0f*z2 - 1.0f));                        // -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
		v_coeffs[34 * num_channels + c] = v_values[c] * (2.0756623148810411f*z*(-6.0f*x2*y2 + x4 + y4));                       // 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
		v_coeffs[35 * num_channels + c] = v_values[c] * (0.65638205684017015f*x*(10.0f*x2*y2 - x4 - 5.0f*y4));                          // 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))


		if (degree <= 6) { continue; }

		v_coeffs[36 * num_channels + c] = v_values[c] * (1.3663682103838286f*xy*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4));                               // sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
		v_coeffs[37 * num_channels + c] = v_values[c] * (2.3666191622317521f*yz*(10.0f*x2*y2 - 5.0f*x4 - y4));                            // 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
		v_coeffs[38 * num_channels + c] = v_values[c] * (2.0182596029148963f*xy*(x2 - y2)*(11.0f*z2 - 1.0f));                             // 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
		v_coeffs[39 * num_channels + c] = v_values[c] * (-0.92120525951492349f*yz*(3.0f*x2 - y2)*(11.0f*z2 - 3.0f));                               // -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
		v_coeffs[40 * num_channels + c] = v_values[c] * (0.92120525951492349f*xy*(-18.0f*z2 + 33.0f*z4 + 1.0f));                           // sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
		v_coeffs[41 * num_channels + c] = v_values[c] * (0.58262136251873131f*yz*(30.0f*z2 - 33.0f*z4 - 5.0f));                            // sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
		v_coeffs[42 * num_channels + c] = v_values[c] * (6.6747662381009842f*z2 - 20.024298714302954f*z4 + 14.684485723822165f*z6 - 0.31784601133814211f);                        // sqrt(13)*(105*z2 - 315*z4 + 231*z6 - 5)/(32*sqrt(pi))
		v_coeffs[43 * num_channels + c] = v_values[c] * (0.58262136251873131f*xz*(30.0f*z2 - 33.0f*z4 - 5.0f));                            // sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
		v_coeffs[44 * num_channels + c] = v_values[c] * (0.46060262975746175f*(x2 - y2)*(11.0f*z2*(3.0f*z2 - 1.0f) - 7.0f*z2 + 1.0f));                               // sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2 + 1)/(64*sqrt(pi))
		v_coeffs[45 * num_channels + c] = v_values[c] * (-0.92120525951492349f*xz*(x2 - 3.0f*y2)*(11.0f*z2 - 3.0f));                               // -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
		v_coeffs[46 * num_channels + c] = v_values[c] * (0.50456490072872406f*(11.0f*z2 - 1.0f)*(-6.0f*x2*y2 + x4 + y4));                          // 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
		v_coeffs[47 * num_channels + c] = v_values[c] * (2.3666191622317521f*xz*(10.0f*x2*y2 - x4 - 5.0f*y4));                            // 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
		v_coeffs[48 * num_channels + c] = v_values[c] * (10.247761577878714f*x2*y4 - 10.247761577878714f*x4*y2 + 0.6831841051919143f*x6 - 0.6831841051919143f*y6);                         // sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))

		if (degree <= 7) { continue; }

		v_coeffs[49 * num_channels + c] = v_values[c] * (0.70716273252459627f*y*(-21.0f*x2*y4 + 35.0f*x4*y2 - 7.0f*x6 + y6));                              // 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 - 7*x6 + y6)/(64*sqrt(pi))
		v_coeffs[50 * num_channels + c] = v_values[c] * (5.2919213236038001f*xy*z*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4));                             // 3*sqrt(10010)*xy*z*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
		v_coeffs[51 * num_channels + c] = v_values[c] * (-0.51891557872026028f*y*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + 5.0f*x4 + y4));                          // -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 + 5*x4 + y4)/(64*sqrt(pi))
		v_coeffs[52 * num_channels + c] = v_values[c] * (4.1513246297620823f*xy*z*(x2 - y2)*(13.0f*z2 - 3.0f));                           // 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
		v_coeffs[53 * num_channels + c] = v_values[c] * (-0.15645893386229404f*y*(3.0f*x2 - y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f));                              // -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
		v_coeffs[54 * num_channels + c] = v_values[c] * (0.44253269244498261f*xy*z*(-110.0f*z2 + 143.0f*z4 + 15.0f));                              // 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
		v_coeffs[55 * num_channels + c] = v_values[c] * (0.090331607582517306f*y*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f));                              // sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
		v_coeffs[56 * num_channels + c] = v_values[c] * (0.068284276912004949f*z*(315.0f*z2 - 693.0f*z4 + 429.0f*z6 - 35.0f));                              // sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
		v_coeffs[57 * num_channels + c] = v_values[c] * (0.090331607582517306f*x*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f));                              // sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
		v_coeffs[58 * num_channels + c] = v_values[c] * (0.07375544874083044f*z*(x2 - y2)*(143.0f*z2*(3.0f*z2 - 1.0f) - 187.0f*z2 + 45.0f));                         // sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2 - 1) - 187*z2 + 45)/(64*sqrt(pi))
		v_coeffs[59 * num_channels + c] = v_values[c] * (-0.15645893386229404f*x*(x2 - 3.0f*y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f));                              // -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
		v_coeffs[60 * num_channels + c] = v_values[c] * (1.0378311574405206f*z*(13.0f*z2 - 3.0f)*(-6.0f*x2*y2 + x4 + y4));                         // 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
		v_coeffs[61 * num_channels + c] = v_values[c] * (-0.51891557872026028f*x*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + x4 + 5.0f*y4));                          // -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 + x4 + 5*y4)/(64*sqrt(pi))
		v_coeffs[62 * num_channels + c] = v_values[c] * (2.6459606618019f*z*(15.0f*x2*y4 - 15.0f*x4*y2 + x6 - y6));                               // 3*sqrt(10010)*z*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
		v_coeffs[63 * num_channels + c] = v_values[c] * (0.70716273252459627f*x*(-35.0f*x2*y4 + 21.0f*x4*y2 - x6 + 7.0f*y6));                              // 3*sqrt(715)*x*(-35*x2*y4 + 21*x4*y2 - x6 + 7*y6)/(64*sqrt(pi))

    }
}

__global__ void eval_sh_forward_kernel(
    const unsigned num_points,
	const unsigned num_channels,
    const unsigned degree,
    const unsigned degrees_to_use,
    const float3* __restrict__ viewdirs,
    const float* __restrict__ coeffs,
    float* __restrict__ colors
) {
	const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_points) {
        return;
    }
    unsigned num_bases = num_sh_bases(degree);
    unsigned idx_sh = num_bases * num_channels * idx;
    unsigned idx_col = num_channels * idx;

    eval_sh(
        degrees_to_use, num_channels, viewdirs[idx], &(coeffs[idx_sh]), &(colors[idx_col])
    );
}

__global__ void eval_sh_backward_kernel(
    const unsigned num_points,
    const unsigned num_channels,
    const unsigned degree,
    const unsigned degrees_to_use,
    const float3* __restrict__ viewdirs,
    const float* __restrict__ v_colors,
    float* __restrict__ v_coeffs
) {
	const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_points) {
        return;
    }
    unsigned num_bases = num_sh_bases(degree);
    unsigned idx_sh = num_bases * num_channels * idx;
    unsigned idx_col = num_channels * idx;

    eval_sh_grad(
        degrees_to_use, num_channels, viewdirs[idx], &(v_colors[idx_col]), &(v_coeffs[idx_sh])
    );
}
