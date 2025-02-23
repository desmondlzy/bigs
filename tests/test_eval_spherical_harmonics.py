#%%
import unittest
import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from relight.eval_spherical_harmonics import eval_spherical_harmonics
from gsplat.cuda._wrapper import spherical_harmonics as gsplat_sh

import torch
import numpy as np

import tinycudann as tcnn


class TestEvalSphericalHarmonics(unittest.TestCase):
	def test_easy_run(self):
		degree = 3
		degree_to_use = 3
		num_points = 100
		num_channels = 3

		coeffs = torch.randn(num_points, degree ** 2, num_channels).cuda()
		dirs = torch.randn(num_points, 3).cuda()
		dirs /= torch.norm(dirs, dim=1, keepdim=True)

		my_sh_vals = eval_spherical_harmonics(degree, degree_to_use, dirs, coeffs)

		gsplat_sh_vals = gsplat_sh(degree_to_use - 1, dirs, coeffs)

		self.assertTrue(torch.allclose(my_sh_vals, gsplat_sh_vals, atol=1e-6))

	def test_single_channel_forward(self):
		degree = 3
		degree_to_use = 2
		num_points = 100
		num_channels = 1

		coeffs = torch.randn(num_points, degree ** 2, num_channels).cuda()
		dirs = torch.randn(num_points, 3).cuda()
		dirs /= torch.norm(dirs, dim=1, keepdim=True)

		my_sh_vals = eval_spherical_harmonics(degree, degree_to_use, dirs, coeffs)

		coeffs_dup = torch.stack([coeffs[:, :, 0] for i in range(3)], dim=-1)
		gsplat_sh_vals = gsplat_sh(degree_to_use - 1, dirs, coeffs_dup)[:, 0].unsqueeze(-1)

		self.assertTrue(torch.allclose(my_sh_vals, gsplat_sh_vals, atol=1e-6))


	def test_single_channel_backward(self):
		degree = 3
		degree_to_use = 2
		num_points = 100
		num_channels = 1

		coeffs = torch.randn(num_points, degree ** 2, num_channels, device="cuda", requires_grad=True)

		dirs = torch.randn(num_points, 3).cuda()
		dirs /= torch.norm(dirs, dim=1, keepdim=True)

		my_sh_vals = eval_spherical_harmonics(degree, degree_to_use, dirs, coeffs).reshape(-1)

		weights = torch.rand((num_points, ), device="cuda") * 10

		total = (my_sh_vals * weights).sum()
		total.backward()

		my_grad = coeffs.grad.clone()

		coeffs.grad = None

		coeffs_dup = torch.stack([coeffs[:, :, 0] for i in range(3)], dim=-1)
		gsplat_sh_vals = gsplat_sh(degree_to_use - 1, dirs, coeffs_dup)[:, 0].reshape(-1)

		gsplat_total = (gsplat_sh_vals * weights).sum()
		gsplat_total.backward()

		gsplat_grad = coeffs.grad.clone()

		self.assertTrue(torch.allclose(my_grad, gsplat_grad, atol=1e-6))

	def test_many_channels_forward(self):
		degree = 5
		degree_to_use = 4
		num_points = 1000
		num_channels = 9

		coeffs = torch.randn(num_points, degree ** 2, num_channels).cuda()

		dirs = torch.randn(num_points, 3).cuda()
		dirs /= torch.norm(dirs, dim=1, keepdim=True)	

		my_sh_vals = eval_spherical_harmonics(degree, degree_to_use, dirs, coeffs)

		gsplat_sh_vals = torch.zeros_like(my_sh_vals)
		for i in range(0, num_channels, 3):
			gsplat_sh_vals[:, i:i + 3] = gsplat_sh(degree_to_use - 1, dirs, coeffs[:, :, i:i+3])
		
		self.assertTrue(torch.allclose(my_sh_vals, gsplat_sh_vals, atol=1e-6))

	def test_against_tinycudann(self):
		degree = 7
		degree_to_use = degree
		num_points = 1000
		num_channels = 9

		coeffs = torch.randn(num_points, degree ** 2, num_channels).cuda()
		dirs = torch.randn(num_points, 3).cuda()
		dirs /= torch.norm(dirs, dim=1, keepdim=True)

		my_sh_vals = eval_spherical_harmonics(degree, degree_to_use, dirs, coeffs)

		sph_tinycudann = tcnn.Encoding(
			n_input_dims=3,
			encoding_config={
				"otype": "SphericalHarmonics",
				"degree": degree,
			},
			dtype=torch.float32,
		)

		sh_basis_vals = sph_tinycudann((dirs + 1) / 2)
		tcnn_sh_vals = torch.einsum("ijk,ij->ik", coeffs, sh_basis_vals)

		self.assertTrue(torch.allclose(my_sh_vals, tcnn_sh_vals, atol=1e-4))


TestEvalSphericalHarmonics().test_many_channels_forward()
TestEvalSphericalHarmonics().test_against_tinycudann()

