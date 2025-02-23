import torch
import os

from torch.utils.cpp_extension import load
dirname = os.path.join(os.path.dirname(__file__), "src")
build_dir = os.path.join(os.path.dirname(__file__), "build")

if not os.path.exists(build_dir):
	os.makedirs(build_dir)

sources = [
	os.path.join(dirname, "bindings.cpp"),
	# os.path.join(dirname, "spherical_harmonics.cuh"),
	# os.path.join(dirname, "tensors.cuh"),
	os.path.join(dirname, "tensors.cu"),
]

_relight_impl = load(
	name="_relight_impl", 
	sources=sources,  
	build_directory=build_dir, 
	verbose=True)
