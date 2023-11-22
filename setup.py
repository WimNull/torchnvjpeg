from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



packages_name = 'torchnvjpeg'

setup(
    name=packages_name,
    version="0.2.0",
    description="Using gpu decode jpeg image.",
    author="wimnull",
    classifiers=[
        "Development Status :: 4 - Beta", "Environment :: GPU :: NVIDIA CUDA",
        "License :: OSI Approved :: BSD License", "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", "Operating System :: POSIX :: Linux", "Programming Language :: C++",
        "Programming Language :: Python", "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering", "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development", "Topic :: Software Development :: Libraries"
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.7.0',
    ],
    packages=find_packages(),
    package_data={packages_name:["*.pyi", "*.so"]},
    ext_modules=[
        CUDAExtension(name=packages_name+'.lib',
                      sources=['csrc/torchnvjpeg.cpp'],
                      # extra_compile_args=['-g', '-std=c++14', '-fopenmp'],
                      extra_compile_args=['-std=c++14'],
                      libraries=['nvjpeg'],
                      define_macros=[('PYBIND', None)]),
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)},
)

