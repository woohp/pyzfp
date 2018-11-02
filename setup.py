#!/usr/bin/env python
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import glob


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


class get_numpy_include(object):
    def __str__(self):
        import numpy
        return numpy.get_include()


libraries = []  # add any libraries, such as sqlite3, here

zfp_sources = glob.glob('third_party/zfp/src/**/*.c', recursive=True)
zfp_sources = [s for s in zfp_sources if 'template' not in s and 'inline' not in s and 'cuda' not in s]

ext_modules = [
    Extension(
        'zfp', [
            'src/module.cpp',
        ] + zfp_sources,
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            get_numpy_include(),
            'third_party/zfp/include',
        ],
        libraries=libraries,
        language='c++'
    ),
]


class BuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type

        opts = ['-O2', '-march=native']
        if sys.platform == 'darwin':
            opts += ['-stdlib=libc++', '-mmacosx-version-min=10.8']

        if compiler_type == 'unix':
            opts.extend([
                '-DVERSION_INFO="{}"'.format(self.distribution.get_version()),
            ])

        original__compile = self.compiler._compile

        def new__compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith('.cpp'):
                extra_postargs = extra_postargs + ['-std=c++14']
            return original__compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = new__compile

        for ext in self.extensions:
            ext.extra_compile_args = opts

        try:
            build_ext.build_extensions(self)
        finally:
            del self.compiler._compile


setup(
    name='pyzfp',
    description='Zfp compression',
    version='0.1.0',
    setup_requires=['pybind11>=2.2.4'],
    install_requires=['pybind11>=2.2.4'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    test_suite='tests',
    zip_safe=False,
)
