# coding: utf-8

import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


requirements = open("requirements.txt").read().splitlines()
requirements.remove("numpy")
# Do not require numpy because it's better to install it
# from a system package to avoid compilation


setup(
    name="yalign",
    version="0.1",
    description="A tool to align comparable corpora",
    long_description=open('README.rst').read(),
    author="Rafael Carrascosa, Gonzalo Garcia Berrotaran, Andrew Vine",
    author_email="rafacarrascosa@gmail.com",
    url="https://github.com/machinalis/yalign",
    keywords=["align", "corpus", "corpus alignment"],
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Text Processing",
        "Topic :: Utilities",
    ],
    packages=["yalign"],
    install_requires=requirements,
    scripts=[os.path.join("scripts", x) for x in os.listdir("scripts")],
)
