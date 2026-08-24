#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   setup.py.py    
@Contact :   https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution/issues
@License :   (C)Copyright 2020-2100

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
9/6/21 5:16 PM   Haohe Liu      1.0         None
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = "audiosr"
DESCRIPTION = "Versatile audio super-resolution for speech, music, and sound."
URL = "https://github.com/1llum1n4t1s/KG.versatile_audio_super_resolution"
EMAIL = ""
AUTHOR = "Haohe Liu"
REQUIRES_PYTHON = ">=3.10,<3.15"
VERSION = "1.0.0"

# What packages are required for this module to be executed?
REQUIRED = [
    "torch>=2.13",
    "torchaudio>=2.11",
    "torchvision>=0.28",
    "tqdm>=4.70",
    "gradio>=6.25,<7",
    "pyyaml>=6.0.3",
    "einops>=0.8.2",
    "chardet>=7.6",
    "numpy>=2.2.6,<3",
    "soundfile>=0.14",
    'librosa>=0.11,<1; python_version < "3.12"',
    'librosa>=1,<2; python_version >= "3.12"',
    "scipy>=1.15.3,<2",
    "pandas>=2.3.3,<4",
    "unidecode>=1.4",
    "phonemizer>=3.4",
    "torchlibrosa>=0.1",
    "transformers>=5.15.1,<6",
    "huggingface-hub>=1.28,<2",
    "Pillow>=12.3,<13",
    "requests>=2.34.2,<3",
    "scikit-learn>=1.7.2,<2",
    "progressbar2>=4.6,<5",
    "ftfy>=6.3.1,<7",
    "timm>=1.0.28,<2",
    "matplotlib>=3.10.9,<4",
    "pyloudnorm>=0.2,<1",
    "safetensors>=0.8,<1",
]

# What packages are optional?
EXTRAS = {
    "test": ["pytest>=9.1.1,<10"],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    entry_points={
        "console_scripts": ["audiosr=audiosr.__main__:main"],
    },
)
