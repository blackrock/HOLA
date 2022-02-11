# Copyright 2021 BlackRock, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="hola",
    version="0.1",
    author="Gabriel Maher",
    author_email="gabriel.maher@blackrock.com",
    url="https://1A4D@dev.azure.com/1A4D/AI%20Labs/_git/blk_hyper",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    scripts=[
        "hola/hola_serve"
    ],
    packages=find_packages(include=['hola*']),
    install_requires=(Path(__file__).parent / 'requirements.txt').read_text(),
    python_requires='>=3.7',
    zip_safe=False,  # mypy requires this to be able to find the installed package
    package_data={'hola': ['py.typed']},  # for mypy
)
