import os
from setuptools import setup

VERSION = "1.0.0"

here = os.path.abspath(os.path.dirname(__file__))

print(os.listdir(f'{here}/MyRecbole'))

setup(
    name="Scalable Recommendations Transformer",
    version=VERSION,
    description="Experiments demonstrating scaling laws on sequential recommendations",
    author="Pablo Zivic, Jorge Sanchez, Hernan C. Vazquez",
    author_email="",
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    packages=["src"],
    install_requires=[
    ],
    tests_require=[

    ],
    entry_points={
    }
)
