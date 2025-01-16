# setup.py
import setuptools

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="forecaster",                 # Replace with your desired package name
    version="0.1.0",                   # Choose any initial version number
    author="Ruipu Li, Alexander Rodríguez",
    author_email="liruipu@umich.edu",
    description="Neural Conformal Control for Time Series Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/yourusername/my_package",  # optional if you have a GitHub repo
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)