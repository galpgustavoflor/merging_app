from setuptools import setup, find_packages

setup(
    name="merging_app",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "streamlit",
        "pytest",
        # Add other dependencies as needed
    ],
)
