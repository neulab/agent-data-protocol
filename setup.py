from setuptools import find_namespace_packages, setup

setup(
    name="agent-data-protocol",
    version="0.0.0",
    python_requires=">=3.12",
    packages=find_namespace_packages(include=["agents*", "schema*", "scripts*"]),
)
