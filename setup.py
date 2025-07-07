from setuptools import find_packages, setup

setup(
    name="cryo_experiments",
    version="1.0.0",
    author="Octavian-Vlad Murad",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
