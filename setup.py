from setuptools import find_packages, setup

setup(
    name="evidential_learning",
    packages=find_packages(exclude=["examples"]),
    version="0.0.1",
    license="MIT",
    description="Evidential Learning in Pytorch",
    author="Alisson Kolling",
    author_email="alikolling@gmail.com",
    long_description_content_type="text/markdown",
    url="https://github.com/alikolling/evidential-learning",
    install_requires=["torch"],
    classifiers=["Programming Language :: Python :: 3"],
)
