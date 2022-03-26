from setuptools import setup

setup(
    name="nlp_utils",
    version="0.0.1",
    url="https://gitlab.lrz.de/explainable-ai/current-projects/global-xai-methods",
    author="Simon Klimek",
    author_email="simon.klimek@tum.de",
    description="helper lib",
    packages=["nlp_utils"],
    package_data={},
    license="MIT",
    long_description_content_type="text/markdown",
    long_description=""" helper lib """,
    install_requires=["pandas>=1.2.4"],
)
