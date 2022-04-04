from setuptools import setup

setup(
    name="nlp_utils",
    version="0.0.1",
    url="https://github.com/mscholl96/mad-recime",
    author="Hannes Schatz",
    author_email="hannesmarc.schatz@gmail.com",
    description="helper lib",
    packages=["nlp_utils"],
    package_data={},
    long_description_content_type="text/markdown",
    long_description=""" helper lib """,
    install_requires=["pandas>=1.2.4"],
)