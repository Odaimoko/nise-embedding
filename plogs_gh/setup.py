import setuptools


with open('README.md', 'r') as readme:
    long_description = readme.read()


setuptools.setup(
    name="plogs",
    version="0.0.1",
    author="Doug Rudolph",
    author_email="drudolph914@gmail.com",
    description="Bring color and formatting to your logs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/11/plogs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
