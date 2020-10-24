import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Rulern-MatInGit", # Replace with your own username
    version="0.0.1",
    author="Mateusz Leputa",
    author_email="mateo.leputa@gmail.com",
    description="Learning Classifer Systems package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MatInGit/Rulern",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
