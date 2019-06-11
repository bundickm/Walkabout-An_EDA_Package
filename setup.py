'''
walkabout - a package for quick exploratory analysis on all features in a dataframe
'''
import setuptools

REQUIRED = [
]

with open('README.md','r') as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name = 'walkabout',
    version = '0.0.11',
    author = 'bundickm',
    description = 'A package for quick exploratory analysis on all features in a dataframe',
    long_description = LONG_DESCRIPTION,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/bundickm/walkabout',
    packages = setuptools.find_packages(),
    python_requires = '>= 3.5',
    install_requires = REQUIRED,
    classifier = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
)
