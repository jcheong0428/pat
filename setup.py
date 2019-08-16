from setuptools import setup, find_packages

version = {}
with open("pat/version.py") as f:
    exec(f.read(), version)

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# try:
#     from setuptools.core import setup
# except ImportError:
#     from distutils.core import setup
extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name='py-pat',
    version=version['__version__'],
    author='Mind2019-SocialPose',
    author_email='jcheong.gr@dartmouth.edu',
    url='https://github.com/jcheong0428/pat',
    download_url = '',
    install_requires=requirements,
    extras_require = {
    'interactive_plots':['ipywidgets>=5.2.2']
    },
    packages=find_packages(exclude=['pat/tests']),
    package_data={'pat': ['resources/*']},
    license='LICENSE.txt',
    description='A Python package to analyze OpenPose data',
    long_description='Python Pose Analysis Toolbox (PPAT) provides tools and functions for loading, preprocessing, analyzing, and visualizing OpenPose data.',
    keywords = ['openpose', 'socialpose', 'pose'],
    classifiers = [
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"
    ],
    **extra_setuptools_args
)
