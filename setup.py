import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='coropy',
    version='0.0.1',
    author='Ante Lojic Kapetanovic',
    author_email='alojic00@fesb.hr',
    description='A set of Python modules for COVID-19 epidemics modeling',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/antelk/coropy',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy','scipy', 'scikit-learn', 'matplotlib', 'setuptools'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
        'Topic :: Scientific/Engineering :: Epidemiology',
        'Intended Audience :: Science/Research',
    ],
    python_requires='>=3.6',
)
