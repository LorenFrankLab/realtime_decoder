from setuptools import setup, find_packages

# long_description = read('README.rst')

def get_version_string():
    version = {}
    with open("realtime_decoder/version.py") as f:
        exec(f.read(), version)
    return version["__version__"]

setup(
    name='realtime_decoder',
    version=get_version_string(),
    license='Apache License',
    author='Joshua Chu',
    python_requires='>=3.7',
    install_requires=['ghostipy>=0.2.0',
                      'pandas',
                      'scipy',
                      'mpi4py',
                      'Cython',
                      'trodesnetwork',
                      'pyqtgraph',
                      'oyaml'
                     ],
    author_email='jpc6@rice.edu',
    description='Realtime clusterless decoding',
    packages=find_packages(),
    keywords="neuroscience clusterless decoding",
    include_package_data=True,
    platforms='any',
    classifiers=['Programming Language :: Python :: 3'],
)
