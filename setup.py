from ast import literal_eval
from distutils.core import setup


def get_version(source='src/anhima/__init__.py'):
    with open(source) as f:
        for line in f:
            if line.startswith('__version__'):
                return literal_eval(line.split('=')[-1].lstrip())
    raise ValueError("__version__ not found")


setup(
    name='anhima',
    version=get_version(),
    author='Alistair Miles',
    author_email='alimanfoo@googlemail.com',
    package_dir={'': 'src'},
    packages=['anhima'],
    url='https://github.com/alimanfoo/anhima',
    license='MIT License',
    description='Exploration and Analysis of genetic variation data.',
    long_description=open('README.md').read(),
    classifiers=['Intended Audience :: Developers',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Topic :: Software Development :: Libraries :: Python Modules'
                 ]
)
