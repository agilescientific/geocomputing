from setuptools import setup

setup(
    name='hello',
    version='0.1',
    py_modules=['hello'],
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
            'hello=hello:main'
        ]
    },
)
