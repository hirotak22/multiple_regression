from setuptools import setup, find_packages

with open('LICENSE') as f:
    license = f.read()

setup(
    name='LinearRegression',
    version='1.0.0',
    license=license,
    install_requires=['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn', 'pyyaml'],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    entry_points={
        "console_scripts": ["linearreg=LinearRegression.main:main"]
    }
)