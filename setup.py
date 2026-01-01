from setuptools import setup, find_namespace_packages

setup(
    name="orange3-custom-addon",
    version="0.2.0",
    packages=find_namespace_packages(include=["orangecontrib.*"]),
    include_package_data=True,
    install_requires=[
        "Orange3",
        "scikit-learn",
        "matplotlib",
        "hmmlearn",
        "numpy<2",
        "yfinance",
        "lightgbm",
        "optuna"
    ],
    entry_points={
        'orange3.addon': [
            'custom = orangecontrib.custom',
        ],
        'orange.widgets': [
            'Custom = orangecontrib.custom.widgets',
        ],
    },
    package_data={
        'orangecontrib.custom.widgets': ['icons/*.svg'],
    },
    zip_safe=False,
    python_requires=">=3.8",
    classifiers=[
        "Framework :: Orange3",
        "License :: OSI Approved :: MIT License",
    ],
)
