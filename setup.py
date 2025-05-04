from setuptools import setup, find_packages

setup(
    name="RPSLearner",
    version="1.0.0",
    description="A novel approach based on random projection and deep stacking learning for categorizing NSCLC",
    url="https://github.com/wan-mlab/RPSLearner",
    author="Xinchao Wu, Shibiao Wan",
    author_email="xwu@unmc.edu",
    license="MIT",
    packages=find_packages(where='./RPSLearner'),
    package_dir={
        '': 'RPSLearner'
    },
    include_package_data=True,
    install_requires=[
        "scikit-learn>=1.2.1",
        "scipy>=1.7.3",
        "xgboost>=1.7.0",
        "torch>=1.0",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.8"
)
