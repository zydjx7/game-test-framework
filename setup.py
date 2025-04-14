from setuptools import setup, find_packages

setup(
    name="game-testing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "pyyaml>=6.0",
        "behave>=1.2.6",
        "pytest>=7.0.0",
        "python-dotenv>=0.19.0",
        "requests>=2.26.0",
        "aiohttp>=3.8.0",
        "loguru>=0.5.3",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "pillow>=8.0.0"
    ],
    python_requires=">=3.8",
)