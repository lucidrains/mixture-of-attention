from setuptools import setup, find_packages

setup(
  name = 'mixture-of-attention',
  packages = find_packages(exclude=[]),
  version = '0.0.23',
  license='MIT',
  description = 'Mixture of Attention',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/mixture-of-attention',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'mixture-of-experts',
    'routed attention'
  ],
  install_requires=[
    'colt5-attention>=0.10.6',
    'einops>=0.6.1',
    'local-attention>=1.8.6',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
