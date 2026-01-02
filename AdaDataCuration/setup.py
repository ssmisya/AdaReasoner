from setuptools import setup, find_packages
import os
setup(
    name='tool_data_curation',              # 包名
    version='0.1.0',                # 版本号
    packages=find_packages(),       # 自动查找所有子包
    install_requires=[],            # 依赖列表（可填如 ['numpy', 'requests']）
    author='Mingyang Song',
    author_email='ssmisya14@gmail.com',
    description='A simple Python package',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/ssmisya',  # 可选
    classifiers=[                   # 可选的分类标签
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.10',
)