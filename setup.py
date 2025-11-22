from setuptools import Command, find_packages, setup

__lib_name__ = "scGES"
__lib_version__ = "1.0.0"
__description__ = "scGES: Integrating and mapping single-cell transcriptomics across the entire gene expression space"
__url__ = "https://github.com/szding/scGES"
__author__ = "Shuzhen Ding"
__author_email__ = "dszspur@xju.edu.cn"
__license__ = "MIT"
__keywords__ = ["Integration", "Mapping", "Deep learning", "MNN"]
__requires__ = ["requests",]

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['scGES'],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True
)