#import os
#import sys
#sys.path.insert(0, os.path.abspath('/home/loqman/Downloads/projs/TorchDiff'))  # Points to TorchDiff directory
import os
import sys
from importlib.metadata import version as pkg_version



sys.path.insert(0, os.path.abspath('../..'))  # Points to TorchDiff from docs/source/


project = 'TorchDiff'
copyright = '2025, Loghman Samani'
author = 'Loghman Samani'
release = pkg_version("torchdiff")  # reads __version__ from installed package
version = release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# Autodoc settings
autoclass_content = 'both'
autodoc_member_order = 'bysource'

autodoc_mock_imports = ["torch", "torchvision", "torchaudio", "lpips", "pytorch_fid", "transformers"]


