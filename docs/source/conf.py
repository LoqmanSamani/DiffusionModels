import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'TorchDiff'
author = 'Loghman Samani'

# Try to get version from package, fallback to manual version
try:
    from importlib.metadata import version as pkg_version
    release = version = pkg_version("torchdiff")
except Exception:
    # Fallback version if package import fails
    release = version = "2.5.0"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

autoclass_content = 'both'
autodoc_member_order = 'bysource'

# Mock heavy imports so RTD build doesn't break
autodoc_mock_imports = [
    "torch", "torchvision", "torchaudio",
    "lpips", "pytorch_fid", "transformers",
    "torchmetrics"
]

# Autodoc settings — only document members DEFINED in each module
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Show more warnings
suppress_warnings = []

