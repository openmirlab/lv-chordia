"""Resolve installed or source-tree resources without creating directories.

Model weights live separately from the vendored toolkit's writable cache.
The model loader and checkpoint inspection share WEIGHTS_PATH.

Reads: mir/settings.py, sys.prefix
"""

import os
import sys
from .settings import *

# Use the package directory instead of current working directory
PACKAGE_PATH=os.path.dirname(os.path.abspath(__file__))
WORKING_PATH=os.path.dirname(os.path.dirname(PACKAGE_PATH))  # Go up two levels to package root

def _resource_path(directory):
    shared_path = os.path.join(sys.prefix, 'share', 'lv-chordia', directory)
    if os.path.exists(shared_path):
        return shared_path
    return os.path.join(WORKING_PATH, directory)


WEIGHTS_PATH = _resource_path('weights')
# Preserve the vendored toolkit's cache location; it contains no model weights.
CACHE_DATA_PATH = _resource_path('cache_data')
