import os
import sys
from .settings import *

# Use the package directory instead of current working directory
PACKAGE_PATH=os.path.dirname(os.path.abspath(__file__))
WORKING_PATH=os.path.dirname(os.path.dirname(PACKAGE_PATH))  # Go up two levels to package root

# Check for cache_data in shared data location (when installed via pip)
# Shared data is installed to: <prefix>/share/lv-chordia/cache_data
# We need to find the site-packages location and go up to find share/
CACHE_DATA_PATH = None
if hasattr(sys, 'prefix'):
    # Try shared data location first (for pip-installed packages)
    shared_cache_path = os.path.join(sys.prefix, 'share', 'lv-chordia', 'cache_data')
    if os.path.exists(shared_cache_path):
        CACHE_DATA_PATH = shared_cache_path
    else:
        # Fall back to WORKING_PATH location (for development/local installs)
        local_cache_path = os.path.join(WORKING_PATH, 'cache_data')
        if os.path.exists(local_cache_path):
            CACHE_DATA_PATH = local_cache_path
        else:
            # Default to WORKING_PATH for backward compatibility
            CACHE_DATA_PATH = os.path.join(WORKING_PATH, 'cache_data')

DEFAULT_DATA_STORAGE_PATH=DEFAULT_DATA_STORAGE_PATH.replace('$project_name$',os.path.basename(WORKING_PATH))
