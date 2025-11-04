import os
from .settings import *

# Use the package directory instead of current working directory
PACKAGE_PATH=os.path.dirname(os.path.abspath(__file__))
WORKING_PATH=os.path.dirname(os.path.dirname(PACKAGE_PATH))  # Go up two levels to package root

DEFAULT_DATA_STORAGE_PATH=DEFAULT_DATA_STORAGE_PATH.replace('$project_name$',os.path.basename(WORKING_PATH))
