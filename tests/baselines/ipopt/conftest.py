"""Set KMP_DUPLICATE_LIB_OK on macOS before cyipopt loads.

Homebrew cyipopt and torch ship clashing libomp copies that abort the process.
"""

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
