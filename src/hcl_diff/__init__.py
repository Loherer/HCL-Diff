"""HCL-Diff research implementation."""

from .config import load_config
from .variants import VariantSpec, get_variant

__all__ = ["VariantSpec", "get_variant", "load_config"]
__version__ = "0.1.0"

