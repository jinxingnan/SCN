"""
Embeddings module for SCN.

Provides KGE loading and fusion operations.
"""

from .kge_loader import KGELoader
from .fusion import RotatEFusion, TransEFusion, get_fusion

__all__ = ["KGELoader", "RotatEFusion", "TransEFusion", "get_fusion"]
