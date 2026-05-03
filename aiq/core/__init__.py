"""AIQ Core — shared types, interfaces, and constants.

All modules import from here. No module defines its own domain vocabulary
or output format. Everything flows through these shared types.
"""

from .types import (
    Chunk,
    ChunkTag,
    ModuleOutput,
    DomainContext,
    TokenChange,
    TokenAccounting,
)

__all__ = [
    "Chunk",
    "ChunkTag",
    "ModuleOutput",
    "DomainContext",
    "TokenChange",
    "TokenAccounting",
]
