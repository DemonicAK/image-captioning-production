"""I/O utilities.

This module provides file I/O utilities for
saving and loading training artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path.
        
    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save data to JSON file.
    
    Args:
        data: Dictionary to save.
        path: Output file path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load data from JSON file.
    
    Args:
        path: Input file path.
        
    Returns:
        Loaded dictionary.
        
    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
