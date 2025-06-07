"""
Climate RAG - A retrieval-augmented generation model for climate and energy data.

This module serves as the main entry point for the Climate RAG package,
providing access to all core functionality through simple imports.
"""

# Import modules to maintain clear namespaces
import agents
import api
import cache
import constants
import graph
import helpers
import llms
import prompts
import query_data
import schemas
import text_splitters
import tools
import webapp

# Package metadata
__version__ = "0.1.0"
__author__ = "Climate RAG Team"
__description__ = "A retrieval-augmented generation model for climate and energy data"

# Main exports - expose the modules
__all__ = [
    "query_data",
    "graph",
    "schemas",
    "tools",
    "agents",
    "api",
    "webapp",
    "helpers",
    "cache",
    "llms",
    "prompts",
    "text_splitters",
    "constants",
]
