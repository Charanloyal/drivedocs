"""DriveDocs: a multimodal RAG assistant for vehicle manuals.

A LangChain tool-calling agent answers questions over owner's-manual text and diagrams using MiniLM and CLIP embeddings
in two FAISS indexes, with verified page-level citations and guardrails against prompt injection and PII leaks.
"""

__version__ = "0.3.0"
__author__ = "Charanloyal"
