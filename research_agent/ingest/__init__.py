"""Real document corpora -> A-RAG chunks and indexes.

sources (directory / git ref) -> loaders (Markdown) -> chunker (heading-aware, token limit)
-> index (incremental, upstream on-disk format).
"""
