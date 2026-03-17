# agent.rag — retrieval-augmented generation (document store, search, QA)
#
# Imports are deferred via __all__ + __getattr__ so that
# `python -m agentforge.rag.document_store` doesn't trigger a double-import
# warning.  Normal `from agentforge.rag import search_docs` still works.

__all__ = [
    "chunk_text",
    "load_corpus",
    "save_corpus",
    "add_document",
    "search_docs",
    "ingest_file",
    "answer_from_docs",
]


def __getattr__(name):
    _ds_names = {
        "chunk_text", "load_corpus", "save_corpus",
        "add_document", "search_docs", "ingest_file",
    }
    if name in _ds_names:
        from agentforge.rag import document_store as _ds
        for _n in _ds_names:
            globals()[_n] = getattr(_ds, _n)
        return globals()[name]

    if name == "answer_from_docs":
        from agentforge.rag.qa import answer_from_docs
        globals()["answer_from_docs"] = answer_from_docs
        return answer_from_docs

    raise AttributeError(f"module 'agentforge.rag' has no attribute {name!r}")
