import argparse

from pubmed_rag.search import main as search_main


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag", description="PubMed RAG command-line tool."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser(
        "index", help="Index cached document vectors in Postgres"
    )
    index_parser.add_argument("--corpus", default="data/corpus_Oct2026.jsonl")
    index_parser.add_argument("--selection", default="config/mvp_corpus.json")
    index_parser.add_argument("--archive", default="data/vectors/vector_embeddings.npz")
    index_parser.add_argument(
        "--database-url",
        default="postgresql://rag:rag_project@localhost:5432/pubmed_rag",
    )
    search_parser = subparsers.add_parser("search", help="Search the frozen corpus")
    search_parser.add_argument("query", help="Question or search terms")
    search_parser.add_argument("--method", choices=("bm25", "dense"), default="bm25")
    search_parser.add_argument(
        "--allow-paid-query",
        action="store_true",
        help="Allow one paid question embedding request for dense search",
    )
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--corpus", default="data/corpus_Oct2026.jsonl")
    search_parser.add_argument("--selection", default="config/mvp_corpus.json")
    search_parser.add_argument(
        "--archive", default="data/vectors/vector_embeddings.npz"
    )
    search_parser.add_argument(
        "--database-url",
        default="postgresql://rag:rag_project@localhost:5432/pubmed_rag",
    )
    args = parser.parse_args()
    search_main(args)
