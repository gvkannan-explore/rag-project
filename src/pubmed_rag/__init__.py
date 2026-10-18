import argparse

from pubmed_rag.retrieval import METHODS
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
    search_parser = subparsers.add_parser(
        "search",
        help="Search the frozen corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n  rag search "HER2 treatment" --method B\n'
        '  rag search "HER2 treatment" --method C --limit 20 --json '
        "--allow-paid-query --query-budget-usd 0.001\n"
        '  rag search "HER2 treatment" --method D '
        "--allow-paid-query --query-budget-usd 0.001",
    )
    search_parser.add_argument("query", help="Question or search terms")
    search_parser.add_argument(
        "--method",
        choices=(*METHODS.values(), *METHODS),
        default="bm25",
        help="A=dense, B=BM25, C=hybrid, D=hybrid with reranking",
    )
    search_parser.add_argument(
        "--allow-paid-query",
        action="store_true",
        help="Allow one paid question embedding request for A, C, or D",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Displayed results, default 5; use 20 for labeling or 50 for the C/D pool",
    )
    search_parser.add_argument(
        "--json", action="store_true", help="Print results and provenance as JSON"
    )
    search_parser.add_argument(
        "--query-budget-usd",
        type=float,
        help="Maximum estimated USD for this one-request dense query; required with --allow-paid-query",
    )
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
    try:
        search_main(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
