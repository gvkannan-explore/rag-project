import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="PubMed RAG command-line tool."
    )
    parser.parse_args()
    
    print("Hello from rag-project!")
