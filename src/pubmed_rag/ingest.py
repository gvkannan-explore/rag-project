import argparse
import json
import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import httpx
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Request URLs contain credentials.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def parsed_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rag-ingest",
        description="PubMed Corpus ingestion to `corpus_date.jsonl` for a given query config",
    )

    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Directory to store the generated corpus from PubMed",
    )

    parser.add_argument(
        "--query_config",
        required=True,
        type=Path,
        help="Path to the YAML query configuration",
    )

    parser.add_argument(
        "--timeout",
        required=True,
        type=int,
        help="Client timeout in seconds",
        default=60,
    )

    return parser.parse_args()


def main() -> None:
    args = parsed_args()
    save_date = datetime.now().strftime("%b%Y")
    ncbi_email = os.environ.get("NCBI_EMAIL")
    if not ncbi_email or not ncbi_email.strip():
        raise ValueError("Set NCBI_EMAIL to a nonempty email before ingestion.")

    with open(args.query_config, "r") as fp:
        query_config = yaml.safe_load(fp)

    auth = {
        "email": ncbi_email,
        "api_key": os.environ.get("NCBI_API_KEY", ""),
    }
    rejected_abstracts: list[dict[str, str]] = []
    saved_pmids: list[str | None] = []
    logger.info("Loading client")
    with httpx.Client(timeout=args.timeout) as c:
        r = c.get(
            f"{BASE}/esearch.fcgi",
            params={**auth, **query_config},
        )

        pmids = sorted(r.json()["esearchresult"]["idlist"], key=int)

        with (args.save_dir / f"corpus_{save_date}.jsonl").open("w") as out_fp:
            logger.info("Query Client")
            for i in range(0, len(pmids), 200):
                r = c.post(
                    f"{BASE}/efetch.fcgi",
                    data={
                        **auth,
                        "db": "pubmed",
                        "id": ",".join(pmids[i : i + 200]),
                        "retmode": "xml",
                    },
                )
                for art in ET.fromstring(r.content).iter("PubmedArticle"):
                    pmid = art.findtext(".//MedlineCitation/PMID")
                    abstract = "\n".join(
                        (f"{a.get('Label')}: " if a.get("Label") else "")
                        + "".join(a.itertext())
                        for a in art.iter("AbstractText")
                    )
                    if not abstract:
                        rejected_abstracts.append(
                            {
                                "pmid": pmid
                                if pmid is not None
                                else "Failed_to_Find_with_regex",
                                "reason": "Invalid abstract",
                            }
                        )
                        continue

                    saved_pmids.append(pmid)
                    title = art.find(".//ArticleTitle")
                    out_fp.write(
                        json.dumps(
                            {
                                "pmid": pmid,
                                "title": "".join(
                                    title.itertext()  # type: ignore[reportOptionalMemberAccess]
                                ),
                                "abstract": abstract,
                                "journal": art.findtext(".//Journal/Title"),
                                "pub_year": art.findtext(
                                    ".//JournalIssue/PubDate/Year"
                                ),
                                "pub_types": [
                                    p.text for p in art.iter("PublicationType")
                                ],
                                "mesh_terms": [
                                    d.text for d in art.iter("DescriptorName")
                                ],
                                "doi": art.findtext(
                                    ".//PubmedData/ArticleIdList/ArticleId[@IdType='doi']"
                                ),
                            }
                        )
                        + "\n"
                    )

    logger.info("Corpus successfully saved")
    with (args.save_dir / f"rejectedAbstract_{save_date}.json").open("w") as out_fp:
        json.dump(rejected_abstracts, out_fp, allow_nan=False)
    logger.info("Rejected abstracts saved")

    if len(set(saved_pmids)) < len(saved_pmids):
        logger.warning(
            "Duplicate PMIDs found: total=%d, unique=%d",
            len(saved_pmids),
            len(set(saved_pmids)),
        )


if __name__ == "__main__":
    main()
