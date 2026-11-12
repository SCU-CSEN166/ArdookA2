"""Run grant recommendations on the real workbook or seeded synthetic data."""

import argparse
from dataclasses import asdict
from datetime import date
import json
import sys

from src.ardook_extra_synthetic_data import (
    DB_PATH as SYNTHETIC_DB,
    REFERENCE_DATE,
    load_checker as load_synthetic_checker,
    load_grants as load_synthetic_grants,
    load_profiles as load_synthetic_profiles,
)
from src.deepseek_extractor import (
    DEFAULT_MODEL as DEFAULT_DEEPSEEK_MODEL,
    RANKING_FIELDS as DEEPSEEK_RANKING_FIELDS,
    DeepSeekGrantExtractor,
    GrantDetailExtractionError,
    grant_source_text,
)
from src.pipeline import (
    DEFAULT_MIN_SIMILARITY,
    LocalGenerator,
    OrganizationProfile,
    Pipeline,
)
from src.vector_store import VectorStore
from src.ingest import load_grants
from src.real_eligibility import load_checker as load_real_checker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("real", "synthetic"), default="real")
    parser.add_argument("--description", help="Organization mission and program description.")
    parser.add_argument(
        "--attributes",
        default="{}",
        help='JSON organization facts, e.g. \'{"entity_type":"nonprofit","region":"US"}\'.',
    )
    parser.add_argument(
        "--profile-id",
        type=int,
        help="Use a generated synthetic profile by ID (synthetic dataset only).",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help="Minimum semantic similarity from 0 to 1 (default: %(default)s).",
    )
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--explain", action="store_true", help="Generate advisory fit notes.")
    parser.add_argument(
        "--deepseek-extract",
        action="store_true",
        help="Locally extract source-grounded grant details with DeepSeek and use them as extra ranking text.",
    )
    parser.add_argument(
        "--deepseek-model",
        default=DEFAULT_DEEPSEEK_MODEL,
        help="Hugging Face model ID used with --deepseek-extract.",
    )
    parser.add_argument(
        "--deepseek-cache",
        default=".cache/deepseek_grant_details.json",
        help="Local JSON cache path for validated DeepSeek extractions.",
    )
    args = parser.parse_args()

    try:
        attributes = json.loads(args.attributes)
    except json.JSONDecodeError as error:
        parser.error(f"--attributes must be valid JSON: {error}")
    if not isinstance(attributes, dict):
        parser.error("--attributes must be a JSON object")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not 0 <= args.min_similarity <= 1:
        parser.error("--min-similarity must be between 0 and 1")
    if args.dataset == "synthetic":
        as_of = args.as_of or REFERENCE_DATE
        if args.profile_id is not None:
            profiles = load_synthetic_profiles(SYNTHETIC_DB)
            selected = profiles.loc[profiles["id"] == args.profile_id]
            if selected.empty:
                parser.error(f"No synthetic profile with id {args.profile_id}")
            row = selected.iloc[0]
            profile = OrganizationProfile(row["description"], row["attributes"])
        elif args.description:
            profile = OrganizationProfile(args.description, attributes)
        else:
            parser.error("Provide --profile-id or --description for synthetic recommendations")
        grants = load_synthetic_grants(SYNTHETIC_DB)
        checker = load_synthetic_checker(SYNTHETIC_DB)
        store = VectorStore(path="vector_store_synthetic")
    else:
        as_of = args.as_of or date.today()
        if args.profile_id is not None:
            parser.error("--profile-id is available only with --dataset synthetic")
        if not args.description:
            parser.error("--description is required")
        profile = OrganizationProfile(args.description, attributes)
        grants = load_grants()
        checker = load_real_checker(grants)
        store = VectorStore()

    additional_chunks = None
    extracted_details = None
    if args.deepseek_extract:
        extractor = DeepSeekGrantExtractor(
            args.deepseek_model, cache_path=args.deepseek_cache
        )
        additional_chunks = {}
        extracted_details = {}
        grants_processed = 0
        for grant in grants.to_dict("records"):
            source = grant_source_text(grant)
            try:
                details = extractor.extract(source)
            except GrantDetailExtractionError as error:
                # One unusable model answer should not abort the whole run; rank this grant
                # on its workbook text alone, like a grant with no validated fields.
                print(
                    f"DeepSeek skipped source row {grant['source_row']} "
                    f"({grant['program']}): {error}",
                    file=sys.stderr,
                )
                continue
            extracted_details[int(grant["source_row"])] = details
            chunks = [
                f"{field}: {details[field]}"
                for field in DEEPSEEK_RANKING_FIELDS
                if details[field] is not None
            ]
            if chunks:
                additional_chunks[int(grant["source_row"])] = chunks
            grants_processed += 1
        print(
            f"DeepSeek processed {grants_processed} grants; "
            f"{extractor.cache_hits} cache hits, {extractor.model_calls} model calls. "
            "Validated extracted details are included in recommendation evidence.",
            file=sys.stderr,
        )

    pipeline = Pipeline(
        grants=grants,
        checker=checker,
        store=store,
        additional_chunks=additional_chunks,
        extracted_details=extracted_details,
        generator=LocalGenerator() if args.explain else None,
    )
    recommendations = pipeline.recommend(
        profile,
        as_of=as_of,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )
    print(json.dumps([asdict(item) for item in recommendations], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
