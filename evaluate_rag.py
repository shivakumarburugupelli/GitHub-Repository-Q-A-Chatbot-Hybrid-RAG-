import argparse
import json
import logging
import os
import sys
import time
import warnings

# RAGAS anonymous usage tracking is switched off unless you set it yourself.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATASET_PATH = "eval_dataset.json"
DEFAULT_OUTPUT_PATH = "evaluation_results.csv"

# Judge model used by RAGAS (defaults to the same Groq model as the app).
# A larger Groq model usually gives more reliable scores, e.g. in .env:
#   EVAL_LLM_MODEL=llama-3.3-70b-versatile
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "")

# Groq rate limits are strict: keep the number of parallel judge calls low.
EVAL_MAX_WORKERS = int(os.getenv("EVAL_MAX_WORKERS", "1"))
EVAL_TIMEOUT_SECONDS = int(os.getenv("EVAL_TIMEOUT_SECONDS", "240"))
EVAL_MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "6"))

# RAGAS default is 3 generated questions per answer. 1 is cheaper / safer on
# providers that do not support multiple completions per call.
ANSWER_RELEVANCY_STRICTNESS = int(os.getenv("EVAL_ANSWER_RELEVANCY_STRICTNESS", "1"))

METRIC_COLUMNS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_test_dataset(path):
    """Return (github_url, samples). Each sample has 'question' and 'ground_truth'."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Test dataset not found: {path}. Create it or pass --dataset <file>."
        )
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    github_url = data.get("github_url", "")
    samples = data.get("samples", [])
    if not samples:
        raise ValueError(f"'{path}' contains no samples.")

    for index, sample in enumerate(samples, start=1):
        if not sample.get("question") or not sample.get("ground_truth"):
            raise ValueError(
                f"Sample {index} in '{path}' needs both 'question' and 'ground_truth'."
            )
    return github_url, samples


# ---------------------------------------------------------------------------
# Run the real RAG pipeline
# ---------------------------------------------------------------------------
def run_pipeline(rag, samples, delay):
    """Ask every question; return rows in the format RAGAS expects."""
    rows = []
    for index, sample in enumerate(samples, start=1):
        question = sample["question"]
        print(f"  [{index}/{len(samples)}] {question}")
        try:
            result = rag.answer_question(question)
        except Exception as exc:
            print(f"      ! skipped, the pipeline failed: {exc}")
            continue

        if not result["contexts"]:
            print("      ! skipped, no contexts were retrieved")
            continue

        rows.append(
            {
                "user_input": question,
                "response": result["answer"],
                "retrieved_contexts": result["contexts"],
                "reference": sample["ground_truth"],
                "sources": sorted({s["source"] for s in result["sources"]}),
            }
        )
        if delay and index < len(samples):
            time.sleep(delay)
    return rows


# ---------------------------------------------------------------------------
# RAGAS
# ---------------------------------------------------------------------------
def run_ragas(rag, rows):
    from langchain_groq import ChatGroq
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    judge_model = EVAL_LLM_MODEL or rag.LLM_MODEL
    print(f"  Judge LLM: {judge_model} (Groq)")
    print(f"  Embeddings for Answer Relevancy: {rag.EMBEDDING_MODEL}")

    judge_llm = LangchainLLMWrapper(ChatGroq(model=judge_model, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(rag.get_embeddings())

    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=ANSWER_RELEVANCY_STRICTNESS),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    ragas_rows = [
        {
            "user_input": row["user_input"],
            "response": row["response"],
            "retrieved_contexts": row["retrieved_contexts"],
            "reference": row["reference"],
        }
        for row in rows
    ]

    result = evaluate(
        dataset=EvaluationDataset.from_list(ragas_rows),
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(
            timeout=EVAL_TIMEOUT_SECONDS,
            max_retries=EVAL_MAX_RETRIES,
            max_workers=EVAL_MAX_WORKERS,
        ),
        raise_exceptions=False,   # a failed metric becomes NaN instead of crashing
        show_progress=True,
    )

    frame = result.to_pandas()
    frame = frame.rename(
        columns={"llm_context_precision_with_reference": "context_precision"}
    )
    return frame


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def shorten(text, width):
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def show_and_save(frame, rows, output_path):
    import pandas as pd

    metric_columns = [c for c in METRIC_COLUMNS if c in frame.columns]

    # Add the retrieved source files and store contexts as readable text.
    frame = frame.copy()
    frame["sources"] = [", ".join(row["sources"]) for row in rows]
    frame["retrieved_contexts"] = [
        "\n---\n".join(row["retrieved_contexts"]) for row in rows
    ]

    ordered = ["user_input", "response", "reference", "retrieved_contexts", "sources"]
    frame = frame[ordered + metric_columns]
    frame.to_csv(output_path, index=False, encoding="utf-8")

    summary = pd.DataFrame(
        {
            "metric": metric_columns,
            "mean_score": [frame[c].mean(skipna=True) for c in metric_columns],
            "failed_samples": [int(frame[c].isna().sum()) for c in metric_columns],
        }
    )
    stem, extension = os.path.splitext(output_path)
    summary_path = f"{stem}_summary{extension or '.csv'}"
    summary.to_csv(summary_path, index=False)

    line = "=" * 78
    print(f"\n{line}\nPER-QUESTION SCORES\n{line}")
    table = frame[metric_columns].copy()
    table.insert(0, "question", [shorten(q, 38) for q in frame["user_input"]])
    print(table.to_string(index=True, float_format=lambda v: f"{v:.3f}"))

    print(f"\n{line}\nAVERAGE SCORES\n{line}")
    for _, row in summary.iterrows():
        score = "n/a" if pd.isna(row["mean_score"]) else f"{row['mean_score']:.3f}"
        note = f"  ({row['failed_samples']} sample(s) failed)" if row["failed_samples"] else ""
        print(f"  {row['metric']:<20} {score}{note}")

    print(f"\nSaved per-question results : {output_path}")
    print(f"Saved average scores       : {summary_path}")

    if summary["failed_samples"].sum():
        print(
            "\nWARNING: some metric calls failed (shown as NaN). Usually this is a "
            "Groq rate limit or a judge model that returned invalid JSON. "
            "Try EVAL_MAX_WORKERS=1 or a larger EVAL_LLM_MODEL in your .env file."
        )

    # False when no metric could be computed at all.
    return bool(summary["mean_score"].notna().any())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Test dataset JSON file.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="CSV file for the results.")
    parser.add_argument("--repo", default=None, help="GitHub URL (overrides 'github_url' in the dataset).")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking (baseline run).")
    parser.add_argument("--top-n", type=int, default=None, help="Number of chunks sent to the LLM.")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between questions.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY is missing. Add it to your .env file (see .env.example).")
        return 1
    if not os.getenv("GITHUB_TOKEN"):
        print("Note: GITHUB_TOKEN is not set. GitHub allows only 60 API requests per hour "
              "without a token; set one in .env if loading fails with a rate-limit error.")

    try:
        github_url, samples = load_test_dataset(args.dataset)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    github_url = args.repo or github_url
    if not github_url:
        print("ERROR: no GitHub URL. Put 'github_url' in the dataset or pass --repo.")
        return 1
    if args.limit:
        samples = samples[: args.limit]

    try:
        import rag
        from github_loader import GitHubLoaderError
    except ImportError as exc:
        print(f"ERROR: could not import the project modules ({exc}). "
              "Run this script from the project folder after `pip install -r requirements.txt`.")
        return 1

    if args.no_rerank:
        rag.RERANK_ENABLED = False
    if args.top_n:
        rag.RERANK_TOP_N = args.top_n

    print("=" * 78)
    print("RAG EVALUATION")
    print("=" * 78)
    print(f"Repository : {github_url}")
    print(f"Questions  : {len(samples)}")
    print(f"Reranking  : {'ON  (' + rag.RERANKER_MODEL + ')' if rag.RERANK_ENABLED else 'OFF'}")
    print(f"Top-N      : {rag.RERANK_TOP_N} chunks sent to the LLM")

    print("\n[1/3] Loading repository through the GitHub API ...")
    try:
        print("  " + rag.load_repository(github_url))
    except GitHubLoaderError as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"ERROR: could not build the index: {exc}")
        return 1

    print("\n[2/3] Running the RAG pipeline on the test questions ...")
    rows = run_pipeline(rag, samples, args.delay)
    if not rows:
        print("ERROR: the pipeline produced no results to evaluate.")
        return 1

    print("\n[3/3] Scoring with RAGAS ...")
    try:
        frame = run_ragas(rag, rows)
    except Exception as exc:
        print(f"ERROR: RAGAS evaluation failed: {exc}")
        return 1

    scored = show_and_save(frame, rows, args.output)
    return 0 if scored else 1


if __name__ == "__main__":
    sys.exit(main())