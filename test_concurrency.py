"""
test_concurrency.py

Runs N concurrent requests against /chat and reports:
  - Concurrency stats (success/fail, latency, parallelism)
  - Keyword-grounded Precision / Recall / F1
    (required keywords + forbidden keyword penalties)
  - ROUGE-1 / ROUGE-2 / ROUGE-L vs compact reference answer
  - Semantic similarity via all-MiniLM-L6-v2

Keyword PRF is designed for practical RAG QA evaluation:
  - True positives: required keywords present in answer
  - False negatives: required keywords missing
  - False positives: forbidden keywords present

Usage:
  python test_concurrency.py
  python test_concurrency.py --n 15 --url http://localhost:8000
"""

import argparse
import asyncio
import re
import time

import httpx
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


THRESHOLDS = {
    "precision": {"min": 0.45, "great": 0.70},
    "recall": {"min": 0.45, "great": 0.70},
    "f1": {"min": 0.45, "great": 0.70},
    "rouge1": {"min": 0.35, "great": 0.55},
    "rouge2": {"min": 0.12, "great": 0.28},
    "rougeL": {"min": 0.28, "great": 0.48},
    "semantic_sim": {"min": 0.60, "great": 0.80},
}


TEST_SET = [
    {
        "question": "What is diabetes?",
        "reference": (
            "Diabetes is a condition with high blood glucose due to insulin "
            "problems, and it can lead to serious complications."
        ),
        "required_keywords": [
            "high blood glucose",
            "insulin",
            "blood sugar",
            "complications",
        ],
        "forbidden_keywords": [
            "lung infection",
            "stroke is always caused by trauma",
            "cancer treatment",
        ],
    },
    {
        "question": "What are the symptoms of asthma?",
        "reference": (
            "Asthma commonly includes coughing, wheezing, shortness of breath, "
            "and chest tightness."
        ),
        "required_keywords": [
            "cough",
            "wheezing",
            "shortness of breath",
            "chest tightness",
        ],
        "forbidden_keywords": [
            "rash",
            "jaundice",
            "kidney stone",
        ],
    },
    {
        "question": "What causes high blood pressure?",
        "reference": (
            "High blood pressure has multiple risk factors such as obesity, high salt "
            "intake, stress, age, and chronic disease."
        ),
        "required_keywords": [
            "salt",
            "stress",
            "obese",
            "genetics",
        ],
        "forbidden_keywords": [
            "virus is the only cause",
            "single guaranteed cause",
        ],
    },
    {
        "question": "What is a heart attack?",
        "reference": (
            "A heart attack happens when blood flow to heart muscle is blocked, "
            "causing tissue damage; chest discomfort and shortness of breath are common signs."
        ),
        "required_keywords": [
            "blood flow",
            "blocked",
            "heart muscle",
            "chest pain",
        ],
        "forbidden_keywords": [
            "brain tumor",
            "skin allergy",
        ],
    },
    {
        "question": "What is Alzheimer's disease?",
        "reference": (
            "Alzheimer's disease is a progressive brain disorder causing memory and thinking decline."
        ),
        "required_keywords": [
            "progressive",
            "memory",
            "brain disorder",
            "dementia",
        ],
        "forbidden_keywords": [
            "curable overnight",
            "bacterial pneumonia",
        ],
    },
    {
        "question": "What are the symptoms of depression?",
        "reference": (
            "Depression symptoms include persistent low mood, loss of interest, sleep or appetite changes, "
            "low energy, and possible suicidal thoughts."
        ),
        "required_keywords": [
            "sad mood",
            "loss of interest",
            "sleep",
            "suicide",
        ],
        "forbidden_keywords": [
            "always fever",
            "fracture pain only",
        ],
    },
    {
        "question": "What is a stroke?",
        "reference": (
            "A stroke is interrupted blood supply to the brain and is a medical emergency."
        ),
        "required_keywords": [
            "brain",
            "blood supply",
            "emergency",
            "weakness",
        ],
        "forbidden_keywords": [
            "joint inflammation",
            "liver failure",
        ],
    },
    {
        "question": "What causes kidney failure?",
        "reference": (
            "Kidney failure is often caused by diabetes and high blood pressure, with additional causes "
            "including kidney disease and prolonged obstruction."
        ),
        "required_keywords": [
            "diabetes",
            "high blood pressure",
            "kidney disease",
            "obstruction",
        ],
        "forbidden_keywords": [
            "sunlight exposure",
            "hair loss",
        ],
    },
    {
        "question": "What is arthritis?",
        "reference": (
            "Arthritis is inflammation of joints causing pain and stiffness; common forms include osteoarthritis and rheumatoid arthritis."
        ),
        "required_keywords": [
            "inflammation",
            "joints",
            "pain",
            "stiffness",
        ],
        "forbidden_keywords": [
            "lung infection only",
            "virus in all cases",
        ],
    },
    {
        "question": "How does insulin work in the body?",
        "reference": (
            "Insulin helps glucose move into cells and regulates blood sugar levels."
        ),
        "required_keywords": [
            "pancreas",
            "glucose",
            "cells",
            "blood sugar",
        ],
        "forbidden_keywords": [
            "insulin increases blood sugar permanently",
            "insulin is made by lungs",
        ],
    },
    {
        "question": "What is multiple sclerosis?",
        "reference": (
            "Multiple sclerosis is an immune-mediated disease affecting the brain and spinal cord by damaging protective nerve covering."
        ),
        "required_keywords": [
            "immune system",
            "brain",
            "spinal cord",
            "nerve fibers",
        ],
        "forbidden_keywords": [
            "caused by kidney stones",
            "only affects skin",
        ],
    },
    {
        "question": "What causes migraines?",
        "reference": (
            "Migraine cause is not fully known; triggers may include hormonal, emotional, dietary, and environmental factors."
        ),
        "required_keywords": [
            "unknown cause",
            "triggers",
            "hormonal",
            "stress",
        ],
        "forbidden_keywords": [
            "always caused by infection",
            "bone fracture trigger only",
        ],
    },
    {
        "question": "What is Parkinson's disease?",
        "reference": (
            "Parkinson's disease is a progressive nervous system disorder affecting movement with tremor, stiffness, and slowness."
        ),
        "required_keywords": [
            "progressive",
            "movement",
            "tremor",
            "stiffness",
        ],
        "forbidden_keywords": [
            "acute bacterial disease",
            "lung-only disorder",
        ],
    },
    {
        "question": "What is pneumonia?",
        "reference": (
            "Pneumonia is infection and inflammation of air sacs in the lungs causing cough, fever, and breathing difficulty."
        ),
        "required_keywords": [
            "infection",
            "lungs",
            "cough",
            "difficulty breathing",
        ],
        "forbidden_keywords": [
            "joint cartilage breakdown",
            "always viral and harmless",
        ],
    },
    {
        "question": "What is chronic obstructive pulmonary disease?",
        "reference": (
            "COPD is chronic inflammatory lung disease with obstructed airflow, often linked to smoking."
        ),
        "required_keywords": [
            "chronic",
            "lung disease",
            "airflow",
            "smoking",
        ],
        "forbidden_keywords": [
            "brain hemorrhage",
            "purely kidney condition",
        ],
    },
]


print("  Loading embedding model for semantic similarity...")
_embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
print("  Model loaded.\n")


def _clean(text: str) -> str:
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _contains_keyword(text: str, keyword: str) -> bool:
    text_norm = " " + _clean(text).lower() + " "
    key_norm = " " + _clean(keyword).lower() + " "
    return key_norm in text_norm


def _keyword_prf(answer: str, required_keywords: list[str], forbidden_keywords: list[str]) -> dict:
    required_hits = sum(1 for kw in required_keywords if _contains_keyword(answer, kw))
    required_total = len(required_keywords)
    forbidden_hits = sum(1 for kw in forbidden_keywords if _contains_keyword(answer, kw))

    tp = required_hits
    fn = max(required_total - required_hits, 0)
    fp = forbidden_hits

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "required_hits": required_hits,
        "required_total": required_total,
        "forbidden_hits": forbidden_hits,
    }


def _semantic_sim(pred: str, ref: str) -> float:
    embeddings = _embed_model.encode([pred, ref], convert_to_tensor=True)
    return round(float(util.cos_sim(embeddings[0], embeddings[1]).item()), 3)


def compute_metrics(answer: str, testcase: dict) -> dict:
    answer_clean = _clean(answer)
    ref_clean = _clean(testcase["reference"])
    keyword_scores = _keyword_prf(
        answer_clean,
        testcase["required_keywords"],
        testcase["forbidden_keywords"],
    )
    rouge = _rouge.score(ref_clean, answer_clean)
    sem = _semantic_sim(answer, testcase["reference"])

    return {
        "precision": keyword_scores["precision"],
        "recall": keyword_scores["recall"],
        "f1": keyword_scores["f1"],
        "rouge1": round(rouge["rouge1"].fmeasure, 3),
        "rouge2": round(rouge["rouge2"].fmeasure, 3),
        "rougeL": round(rouge["rougeL"].fmeasure, 3),
        "semantic_sim": sem,
        "required_hits": keyword_scores["required_hits"],
        "required_total": keyword_scores["required_total"],
        "forbidden_hits": keyword_scores["forbidden_hits"],
    }


def _color(val: float, metric: str) -> str:
    t = THRESHOLDS[metric]
    color = GREEN if val >= t["great"] else YELLOW if val >= t["min"] else RED
    return f"{color}{val:.3f}{RESET}"


def _status(val: float, metric: str) -> str:
    t = THRESHOLDS[metric]
    if val >= t["great"]:
        return f"{GREEN} GREAT{RESET}"
    if val >= t["min"]:
        return f"{YELLOW}  PASS{RESET}"
    return f"{RED}  FAIL{RESET}"


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    session_id: str,
    testcase: dict,
    index: int,
) -> dict:
    start = time.perf_counter()
    try:
        response = await client.post(
            f"{url}/api/v1/chat",
            data={"session_id": session_id, "message": testcase["question"], "mode": "structured", "top_k": 12},
            timeout=120.0,
        )
        elapsed = time.perf_counter() - start
        success = response.status_code == 200
        payload = response.json() if success else {}
        answer = payload.get("answer", "")
        metrics = compute_metrics(answer, testcase) if success and answer else None
        return {
            "index": index,
            "question": testcase["question"],
            "answer": answer,
            "success": success,
            "status": response.status_code,
            "elapsed": elapsed,
            "metrics": metrics,
            "tool_used": payload.get("tool_used"),
        }
    except Exception as exc:
        return {
            "index": index,
            "question": testcase["question"],
            "answer": "",
            "success": False,
            "status": None,
            "elapsed": time.perf_counter() - start,
            "metrics": None,
            "error": str(exc),
        }


METRIC_ROWS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("rouge1", "ROUGE-1"),
    ("rouge2", "ROUGE-2"),
    ("rougeL", "ROUGE-L"),
    ("semantic_sim", "Semantic"),
]


def print_query_result(result: dict) -> None:
    print(f"\n  {BOLD}[{result['index']:02d}] {result['question']}{RESET}")
    print(f"       Latency : {result['elapsed']:.2f}s")
    if not result["success"]:
        err_msg = result.get("error") or f"HTTP {result.get('status')}"
        print(f"       {RED}FAILED - {err_msg}{RESET}")
        return

    metrics = result.get("metrics")
    if not metrics:
        print(f"       {RED}FAILED - No answer returned{RESET}")
        return

    preview = result["answer"][:120].replace("\n", " ")
    print(f"       Answer  : {CYAN}{preview}{'...' if len(result['answer']) > 120 else ''}{RESET}")
    print(
        f"       Keyword coverage: {metrics['required_hits']}/{metrics['required_total']} required hit | "
        f"{metrics['forbidden_hits']} forbidden hit"
    )
    print("       ------------------------------------------------------")
    print("       Metric         Score   min / great          Status")
    print("       ------------------------------------------------------")
    for key, label in METRIC_ROWS:
        t = THRESHOLDS[key]
        print(
            f"       {label:<12} {_color(metrics[key], key):>7}   "
            f"{t['min']} / {t['great']:<18} {_status(metrics[key], key)}"
        )
    print("       ------------------------------------------------------")


def print_averages(results: list[dict]) -> None:
    scored = [r for r in results if r["success"] and r["metrics"]]
    if not scored:
        print(f"\n  {RED}No successful queries to average.{RESET}")
        return

    keys = [k for k, _ in METRIC_ROWS]
    averages = {k: sum(r["metrics"][k] for r in scored) / len(scored) for k in keys}
    avg_required_hits = sum(r["metrics"]["required_hits"] for r in scored) / len(scored)
    avg_required_total = sum(r["metrics"]["required_total"] for r in scored) / len(scored)
    avg_forbidden_hits = sum(r["metrics"]["forbidden_hits"] for r in scored) / len(scored)

    print("\n==============================================================")
    print(f"  {BOLD}AVERAGE SCORES ({len(scored)} queries){RESET}")
    print("==============================================================")
    print("  Metric         Avg     min / great          Status")
    print("  ------------------------------------------------------------")
    for key, label in METRIC_ROWS:
        t = THRESHOLDS[key]
        print(
            f"  {label:<12} {_color(averages[key], key):>7}   "
            f"{t['min']} / {t['great']:<18} {_status(averages[key], key)}"
        )
    print("  ------------------------------------------------------------")
    print(
        f"  Avg keyword coverage : {avg_required_hits:.2f}/{avg_required_total:.2f} required hit"
    )
    print(f"  Avg forbidden hits   : {avg_forbidden_hits:.2f}")

    passing = sum(1 for key in keys if averages[key] >= THRESHOLDS[key]["min"])
    total = len(keys)
    pct = int(passing / total * 100)
    if passing == total:
        verdict = f"{GREEN}ALL {total}/{total} METRICS PASSING{RESET}"
    elif passing >= int(total * 0.6):
        verdict = f"{YELLOW}{passing}/{total} METRICS PASSING ({pct}%) - fair{RESET}"
    else:
        verdict = f"{RED}{passing}/{total} METRICS PASSING ({pct}%) - needs work{RESET}"
    print(f"\n  Overall  : {verdict}")
    print("==============================================================\n")


async def run_test(n: int, url: str) -> None:
    test_items = (TEST_SET * ((n // len(TEST_SET)) + 1))[:n]
    print("\n==============================================================")
    print(f"  {BOLD}Concurrency + Keyword Quality Test - {n} requests{RESET}")
    print(f"  Target : {url}")
    print("  PRF    : Required keyword hits + forbidden keyword penalties")
    print("  Mode   : structured, top_k=12")
    print("==============================================================")

    async with httpx.AsyncClient() as client:
        wall_start = time.perf_counter()
        tasks = [
            send_request(client, url, f"session_{i}", test_items[i], i)
            for i in range(n)
        ]
        results = await asyncio.gather(*tasks)
        wall_elapsed = time.perf_counter() - wall_start

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    times = [r["elapsed"] for r in results]
    parallelism = sum(times) / wall_elapsed if wall_elapsed > 0 else 0.0

    print("\n--------------------------------------------------------------")
    print(f"  {BOLD}Concurrency Summary{RESET}")
    print("--------------------------------------------------------------")
    print(f"  Total requests  : {n}")
    print(f"  Succeeded       : {GREEN}{len(successes)}{RESET}")
    print(f"  Failed          : {(RED if failures else GREEN)}{len(failures)}{RESET}")
    print(f"  Wall clock      : {wall_elapsed:.2f}s")
    print(f"  Fastest         : {min(times):.2f}s")
    print(f"  Slowest         : {max(times):.2f}s")
    print(f"  Avg latency     : {sum(times)/len(times):.2f}s")
    print(f"  Parallelism     : {parallelism:.2f}x")

    if failures:
        print(f"\n  {RED}Failed requests:{RESET}")
        for result in failures:
            print(
                f"    [{result['index']:02d}] "
                f"{result['question'][:45]} -> {result.get('error', result.get('status'))}"
            )

    print("\n--------------------------------------------------------------")
    print(f"  {BOLD}Per-Query Quality Metrics{RESET}")
    print("--------------------------------------------------------------")
    print(f"  {GREEN}Green=GREAT{RESET}  {YELLOW}Yellow=PASS{RESET}  {RED}Red=FAIL{RESET}")

    for result in sorted(results, key=lambda item: item["index"]):
        print_query_result(result)

    print_averages(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrency + keyword-grounded quality test")
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help=f"Number of concurrent requests (max {len(TEST_SET)} for unique cases)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL",
    )
    args = parser.parse_args()
    asyncio.run(run_test(args.n, args.url))
