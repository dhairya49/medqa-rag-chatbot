"""
test_concurrency.py

Fires N concurrent requests to the /chat endpoint and reports:
  - How many succeeded
  - How many failed
  - Min / Max / Avg response time
  - Whether requests truly ran in parallel (overlap in time)

Usage:
    python test_concurrency.py
    python test_concurrency.py --n 15 --url http://localhost:8000
"""

import argparse
import asyncio
import time
import httpx

QUESTIONS = [
    "What is diabetes?",
    "What causes high blood pressure?",
    "What are the symptoms of asthma?",
    "How is cancer treated?",
    "What is a heart attack?",
    "What causes kidney failure?",
    "What is the flu?",
    "What are symptoms of depression?",
    "How does the immune system work?",
    "What is a stroke?",
    "What causes liver disease?",
    "What is arthritis?",
    "What are symptoms of anxiety?",
    "What is pneumonia?",
    "What causes anemia?",
]


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    session_id: str,
    message: str,
    index: int,
) -> dict:
    start = time.perf_counter()
    try:
        response = await client.post(
            f"{url}/api/v1/chat",
            data={"session_id": session_id, "message": message},
            timeout=120.0,
        )
        elapsed = time.perf_counter() - start
        success = response.status_code == 200
        return {
            "index": index,
            "question": message,
            "success": success,
            "status": response.status_code,
            "elapsed": elapsed,
            "tool_used": response.json().get("tool_used") if success else None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "index": index,
            "question": message,
            "success": False,
            "status": None,
            "elapsed": elapsed,
            "error": str(exc),
        }


async def run_test(n: int, url: str) -> None:
    questions = (QUESTIONS * 3)[:n]  # repeat if n > 15

    print(f"\n{'='*60}")
    print(f"  Concurrency Test — {n} simultaneous requests")
    print(f"  Target: {url}")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient() as client:
        wall_start = time.perf_counter()

        # Fire all requests at the same time
        tasks = [
            send_request(client, url, f"session_{i}", questions[i], i)
            for i in range(n)
        ]
        results = await asyncio.gather(*tasks)

        wall_elapsed = time.perf_counter() - wall_start

    # ── Report ────────────────────────────────────────────────────────────────
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    times = [r["elapsed"] for r in results]

    print(f"{'─'*60}")
    print(f"  Results:")
    print(f"    Total requests : {n}")
    print(f"    Succeeded      : {len(successes)}")
    print(f"    Failed         : {len(failures)}")
    print(f"{'─'*60}")
    print(f"  Timing:")
    print(f"    Wall clock     : {wall_elapsed:.1f}s  (total real time)")
    print(f"    Fastest        : {min(times):.1f}s")
    print(f"    Slowest        : {max(times):.1f}s")
    print(f"    Average        : {sum(times)/len(times):.1f}s")
    print(f"    Sum of times   : {sum(times):.1f}s")
    print(f"{'─'*60}")

    # If wall clock << sum of times, requests ran in parallel
    parallelism = sum(times) / wall_elapsed
    print(f"  Parallelism factor: {parallelism:.1f}x")
    print(f"  (1.0 = sequential, higher = more concurrent)\n")

    if failures:
        print(f"  Failed requests:")
        for r in failures:
            print(f"    [{r['index']}] {r['question'][:40]} → {r.get('error', r.get('status'))}")
        print()

    print(f"  Per-request breakdown:")
    for r in sorted(results, key=lambda x: x["index"]):
        status = "✓" if r["success"] else "✗"
        tool = f" [{r['tool_used']}]" if r.get("tool_used") else ""
        print(f"    {status} [{r['index']:2d}] {r['elapsed']:5.1f}s  {r['question'][:45]}{tool}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    asyncio.run(run_test(args.n, args.url))