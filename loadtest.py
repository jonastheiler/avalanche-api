"""Send 16 real bulletins to a running API at once and measure latency and throughput.

uv run python loadtest.py --concurrency 8

bulletins.json: 4 bulletins per language from winter 25/26, by the WSL Institute for Snow
and Avalanche Research SLF (www.slf.ch), licensed CC BY 4.0.
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import httpx


def send(client: httpx.Client, text: str) -> tuple[float, int]:
    start = time.perf_counter()
    response = client.post("/extract", json={"text": text})
    return time.perf_counter() - start, response.status_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=8, help="requests in flight at once")
    args = parser.parse_args()

    with open("bulletins.json", encoding="utf-8") as f:
        texts = [bulletin["text"] for bulletin in json.load(f)]
    with httpx.Client(base_url=args.url, timeout=None) as client:
        with ThreadPoolExecutor(args.concurrency) as pool:
            start = time.perf_counter()
            results = list(pool.map(lambda text: send(client, text), texts))
            total = time.perf_counter() - start

    seconds = [s for s, _ in results]
    errors = sum(status != 200 for _, status in results)
    p95 = statistics.quantiles(seconds, n=20, method="inclusive")[18]
    print(
        f"{len(texts)} requests, concurrency {args.concurrency}: "
        f"p50 {statistics.median(seconds):.1f} s, p95 {p95:.1f} s, "
        f"{len(texts) / total * 60:.1f} requests/min, {errors} errors"
    )


if __name__ == "__main__":
    main()
