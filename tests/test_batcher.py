import threading

import pytest

from avalanche_api.batcher import Batcher


def run_concurrently(batcher: Batcher, texts: list[str]) -> list[str]:
    """Send each text from its own thread, as FastAPI's thread pool does."""
    answers = [""] * len(texts)

    def send(i: int) -> None:
        answers[i] = batcher.generate(texts[i])

    threads = [threading.Thread(target=send, args=(i,)) for i in range(len(texts))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return answers


def test_concurrent_requests_share_a_batch_and_get_their_own_answers():
    batches = []

    def generate_batch(texts):
        batches.append(texts)
        return [t.upper() for t in texts]

    texts = ["a", "b", "c", "d", "e"]
    answers = run_concurrently(Batcher(generate_batch, max_size=4, max_wait=0.5), texts)

    assert answers == ["A", "B", "C", "D", "E"]
    assert sorted(len(b) for b in batches) == [1, 4]


def test_max_size_1_runs_one_request_at_a_time():
    batches = []

    def generate_batch(texts):
        batches.append(texts)
        return texts

    run_concurrently(Batcher(generate_batch, max_size=1), ["a", "b", "c"])

    assert [len(b) for b in batches] == [1, 1, 1]


def test_a_failed_batch_fails_its_requests_and_the_next_batch_still_runs():
    def generate_batch(texts):
        if texts == ["bad"]:
            raise RuntimeError("model crashed")
        return texts

    batcher = Batcher(generate_batch, max_size=1)

    with pytest.raises(RuntimeError):
        batcher.generate("bad")
    assert batcher.generate("good") == "good"
