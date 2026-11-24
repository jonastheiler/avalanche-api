"""Dynamic batching: requests that arrive close together go through the model as one batch.

Writing the answer is limited by reading the model's weights from memory, once per token
whatever the batch size: on a laptop CPU, a step for 8 bulletins takes 3 times as long as
for one, not 8 times. Reading the prompt gains nothing (see the load test in the README).
"""

import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future


class Batcher:
    def __init__(
        self,
        generate_batch: Callable[[list[str]], list[str]],
        max_size: int = 8,
        max_wait: float = 0.05,
    ) -> None:
        self.generate_batch = generate_batch
        self.max_size = max_size  # 1 switches batching off: one request at a time.
        self.max_wait = max_wait  # Seconds the first request waits for others to join.
        self.queue: queue.Queue[tuple[str, Future]] = queue.Queue()
        # One worker thread runs the model, so generations never compete for the cores.
        threading.Thread(target=self.run, daemon=True).start()

    def generate(self, text: str) -> str:
        """Called by each request's thread: blocks until the batch with this text is done."""
        future: Future[str] = Future()
        self.queue.put((text, future))
        return future.result()

    def next_batch(self) -> list[tuple[str, Future]]:
        batch = [self.queue.get()]  # Wait as long as needed for the first request.
        deadline = time.monotonic() + self.max_wait
        while len(batch) < self.max_size:
            try:
                batch.append(self.queue.get(timeout=max(0, deadline - time.monotonic())))
            except queue.Empty:
                break
        return batch

    def run(self) -> None:
        while True:
            batch = self.next_batch()
            try:
                answers = self.generate_batch([text for text, _ in batch])
            except Exception as error:  # Fail these requests, keep serving the next ones.
                for _, future in batch:
                    future.set_exception(error)
                continue
            for (_, future), answer in zip(batch, answers, strict=True):
                future.set_result(answer)
