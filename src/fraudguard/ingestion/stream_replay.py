"""Historical transaction replay engine."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class ReplayEngine:
    """Release bounded transaction windows through an injected prediction action."""
    batch_size: int
    delay_seconds: float

    def replay(self, total_transactions: int, release_batch: Callable[[int, int], int]) -> int:
        processed_events = 0
        for transaction_cursor in range(0, total_transactions, self.batch_size):
            processed_events += release_batch(transaction_cursor, min(self.batch_size, total_transactions - transaction_cursor))
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)
        return processed_events

