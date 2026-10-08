"""Device batch lifetime and stream ordering."""
from dataclasses import dataclass
import torch


@dataclass
class ReplayBatch:
    data: dict[str, torch.Tensor]
    _event: object = None
    _source: object = None

    def wait(self):
        """Make the learner stream wait for transfer without blocking the host."""
        if self._event is not None:
            stream = torch.cuda.current_stream(next(iter(self.data.values())).device)
            stream.wait_event(self._event)
            for value in self.data.values():
                value.record_stream(stream)
        return self.data
