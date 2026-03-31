from dataclasses import dataclass

# signal the inference runner to stop
@dataclass
class StopFlag:
    stop: bool = False