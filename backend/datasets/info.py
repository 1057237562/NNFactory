from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class DatasetInfo:
    id: str
    name: str
    dataset_type: str
    num_samples: int
    num_classes: int = 0
    input_shape: list[int] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    file_path: str = ""
    file_size: int = 0
    created_at: str = ""
    status: str = "ready"
    split_info: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
