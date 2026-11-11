"""Class definitions and 12 -> merged class remapping for YOLO-format labels."""

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "classes.yaml"


@dataclass(frozen=True)
class ClassConfig:
    original: list[str]
    merged: list[str]
    mapping: dict[str, str]
    pcu: dict[str, float]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> "ClassConfig":
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        conf = cls(
            original=cfg["original"],
            merged=cfg["merged"]["names"],
            mapping=cfg["merged"]["mapping"],
            pcu=cfg.get("pcu", {}),
        )
        conf.validate()
        return conf

    def validate(self) -> None:
        missing = [name for name in self.original if name not in self.mapping]
        if missing:
            raise ValueError(f"Classes without a merged mapping: {missing}")
        unknown = sorted(set(self.mapping.values()) - set(self.merged))
        if unknown:
            raise ValueError(f"Mapping targets not in merged names: {unknown}")

    def id_map(self) -> dict[int, int]:
        """Original class id -> merged class id."""
        return {i: self.merged.index(self.mapping[name]) for i, name in enumerate(self.original)}


def remap_label_lines(lines: list[str], id_map: dict[int, int]) -> list[str]:
    """Rewrite the class id of each YOLO label line ("cls cx cy w h"); blank lines dropped."""
    out = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        cls_id = int(float(parts[0]))
        if cls_id not in id_map:
            raise ValueError(f"Unknown class id {cls_id} in line: {line!r}")
        out.append(" ".join([str(id_map[cls_id]), *parts[1:]]))
    return out
