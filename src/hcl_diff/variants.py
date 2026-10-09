from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSpec:
    name: str
    slices: int
    joint_generation: bool
    cross_slice_attention: bool
    relative_slice_encoding: bool
    slice_difference_loss: bool


_VARIANTS = {
    "C0": VariantSpec(
        name="C0",
        slices=1,
        joint_generation=False,
        cross_slice_attention=False,
        relative_slice_encoding=False,
        slice_difference_loss=False,
    ),
    "C1": VariantSpec(
        name="C1",
        slices=3,
        joint_generation=True,
        cross_slice_attention=False,
        relative_slice_encoding=False,
        slice_difference_loss=False,
    ),
    "C2": VariantSpec(
        name="C2",
        slices=3,
        joint_generation=True,
        cross_slice_attention=True,
        relative_slice_encoding=True,
        slice_difference_loss=True,
    ),
}


def get_variant(name: str) -> VariantSpec:
    key = str(name).upper()
    if key not in _VARIANTS:
        raise KeyError(f"Unknown generator variant: {name}")
    return _VARIANTS[key]


def variant_names() -> tuple[str, ...]:
    return tuple(_VARIANTS)

