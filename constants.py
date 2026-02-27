from __future__ import annotations

COMBINE_PAIRS = [
    ("*Failure", "Failure", "Error Code"),
    (
        "Comp. suction pipe temp. (°C)",
        "Accumulator inlet temp. (°C)",
        "*Compressor Suction Temp (°C)",
    ),
    (
        "Comp.1 INV stepping down cntl",
        "Comp.1 OC stepping down cntl",
        "Comp1. stepping down cntl",
    ),
    (
        "Comp.2 INV stepping down cntl",
        "Comp.2 OC stepping down cntl",
        "Comp2. stepping down cntl",
    ),
    ("EVM (Main) (pls)", "EV opening 1 (pls)", "EV Opening 1"),
    ("EVT (subcooling heat xchanger) (pls)", "EV opening 2 (pls)", "EV Opening 2"),
    (
        "INV comp. body temp (°C)",
        "Compressor surface temp. (°C)",
        "Compressor Surface Temp. (°C)",
    ),
]
