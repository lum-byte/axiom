"""Manual import probe for latent parser/model modules.

This is intentionally not named test_*.py because it is a network/runtime probe,
not part of the deterministic pytest suite.
"""

from __future__ import annotations

import importlib


def main() -> None:
    for module in (
        "tag.world_model.world_latent_parser.latent_parser",
        "tag.world_model.world_latent_model.latent_model",
    ):
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - manual probe
            print(f"{module}: FAIL - {exc}")
        else:
            print(f"{module}: ok")


if __name__ == "__main__":
    main()
