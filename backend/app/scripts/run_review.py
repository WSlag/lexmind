"""CLI runner to exercise the review pipeline offline.

Usage:
    python -m app.scripts.run_review ../examples/contracts/sample_gas_supply_agreement.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.workflow.services import ReviewService  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m app.scripts.run_review <file>")
        sys.exit(2)
    path = Path(sys.argv[1])
    data = path.read_bytes()
    review = ReviewService().run(path.name, data)
    out = json.dumps(review.to_dict(), indent=2, ensure_ascii=False)
    out_dir = Path(__file__).resolve().parents[3] / "examples" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{path.stem}.review.json"
    target.write_text(out, encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()