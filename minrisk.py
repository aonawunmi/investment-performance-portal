from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Literal, Dict, Iterable
import csv
import uuid

Status = Literal["Open", "In Progress", "Closed"]
Target = Literal["Likelihood", "Impact"]


@dataclass
class AppConfig:
    """Application-level configuration.

    ``matrix_size`` determines the range of valid likelihood/impact scores and
    also affects the bucket thresholds. ``likelihood_labels`` and
    ``impact_labels`` provide human readable descriptions for each level.
    ``divisions``, ``departments`` and ``categories`` seed the lookup lists.
    """

    matrix_size: int = 5
    likelihood_labels: List[str] = field(
        default_factory=lambda: [
            "Rare (≈10+y)",
            "Unlikely (5–10y)",
            "Possible (2–5y)",
            "Likely (annual)",
            "Almost certain (multi/yr)",
        ]
    )
    impact_labels: List[str] = field(
        default_factory=lambda: [
            "Minimal",
            "Low",
            "Moderate",
            "High",
            "Severe",
        ]
    )
    divisions: List[str] = field(
        default_factory=lambda: ["Clearing", "Operations", "Finance"]
    )
    departments: List[str] = field(
        default_factory=lambda: [
            "Risk Management",
            "IT Ops",
            "Quant/Risk",
            "Treasury",
            "Trading",
        ]
    )
    categories: List[str] = field(
        default_factory=lambda: [
            "Strategic",
            "Credit",
            "Market",
            "Liquidity",
            "Operational",
            "Legal/Compliance",
            "Technology",
            "ESG",
            "Reputational",
        ]
    )


@dataclass
class Control:
    """Represents a single risk control."""
    description: str
    target: Target
    design: int
    implementation: int
    monitoring: int
    effectiveness_evaluation: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class RiskRow:
    """Single risk entry in the register."""
    risk_code: str
    risk_title: str
    risk_description: str
    division: str
    department: str
    category: str
    owner: str
    likelihood_inherent: int
    impact_inherent: int
    controls: List[Control] = field(default_factory=list)
    status: Status = "Open"


@dataclass
class ProcessedRisk(RiskRow):
    """Risk row augmented with calculated scores."""

    likelihood_residual: float = 0.0
    impact_residual: float = 0.0
    inherent_score: float = 0.0
    residual_score: float = 0.0


# ==== helpers ==============================================================

def bucket(l: float, i: float, size: int = 5) -> str:
    """Return a qualitative bucket for a likelihood/impact combination.

    The thresholds match those used in the React application. ``size`` should
    be either 5 or 6; when ``6`` an extra "Catastrophic" bucket is returned for
    very high scores.
    """

    score = l * i
    if size == 6:
        if score >= 30:
            return "Catastrophic"
        if score >= 20:
            return "Severe"
        if score >= 12:
            return "High"
        if score >= 6:
            return "Moderate"
        return "Low"

    # default 5x5 thresholds
    if score >= 20:
        return "Severe"
    if score >= 12:
        return "High"
    if score >= 6:
        return "Moderate"
    if score >= 3:
        return "Low"
    return "Minimal"


def cap3(s: str) -> str:
    return ''.join(ch for ch in s if ch.isalpha())[:3].upper() or "NEW"


def next_risk_code(rows: List[RiskRow], division: str, category: str) -> str:
    prefix = f"{cap3(division)}-{cap3(category)}"
    nums = [int(r.risk_code.split('-')[2]) for r in rows if r.risk_code.startswith(prefix)]
    nxt = max(nums, default=0) + 1
    return f"{prefix}-{nxt:03d}"


def calculate_control_effectiveness(control: Control) -> float:
    if control.design == 0 or control.implementation == 0:
        return 0.0
    return (control.design + control.implementation + control.monitoring + control.effectiveness_evaluation) / 12.0


def calculate_residual_risk(risk: RiskRow) -> Dict[str, float]:
    likelihood_controls = [c for c in risk.controls if c.target == "Likelihood"]
    impact_controls = [c for c in risk.controls if c.target == "Impact"]
    max_l = max((calculate_control_effectiveness(c) for c in likelihood_controls), default=0.0)
    max_i = max((calculate_control_effectiveness(c) for c in impact_controls), default=0.0)
    residual_l = risk.likelihood_inherent - (risk.likelihood_inherent - 1) * max_l
    residual_i = risk.impact_inherent - (risk.impact_inherent - 1) * max_i
    return {
        "likelihood": max(1.0, residual_l),
        "impact": max(1.0, residual_i),
    }


def process_risks(risks: Iterable[RiskRow]) -> List[ProcessedRisk]:
    """Augment each risk with residual values and scores."""

    processed: List[ProcessedRisk] = []
    for r in risks:
        residual = calculate_residual_risk(r)
        data = asdict(r)
        processed.append(
            ProcessedRisk(
                **data,
                likelihood_residual=residual["likelihood"],
                impact_residual=residual["impact"],
                inherent_score=r.likelihood_inherent * r.impact_inherent,
                residual_score=residual["likelihood"] * residual["impact"],
            )
        )
    return processed


def severity_ranking(
    processed: Iterable[ProcessedRisk],
    key: Literal["residual_score", "inherent_score"] = "residual_score",
) -> List[ProcessedRisk]:
    """Return risks sorted by severity (descending)."""

    return sorted(processed, key=lambda r: getattr(r, key), reverse=True)


def export_priority_risks_csv(
    path: str,
    processed: Iterable[ProcessedRisk],
    config: AppConfig,
    codes: Iterable[str],
) -> None:
    """Write selected risks to ``path`` for sharing or further analysis."""

    code_set = set(codes)
    selected = [r for r in processed if r.risk_code in code_set]
    if not selected:
        return
    fieldnames = [
        "S/N",
        "Risk Code",
        "Title",
        "Category",
        "Owner",
        "Inherent Score",
        "Residual Score",
        "Residual Bucket",
        "Status",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, r in enumerate(selected, 1):
            writer.writerow(
                {
                    "S/N": idx,
                    "Risk Code": r.risk_code,
                    "Title": r.risk_title,
                    "Category": r.category,
                    "Owner": r.owner,
                    "Inherent Score": f"{r.inherent_score:.2f}",
                    "Residual Score": f"{r.residual_score:.2f}",
                    "Residual Bucket": bucket(
                        r.likelihood_residual, r.impact_residual, config.matrix_size
                    ),
                    "Status": r.status,
                }
            )


# ==== CSV utilities ========================================================

REQUIRED_HEADERS = [
    "risk_title",
    "risk_description",
    "division",
    "department",
    "category",
    "owner",
    "likelihood_inherent",
    "impact_inherent",
    "status",
]


def load_risks_csv(path: str, config: AppConfig | None = None) -> List[RiskRow]:
    """Load risk rows from a CSV file.

    The CSV must contain the headers defined in ``REQUIRED_HEADERS``. Headers are
    trimmed to avoid issues with leading spaces. Validation of numerical fields
    honours the ``matrix_size`` defined in ``config``.
    """

    cfg = config or AppConfig()
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file is missing a header row")
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        missing = [h for h in REQUIRED_HEADERS if h not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        rows: List[RiskRow] = []
        for raw in reader:
            status = (raw.get("status") or "").strip().title()
            if status not in {"Open", "In Progress", "Closed"}:
                raise ValueError(f"Invalid status: {status}")
            try:
                l = int((raw.get("likelihood_inherent") or "").strip())
                i = int((raw.get("impact_inherent") or "").strip())
            except ValueError:
                raise ValueError("Likelihood/Impact must be integers")
            if not (1 <= l <= cfg.matrix_size):
                raise ValueError(
                    f"likelihood_inherent {l} out of range 1-{cfg.matrix_size}"
                )
            if not (1 <= i <= cfg.matrix_size):
                raise ValueError(
                    f"impact_inherent {i} out of range 1-{cfg.matrix_size}"
                )
            risk = RiskRow(
                risk_code="",  # will be assigned later
                risk_title=(raw.get("risk_title") or "").strip(),
                risk_description=(raw.get("risk_description") or "").strip(),
                division=(raw.get("division") or "").strip(),
                department=(raw.get("department") or "").strip(),
                category=(raw.get("category") or "").strip(),
                owner=(raw.get("owner") or "").strip(),
                likelihood_inherent=l,
                impact_inherent=i,
                status=status,
            )
            rows.append(risk)
    return rows


def export_risks_csv(path: str, risks: List[RiskRow]) -> None:
    """Write risk rows to ``path``."""
    fieldnames = [
        "risk_code",
        "risk_title",
        "risk_description",
        "division",
        "department",
        "category",
        "owner",
        "likelihood_inherent",
        "impact_inherent",
        "status",
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in risks:
            writer.writerow({
                "risk_code": r.risk_code,
                "risk_title": r.risk_title,
                "risk_description": r.risk_description,
                "division": r.division,
                "department": r.department,
                "category": r.category,
                "owner": r.owner,
                "likelihood_inherent": r.likelihood_inherent,
                "impact_inherent": r.impact_inherent,
                "status": r.status,
            })


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="MinRisk CSV utilities")
    parser.add_argument("csv", help="Path to risk CSV file to load")
    parser.add_argument(
        "--config",
        help="Optional path to JSON configuration (overrides defaults)",
        default=None,
    )
    parser.add_argument(
        "--export-priority",
        help="Export selected risk codes to this CSV path",
        default=None,
    )
    parser.add_argument(
        "--priority",
        help="Comma separated list of risk codes to export",
        default="",
    )
    args = parser.parse_args()

    cfg = AppConfig(**json.load(open(args.config))) if args.config else AppConfig()
    risks = load_risks_csv(args.csv, cfg)
    for r in risks:
        r.risk_code = next_risk_code(risks, r.division, r.category)

    processed = process_risks(risks)
    ranking = severity_ranking(processed)
    for idx, r in enumerate(ranking, 1):
        print(f"{idx:>3} {r.risk_code} {r.risk_title} (Residual {r.residual_score:.2f})")

    if args.export_priority and args.priority:
        codes = [c.strip() for c in args.priority.split(",") if c.strip()]
        export_priority_risks_csv(args.export_priority, processed, cfg, codes)
        print(f"Exported {len(codes)} risk(s) to {args.export_priority}")
