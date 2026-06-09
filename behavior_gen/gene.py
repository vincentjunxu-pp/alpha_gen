from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping


UNARY_OP_CHOICES = (
    "current",
    "rank_pct",
    "zscore",
    "direction_rank",
    "direction_zscore",
    "ind_rank_pct",
    "ind_zscore",
    "ts_zscore_5d",
    "ts_zscore_20d",
)

COMBINER_CHOICES = (
    "rank_gap",
    "residual_gap",
    "gated_rank_gap",
    "quality_gap",
    "crowding_interaction",
    "confirm",
    "gated_confirm",
    "risk_minus_confirm",
    "panic_reversal",
    "attention_risk",
    "orderbook_intent",
    "liquidity_gap",
    "anchor_confirm",
)

CONDITION_OP_CHOICES = (
    "top_quantile",
    "bottom_quantile",
    "above_median",
    "below_median",
    "positive",
    "negative",
)

DIRECTION_POLICIES = ("fixed", "train_ic", "regime_switch")


@dataclass(frozen=True)
class BehaviorFieldRule:
    """Metadata for one behavior-finance input field."""

    data_family: str
    behavior_roles: tuple[str, ...]
    sub_family: str = "unknown"
    sub_type: str = "unknown"
    unit_type: str = "unknown"
    window: str = "unknown"
    session: str = "full"
    investor_type: str = "none"
    direction: int = 1
    allowed_slots: tuple[str, ...] = ()
    allowed_unary_ops: tuple[str, ...] = ("current", "rank_pct", "zscore")

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "BehaviorFieldRule":
        roles_raw = raw.get("behavior_roles", raw.get("behavior_role", ()))
        if isinstance(roles_raw, str):
            roles = (roles_raw,)
        else:
            roles = tuple(str(role) for role in roles_raw)  # type: ignore[arg-type]

        slots_raw = raw.get("allowed_slots", ())
        if isinstance(slots_raw, str):
            slots = (slots_raw,)
        else:
            slots = tuple(str(slot) for slot in slots_raw)  # type: ignore[arg-type]

        unary_raw = raw.get("allowed_unary_ops", ("current", "rank_pct", "zscore"))
        if isinstance(unary_raw, str):
            unary_ops = (unary_raw,)
        else:
            unary_ops = tuple(str(op) for op in unary_raw)  # type: ignore[arg-type]

        direction = int(raw.get("direction", 1))
        if direction > 0:
            direction = 1
        elif direction < 0:
            direction = -1
        else:
            direction = 0

        return cls(
            data_family=str(raw.get("data_family", raw.get("family", "other"))),
            behavior_roles=roles,
            sub_family=str(raw.get("sub_family", raw.get("family", "unknown"))),
            sub_type=str(raw.get("sub_type", raw.get("sub_family", "unknown"))),
            unit_type=str(raw.get("unit_type", "unknown")),
            window=str(raw.get("window", raw.get("period_type", "unknown"))),
            session=str(raw.get("session", "full")),
            investor_type=str(raw.get("investor_type", "none")),
            direction=direction,
            allowed_slots=slots,
            allowed_unary_ops=unary_ops,
        )


@dataclass(frozen=True)
class SlotSpec:
    """One semantic slot required or allowed by a behavior mode."""

    role: str
    data_families: tuple[str, ...] = ()
    behavior_roles: tuple[str, ...] = ()
    required: bool = True


@dataclass(frozen=True)
class ModeSpec:
    """A behavior-finance mechanism template."""

    name: str
    description: str
    slots: Mapping[str, SlotSpec]
    allowed_combiners: tuple[str, ...]
    default_combiner: str
    direction: int
    direction_policy: str = "fixed"
    allowed_condition_roles: tuple[str, ...] = ()
    max_conditions: int = 1


@dataclass(frozen=True)
class SlotGene:
    """Selected field and unary transform for one mode slot."""

    field: str
    unary_op: str = "rank_pct"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "SlotGene":
        return cls(field=str(raw["field"]), unary_op=str(raw.get("unary_op", "rank_pct")))


@dataclass(frozen=True)
class ConditionGene:
    """Optional gate/state condition applied by a mode."""

    field: str
    unary_op: str = "rank_pct"
    condition_op: str = "top_quantile"
    threshold: float = 0.6

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ConditionGene":
        return cls(
            field=str(raw["field"]),
            unary_op=str(raw.get("unary_op", "rank_pct")),
            condition_op=str(raw.get("condition_op", "top_quantile")),
            threshold=float(raw.get("threshold", 0.6)),
        )


@dataclass(frozen=True)
class BehaviorGene:
    """Strongly typed behavior-finance gene.

    `mode` defines the economic mechanism. Slots select fields that play typed
    roles in that mechanism. `combiner` decides how slot values are composed.
    Conditions and controls are optional, but still validated against metadata.
    """

    mode: str
    combiner: str
    slots: Mapping[str, SlotGene]
    conditions: tuple[ConditionGene, ...] = ()
    direction_policy: str = "fixed"
    version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "combiner": self.combiner,
            "slots": {name: slot.to_dict() for name, slot in self.slots.items()},
            "conditions": [condition.to_dict() for condition in self.conditions],
            "direction_policy": self.direction_policy,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "BehaviorGene":
        slots_raw = raw.get("slots", {})
        if not isinstance(slots_raw, Mapping):
            raise TypeError("BehaviorGene slots must be a mapping")
        conditions_raw = raw.get("conditions", ())
        return cls(
            mode=str(raw["mode"]),
            combiner=str(raw["combiner"]),
            slots={str(name): SlotGene.from_dict(slot) for name, slot in slots_raw.items()},  # type: ignore[arg-type]
            conditions=tuple(ConditionGene.from_dict(condition) for condition in conditions_raw),  # type: ignore[arg-type]
            direction_policy=str(raw.get("direction_policy", "fixed")),
            version=int(raw.get("version", 1)),
        )


def _slot(
    role: str,
    data_families: tuple[str, ...],
    behavior_roles: tuple[str, ...],
    *,
    required: bool = True,
) -> SlotSpec:
    return SlotSpec(role=role, data_families=data_families, behavior_roles=behavior_roles, required=required)


MODE_REGISTRY: dict[str, ModeSpec] = {
    "fund_price_underreaction": ModeSpec(
        name="fund_price_underreaction",
        description="基本面锚强于价格反应，刻画市场对基本面改善的反应不足。",
        slots={
            "fund_anchor": _slot("fund_anchor", ("fundamental",), ("anchor", "growth", "quality", "support")),
            "price_reaction": _slot("price_reaction", ("price_volume",), ("reaction", "momentum")),
        },
        allowed_combiners=("rank_gap", "residual_gap", "gated_rank_gap"),
        default_combiner="rank_gap",
        direction=1,
        allowed_condition_roles=("attention", "crowding", "liquidity", "open_intent"),
    ),
    "quality_neglect": ModeSpec(
        name="quality_neglect",
        description="利润增长与现金流/经营质量背离，刻画市场忽视利润质量。",
        slots={
            "profit_growth": _slot("profit_growth", ("fundamental",), ("growth", "anchor", "support")),
            "cashflow_quality": _slot("cashflow_quality", ("fundamental",), ("quality", "cashflow_quality")),
            "price_reaction": _slot("price_reaction", ("price_volume",), ("reaction", "attention"), required=False),
        },
        allowed_combiners=("quality_gap", "rank_gap", "gated_rank_gap"),
        default_combiner="quality_gap",
        direction=-1,
        allowed_condition_roles=("attention", "crowding"),
    ),
    "growth_crowding_risk": ModeSpec(
        name="growth_crowding_risk",
        description="成长叙事叠加成交拥挤，刻画成长交易过热后的回撤风险。",
        slots={
            "growth_anchor": _slot("growth_anchor", ("fundamental",), ("growth", "anchor")),
            "crowding_signal": _slot("crowding_signal", ("price_volume", "moneyflow", "orderbook"), ("crowding", "attention", "close_chase", "risk")),
            "fund_support": _slot("fund_support", ("fundamental",), ("quality", "valuation", "support"), required=False),
        },
        allowed_combiners=("crowding_interaction", "risk_minus_confirm"),
        default_combiner="crowding_interaction",
        direction=-1,
        allowed_condition_roles=("crowding", "attention", "liquidity"),
    ),
    "fund_flow_confirmation": ModeSpec(
        name="fund_flow_confirmation",
        description="基本面改善得到大额资金或盘口买入意愿确认。",
        slots={
            "fund_anchor": _slot("fund_anchor", ("fundamental",), ("anchor", "growth", "quality", "support")),
            "flow_confirm": _slot("flow_confirm", ("moneyflow",), ("large_flow", "confirmation", "underreaction")),
            "orderbook_filter": _slot("orderbook_filter", ("orderbook",), ("open_intent", "confirmation", "buy_pressure"), required=False),
            "price_control": _slot("price_control", ("price_volume",), ("reaction", "momentum"), required=False),
        },
        allowed_combiners=("confirm", "gated_confirm", "residual_gap"),
        default_combiner="confirm",
        direction=1,
        allowed_condition_roles=("open_intent", "liquidity", "crowding"),
        max_conditions=2,
    ),
    "retail_chase_risk": ModeSpec(
        name="retail_chase_risk",
        description="价格上涨、小单追涨、尾盘买压较强但大资金不确认。",
        slots={
            "price_momentum": _slot("price_momentum", ("price_volume",), ("momentum", "reaction")),
            "retail_flow": _slot("retail_flow", ("moneyflow",), ("retail", "small_flow", "chase")),
            "close_chase": _slot("close_chase", ("orderbook",), ("close_chase", "crowding")),
            "large_flow": _slot("large_flow", ("moneyflow",), ("large_flow", "confirmation"), required=False),
        },
        allowed_combiners=("risk_minus_confirm",),
        default_combiner="risk_minus_confirm",
        direction=-1,
        allowed_condition_roles=("crowding", "attention", "close_chase"),
        max_conditions=2,
    ),
    "panic_reversal": ModeSpec(
        name="panic_reversal",
        description="基本面尚可但短期恐慌、卖压或流动性冲击造成错杀。",
        slots={
            "fund_anchor": _slot("fund_anchor", ("fundamental",), ("anchor", "quality", "support")),
            "drawdown": _slot("drawdown", ("price_volume",), ("panic", "drawdown", "oversold")),
            "sell_pressure": _slot("sell_pressure", ("moneyflow", "orderbook"), ("retail", "small_flow", "close_chase", "stress", "orderbook_pressure")),
            "orderbook_filter": _slot("orderbook_filter", ("orderbook",), ("open_intent", "buy_pressure", "confirmation"), required=False),
        },
        allowed_combiners=("panic_reversal",),
        default_combiner="panic_reversal",
        direction=1,
        allowed_condition_roles=("panic", "liquidity", "open_intent"),
        max_conditions=2,
    ),
    "attention_overreaction": ModeSpec(
        name="attention_overreaction",
        description="极端收益、成交异常或过热关注造成短期过度反应。",
        slots={
            "attention_heat": _slot("attention_heat", ("price_volume",), ("attention", "overreaction", "lottery")),
            "price_momentum": _slot("price_momentum", ("price_volume",), ("momentum", "reaction"), required=False),
            "fund_support": _slot("fund_support", ("fundamental",), ("support", "quality", "valuation"), required=False),
        },
        allowed_combiners=("attention_risk", "risk_minus_confirm", "gated_rank_gap"),
        default_combiner="attention_risk",
        direction=-1,
        allowed_condition_roles=("attention", "crowding", "volatility"),
    ),
    "orderbook_intent": ModeSpec(
        name="orderbook_intent",
        description="日频盘口买卖压力、净委买变化和价差共同刻画未成交交易意愿。",
        slots={
            "orderbook_pressure": _slot("orderbook_pressure", ("orderbook",), ("orderbook_pressure", "buy_pressure", "open_intent")),
            "liquidity_stress": _slot("liquidity_stress", ("orderbook",), ("liquidity", "stress", "spread"), required=False),
            "price_reaction": _slot("price_reaction", ("price_volume",), ("reaction", "momentum"), required=False),
        },
        allowed_combiners=("orderbook_intent",),
        default_combiner="orderbook_intent",
        direction=1,
        allowed_condition_roles=("liquidity", "open_intent", "crowding"),
    ),
    "liquidity_neglect": ModeSpec(
        name="liquidity_neglect",
        description="表面成交活跃但盘口流动性压力高，刻画市场低估交易摩擦。",
        slots={
            "liquidity_stress": _slot("liquidity_stress", ("orderbook",), ("liquidity", "stress", "spread")),
            "turnover_shock": _slot("turnover_shock", ("price_volume",), ("liquidity", "crowding")),
            "flow_confirm": _slot("flow_confirm", ("moneyflow",), ("large_flow", "confirmation"), required=False),
        },
        allowed_combiners=("liquidity_gap", "risk_minus_confirm"),
        default_combiner="liquidity_gap",
        direction=-1,
        allowed_condition_roles=("liquidity", "crowding"),
    ),
    "anchor_momentum": ModeSpec(
        name="anchor_momentum",
        description="52周高点或成交成本锚定下的动量延续。",
        slots={
            "price_anchor": _slot("price_anchor", ("price_volume",), ("price_anchor", "cost_anchor", "anchor")),
            "price_momentum": _slot("price_momentum", ("price_volume",), ("momentum", "anchor_momentum")),
            "flow_confirm": _slot("flow_confirm", ("moneyflow",), ("large_flow", "confirmation"), required=False),
            "orderbook_filter": _slot("orderbook_filter", ("orderbook",), ("open_intent", "confirmation"), required=False),
        },
        allowed_combiners=("anchor_confirm", "confirm", "gated_confirm"),
        default_combiner="anchor_confirm",
        direction=1,
        allowed_condition_roles=("open_intent", "crowding", "liquidity"),
    ),
    "disposition_anchor": ModeSpec(
        name="disposition_anchor",
        description="价格相对成交成本或锚点的位置剥离动量后，刻画处置效应。",
        slots={
            "cost_anchor": _slot("cost_anchor", ("price_volume",), ("cost_anchor", "anchor")),
            "price_momentum": _slot("price_momentum", ("price_volume",), ("momentum", "reaction")),
            "fund_support": _slot("fund_support", ("fundamental",), ("quality", "support"), required=False),
        },
        allowed_combiners=("residual_gap", "rank_gap", "anchor_confirm"),
        default_combiner="residual_gap",
        direction=1,
        allowed_condition_roles=("attention", "crowding"),
    ),
}


def load_behavior_field_rules(metadata_path: str | Path) -> dict[str, BehaviorFieldRule]:
    """Load behavior-field metadata from a JSON file."""

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    raw_rules = metadata.get("behavior_field_rules")
    if raw_rules is None:
        raw_rules = metadata.get("field_rules", {})
    return {field: BehaviorFieldRule.from_dict(rule) for field, rule in raw_rules.items()}


def mode_names() -> tuple[str, ...]:
    return tuple(MODE_REGISTRY)


def get_mode_spec(mode: str) -> ModeSpec:
    if mode not in MODE_REGISTRY:
        raise KeyError(f"unknown behavior mode: {mode!r}")
    return MODE_REGISTRY[mode]


def _slot_matches_rule(slot_name: str, spec: SlotSpec, rule: BehaviorFieldRule) -> bool:
    if rule.allowed_slots and slot_name not in rule.allowed_slots and spec.role not in rule.allowed_slots:
        return False
    if spec.data_families and rule.data_family not in spec.data_families:
        return False
    if spec.behavior_roles and not set(rule.behavior_roles).intersection(spec.behavior_roles):
        return False
    return True


def fields_for_slot(
    field_rules: Mapping[str, BehaviorFieldRule],
    slot_name: str,
    spec: SlotSpec,
) -> list[str]:
    """Return fields that can legally fill one slot."""

    return [
        field
        for field, rule in field_rules.items()
        if _slot_matches_rule(slot_name, spec, rule) and set(rule.allowed_unary_ops).intersection(UNARY_OP_CHOICES)
    ]


def condition_fields_for_mode(
    field_rules: Mapping[str, BehaviorFieldRule],
    mode_spec: ModeSpec,
) -> list[str]:
    """Return fields that can be used as optional gate/state conditions."""

    if not mode_spec.allowed_condition_roles:
        return []
    allowed = set(mode_spec.allowed_condition_roles)
    output = []
    for field, rule in field_rules.items():
        if rule.data_family == "control":
            continue
        if allowed.intersection(rule.behavior_roles) or "state_signal" in rule.allowed_slots or "orderbook_filter" in rule.allowed_slots:
            output.append(field)
    return output


def validate_gene(gene: BehaviorGene, field_rules: Mapping[str, BehaviorFieldRule]) -> list[str]:
    """Return all validation errors for a behavior gene."""

    errors: list[str] = []
    if gene.mode not in MODE_REGISTRY:
        return [f"unknown mode: {gene.mode!r}"]
    mode_spec = MODE_REGISTRY[gene.mode]

    if gene.combiner not in COMBINER_CHOICES:
        errors.append(f"unknown combiner: {gene.combiner!r}")
    elif gene.combiner not in mode_spec.allowed_combiners:
        errors.append(f"combiner {gene.combiner!r} is not allowed for mode {gene.mode!r}")
    elif gene.combiner in {"rank_gap", "residual_gap", "gated_rank_gap", "confirm", "gated_confirm"}:
        n_selected_slots = len([name for name in gene.slots if name in mode_spec.slots])
        if n_selected_slots < 2:
            errors.append(f"combiner {gene.combiner!r} requires at least two selected slots")

    if gene.direction_policy not in DIRECTION_POLICIES:
        errors.append(f"direction_policy must be one of {DIRECTION_POLICIES}, got {gene.direction_policy!r}")
    elif gene.direction_policy != mode_spec.direction_policy and gene.direction_policy == "regime_switch":
        errors.append(f"mode {gene.mode!r} does not allow regime_switch direction policy")

    unknown_slots = sorted(set(gene.slots) - set(mode_spec.slots))
    if unknown_slots:
        errors.append(f"gene has slots not defined by mode {gene.mode!r}: {unknown_slots}")

    for slot_name, slot_spec in mode_spec.slots.items():
        slot_gene = gene.slots.get(slot_name)
        if slot_gene is None:
            if slot_spec.required:
                errors.append(f"missing required slot {slot_name!r} for mode {gene.mode!r}")
            continue
        if slot_gene.field not in field_rules:
            errors.append(f"slot {slot_name!r} uses unknown field {slot_gene.field!r}")
            continue
        rule = field_rules[slot_gene.field]
        if not _slot_matches_rule(slot_name, slot_spec, rule):
            errors.append(
                f"slot {slot_name!r} field {slot_gene.field!r} does not match "
                f"data_families={slot_spec.data_families!r}, behavior_roles={slot_spec.behavior_roles!r}"
            )
        if slot_gene.unary_op not in UNARY_OP_CHOICES:
            errors.append(f"slot {slot_name!r} uses unknown unary_op {slot_gene.unary_op!r}")
        elif slot_gene.unary_op not in rule.allowed_unary_ops:
            errors.append(f"field {slot_gene.field!r} does not allow unary_op {slot_gene.unary_op!r}")

    if len(gene.conditions) > mode_spec.max_conditions:
        errors.append(f"mode {gene.mode!r} allows at most {mode_spec.max_conditions} conditions")
    allowed_condition_roles = set(mode_spec.allowed_condition_roles)
    for idx, condition in enumerate(gene.conditions):
        if condition.field not in field_rules:
            errors.append(f"condition {idx} uses unknown field {condition.field!r}")
            continue
        rule = field_rules[condition.field]
        if condition.condition_op not in CONDITION_OP_CHOICES:
            errors.append(f"condition {idx} uses unknown condition_op {condition.condition_op!r}")
        if condition.unary_op not in UNARY_OP_CHOICES:
            errors.append(f"condition {idx} uses unknown unary_op {condition.unary_op!r}")
        elif condition.unary_op not in rule.allowed_unary_ops:
            errors.append(f"condition field {condition.field!r} does not allow unary_op {condition.unary_op!r}")
        if allowed_condition_roles and not allowed_condition_roles.intersection(rule.behavior_roles):
            if "state_signal" not in rule.allowed_slots and "orderbook_filter" not in rule.allowed_slots:
                errors.append(f"condition field {condition.field!r} does not match mode condition roles")
        if condition.condition_op in {"top_quantile", "bottom_quantile"} and not 0.0 < condition.threshold < 1.0:
            errors.append(f"condition {idx} quantile threshold must be in (0, 1)")

    return errors


def is_valid_gene(gene: BehaviorGene, field_rules: Mapping[str, BehaviorFieldRule]) -> bool:
    return not validate_gene(gene, field_rules)


def gene_key(gene: BehaviorGene) -> tuple[object, ...]:
    """Stable semantic key for de-duplication."""

    slots_key = tuple(sorted((name, slot.field, slot.unary_op) for name, slot in gene.slots.items()))
    conditions_key = tuple(
        sorted(
            (condition.field, condition.unary_op, condition.condition_op, round(condition.threshold, 4))
            for condition in gene.conditions
        )
    )
    return (
        gene.mode,
        gene.combiner,
        slots_key,
        conditions_key,
        gene.direction_policy,
    )


def describe_gene(gene: BehaviorGene) -> str:
    """Create a compact Chinese description for logs and result tables."""

    if gene.mode in MODE_REGISTRY:
        mode_text = MODE_REGISTRY[gene.mode].description
    else:
        mode_text = f"未知模式 {gene.mode}"
    slot_text = ", ".join(f"{name}={slot.unary_op}({slot.field})" for name, slot in sorted(gene.slots.items()))
    condition_text = ""
    if gene.conditions:
        condition_text = "; 条件: " + ", ".join(
            f"{condition.condition_op}({condition.unary_op}({condition.field}), {condition.threshold:.2f})"
            for condition in gene.conditions
        )
    return f"{gene.mode}/{gene.combiner}: {mode_text}; 槽位: {slot_text}{condition_text}"
