"""配方切换冲洗：求满足首批各元素浓度变化上限的最少冲洗轮数。

从旧配方切换到新配方时，管路中残留有体积 ``V_r`` (mL) 的旧配方工作液
（各元素浓度取旧配方**已保存版本**的实际浓度 ``C_old`` mg/L）。每一轮
冲洗向管路注入 ``W`` mL 清水、与残液**完全混合**后排出同等水量，因此
冲洗后管路中仍有 ``V_r`` mL 液体，而各元素浓度被稀释为原来的

    q = V_r / (V_r + W)

倍。冲洗 ``n`` 轮后残液中元素 e 的浓度为 ``C_old_e · q**n``。

冲洗结束后配制切换后的**首批**新配方：首批体积取**后版本（新版本）目标
体积** ``V_b``，由管路中剩余的 ``V_r`` mL 残液与 ``V_b - V_r`` mL 新配方液
（元素浓度取新版本实际浓度 ``C_new``）组成，首批元素浓度为

    C_batch_e(n) = (V_r · C_old_e · q**n
                    + (V_b - V_r) · C_new_e) / V_b

首批相对新配方**原浓度**的绝对变化（mg/L）：

    delta_e(n) = |C_batch_e(n) - C_new_e|
               = (V_r / V_b) · q**n · |C_old_e - C_new_e|

``delta_e(n)`` 随 n 单调不增，因此取 0, 1, ..., max_rounds 中首个满足
全部元素 ``delta_e(n) <= limit_e`` 的轮数（含 n = 0：零轮即可行时直接
返回零轮、零用水）；到达最大轮数仍不满足时返回 ``feasible=false``，并
给出最大轮数下的逐元素偏移与超限元素。最少轮数按闭式
``n >= ln(limit/delta0) / ln(q)`` 解析求得（并用直接代回校验，与逐轮
枚举逐位一致），总用水 ``n · W`` 随结果一并返回。

元素集合取前后两个版本实际结果中出现元素的**并集**：只在一个版本中
出现的元素，另一版本浓度按 0 计；请求中的元素上限必须恰好覆盖该并集
（多给 / 缺少均为 422）。本接口只读取两个已保存版本，不改写任何数据。
"""
from __future__ import annotations

import math
from typing import Any

from .config import ELEMENTS
from .models import SolveRequest

# mL rounding for reported volumes.
_ROUND = 4
# Concentration / change rounding (mg/L).
_CONC_ROUND = 6
# Relative slack when comparing an achieved change against its limit; only
# absorbs float round-off from the q**n dilution recursion (relative error
# ~1e-16), never a real gap, and stays correct even for tiny limits.
_REL_EPS = 1e-9


def _within_limit(change: float, limit: float) -> bool:
    return change <= limit + _REL_EPS * max(change, limit)


def _required_rounds(delta0: float, limit: float, q: float, change_at) -> float:
    """Smallest n >= 0 with ``delta0 * q**n <= limit`` (math.inf if never).

    Closed form ``n >= ln(limit/delta0) / ln(q)`` (note ln(q) < 0).  The
    returned integer is verified by **direct evaluation** at itself and its
    two neighbours, so the verdict is bit-compatible with an explicit
    round-by-round scan even when ``ceil`` lands on a boundary.
    """
    if _within_limit(delta0, limit):
        return 0
    if limit <= 0.0:
        # A zero limit can only be met in the limit n -> inf (unless delta0
        # is already 0, handled above); no finite round count suffices.
        return math.inf
    raw = math.ceil(math.log(limit / delta0) / math.log(q))
    n0 = max(0, int(raw))
    for n in (max(0, n0 - 1), n0, n0 + 1):
        if _within_limit(change_at(delta0, n), limit):
            return n
    # Float round-off pushed the estimate below the real boundary; walk up
    # (bounded: the geometric decay guarantees convergence).
    n = n0 + 2
    while not _within_limit(change_at(delta0, n), limit):
        n += 1
    return n


class RinseError(ValueError):
    """配方切换冲洗请求非法（残液/冲洗水/体积关系、元素上限不匹配等）。"""


def build_rinse_plan(
    *,
    new_recipe_id: int,
    new_version: int,
    old_recipe_id: int,
    old_version: int,
    old_request_payload: dict[str, Any],
    old_result_payload: dict[str, Any],
    new_request_payload: dict[str, Any],
    new_result_payload: dict[str, Any],
    pipe_residual_ml: float,
    rinse_water_per_round_ml: float,
    max_rounds: int,
    element_limits: dict[str, float],
) -> dict[str, Any]:
    """Compute the minimum-round recipe-switchover rinse plan.

    The two ``*_payload`` pairs are the persisted :class:`SolveRequest` /
    solve-result dicts of the previous (``old``) and target (``new``) saved
    recipe versions.  Element actual concentrations are read from the saved
    results; the first-batch volume is the new version's target volume.
    Raises :class:`RinseError` on invalid volume relations or element-limit
    coverage mismatches.
    """
    V_r = float(pipe_residual_ml)
    W = float(rinse_water_per_round_ml)
    n_max = int(max_rounds)

    for name, val in (("管路残液体积", V_r), ("每轮冲洗水量", W)):
        if not math.isfinite(val) or val <= 0.0:
            raise RinseError(f"{name}必须为有限正数。")
    if not isinstance(n_max, int) or isinstance(n_max, bool) or n_max < 0:
        raise RinseError("最多轮数必须为非负整数。")

    new_req = SolveRequest.model_validate(new_request_payload)
    # Validating the old request too makes sure a (manually corrupted) stored
    # version cannot poison the calculation with non-finite numbers.
    SolveRequest.model_validate(old_request_payload)
    V_b = float(new_req.target_volume_ml)
    if not math.isfinite(V_b) or V_b <= 0.0:
        raise RinseError("后版本目标体积必须为有限正数。")  # pragma: no cover
    if V_r >= V_b:
        raise RinseError(
            f"管路残液体积 {V_r:g} mL 必须小于首批体积（后版本目标体积）"
            f" {V_b:g} mL：首批由残液与新配方液共同组成，需要为新配方液"
            "留出正的体积空间。"
        )

    old_conc = _actual_concentrations(old_result_payload)
    new_conc = _actual_concentrations(new_result_payload)
    elems = _union_elements(old_conc, new_conc)

    limits: dict[str, float] = {}
    for key, val in element_limits.items():
        num = float(val)
        if not math.isfinite(num) or num < 0.0:
            raise RinseError(f"元素 {key} 的首批浓度变化上限必须为有限非负数。")
        limits[key] = num

    unknown = sorted(set(limits) - set(elems))
    if unknown:
        raise RinseError(
            f"元素上限中存在前后两个版本均未出现的元素：{ '、'.join(unknown) }；"
            f"前后版本实际包含的元素并集为：{ '、'.join(elems) }。"
        )
    missing = [e for e in elems if e not in limits]
    if missing:
        raise RinseError(
            f"元素上限缺少：{ '、'.join(missing) }；上限必须覆盖前后两个版本"
            "出现的全部元素（只在一个版本出现的元素按另一版本浓度为 0 计入）。"
        )

    # Per-round dilution factor: residual V_r mixed with W clean water, then
    # W drained — the pipe always re-holds V_r at concentration q times before.
    q = V_r / (V_r + W)
    fill_fraction = V_r / V_b

    def change_at(delta0: float, n: int) -> float:
        # Element change at round n given its round-0 change; evaluated with
        # the exact same q**n expression a round-by-round simulation produces.
        return delta0 * (q ** n)

    # Minimum round per element from delta0 * q**n <= limit, i.e.
    # n >= ln(limit/delta0) / ln(q) (0 < q < 1).  A small verification
    # window around the analytic ceil guards float round-off and keeps the
    # verdict identical to a literal round-by-round scan.
    required: dict[str, float] = {}
    for e in elems:
        delta0 = fill_fraction * abs(old_conc.get(e, 0.0) - new_conc.get(e, 0.0))
        required[e] = _required_rounds(delta0, limits[e], q, change_at)

    required_max = max(required.values(), default=0)
    feasible = required_max <= n_max
    chosen = int(required_max) if feasible else n_max

    # Guard pathological inputs (e.g. max_rounds = 10**100): the reported
    # total water must stay a finite JSON number.
    total_water = chosen * W
    if not math.isfinite(float(total_water)):
        raise RinseError(
            f"冲洗总用水量（轮数 {chosen:g} × 每轮 {W:g} mL）超出可表示范围，"
            "请减小最多轮数或每轮冲洗水量。"
        )

    def changes_at(n: int) -> dict[str, float]:
        return {
            e: change_at(
                fill_fraction * abs(old_conc.get(e, 0.0) - new_conc.get(e, 0.0)),
                n,
            )
            for e in elems
        }

    changes = changes_at(chosen)
    # Per-element verdicts come from the analytic required round counts; at
    # extreme rounds q**n underflows to 0 in double precision, so re-evaluating
    # the rounded change could hide an element that still misses its limit.
    within = {e: required[e] <= chosen for e in elems}

    elements = _element_report(
        elems=elems,
        rounds=chosen,
        q=q,
        V_r=V_r,
        V_b=V_b,
        old_conc=old_conc,
        new_conc=new_conc,
        changes=changes,
        limits=limits,
        within=within,
    )
    violating = [e for e in elems if not within[e]]

    if feasible:
        violations: list[dict[str, Any]] = []
        if chosen == 0:
            message = (
                "零轮冲洗即可切换：首批由残液与新配方液混合后，各元素相对"
                "新配方原浓度的绝对变化均不超过给定上限，无需用水。"
            )
        else:
            message = (
                f"冲洗 {chosen} 轮后首批各元素相对新配方原浓度的绝对变化"
                f"均不超过上限；共消耗清水 {_r(chosen * W):g} mL"
                f"（每轮 {_r(W):g} mL）。"
            )
    else:
        def _violation_msg(e: str) -> str:
            shown = round(changes[e], _CONC_ROUND)
            need = (
                "无穷多"
                if not math.isfinite(required[e])
                else f"{int(required[e]):d}"
            )
            base = (
                f"元素 {e} 在冲洗 {int(n_max):d} 轮（允许的最多轮数）后仍超过首批"
                f"浓度变化上限 {limits[e]:.6g} mg/L；该元素至少需要 "
                f"{need} 轮冲洗"
            )
            if shown > limits[e]:
                base += f"，当前首批绝对变化为 {shown:g} mg/L"
            else:
                # At extreme round counts the residual dilution underflows to
                # 0 in double precision even though the limit is mathematically
                # still out of reach (e.g. limit = 0); say so explicitly.
                base += "；其理论残差低于双精度可表示范围，无法用数值直接展示"
            return base + "；请增加最多轮数、每轮冲洗水量或放宽该元素上限。"

        violations = [
            {
                "code": "ELEMENT_CHANGE_EXCEEDED",
                "element": e,
                "limit": round(limits[e], _CONC_ROUND),
                "actual": round(changes[e], _CONC_ROUND),
                "message": _violation_msg(e),
            }
            for e in violating
        ]
        message = (
            f"冲洗到最多 {n_max} 轮仍无法同时满足全部元素上限，超限元素："
            f"{ '、'.join(violating) }；已返回最大轮数下的逐元素偏移。"
        )

    return {
        "recipe_id": new_recipe_id,
        "version": new_version,
        "old_recipe_id": old_recipe_id,
        "old_version": old_version,
        "feasible": feasible,
        "rounds": chosen,
        "max_rounds": n_max,
        "total_water_ml": _r(chosen * W),
        "first_batch_volume_ml": _r(V_b),
        "pipe_residual_ml": _r(V_r),
        "rinse_water_per_round_ml": _r(W),
        "rinse_factor": round(q, 10),
        "elements": elements,
        "violating_elements": violating,
        "violations": violations,
        "message": message,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _actual_concentrations(result_payload: dict[str, Any]) -> dict[str, float]:
    """Extract ``element -> actual_concentration mg/L`` from a saved result."""
    rows = result_payload.get("elements")
    if not isinstance(rows, list):  # pragma: no cover - stored by this service
        raise RinseError("已保存版本的求解结果缺少 elements 列表，无法读取实际浓度。")
    conc: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or "element" not in row:
            raise RinseError("已保存版本的求解结果元素行格式非法。")  # pragma: no cover
        e = str(row["element"])
        val = row.get("actual_concentration")
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise RinseError(  # pragma: no cover
                f"已保存版本中元素 {e} 的实际浓度缺失或非法。"
            )
        num = float(val)
        if not math.isfinite(num) or num < 0.0:
            raise RinseError(  # pragma: no cover
                f"已保存版本中元素 {e} 的实际浓度必须为有限非负数。"
            )
        conc[e] = num
    return conc


def _union_elements(
    old_conc: dict[str, float], new_conc: dict[str, float]
) -> list[str]:
    """Union of both versions' elements in canonical ELEMENTS order."""
    found = set(old_conc) | set(new_conc)

    def key(e: str) -> tuple[int, str]:
        return (ELEMENTS.index(e), e) if e in ELEMENTS else (len(ELEMENTS), e)

    return sorted(found, key=key)


def _element_report(
    *,
    elems: list[str],
    rounds: int,
    q: float,
    V_r: float,
    V_b: float,
    old_conc: dict[str, float],
    new_conc: dict[str, float],
    changes: dict[str, float],
    limits: dict[str, float],
    within: dict[str, bool],
) -> list[dict[str, Any]]:
    """Per-element residual/first-batch concentrations after ``rounds`` rounds."""
    decay = q ** rounds  # residual concentration decay relative to C_old
    report: list[dict[str, Any]] = []
    for e in elems:
        c_old = old_conc.get(e, 0.0)
        c_new = new_conc.get(e, 0.0)
        residual_conc = c_old * decay
        first_batch_conc = (
            V_r * residual_conc + (V_b - V_r) * c_new
        ) / V_b
        change = changes[e]
        is_within = within[e]
        report.append(
            {
                "element": e,
                "old_concentration": round(c_old, _CONC_ROUND),
                "new_concentration": round(c_new, _CONC_ROUND),
                "residual_concentration": round(residual_conc, _CONC_ROUND),
                "first_batch_concentration": round(first_batch_conc, _CONC_ROUND),
                "absolute_change": round(change, _CONC_ROUND),
                "allowed_change": round(limits[e], _CONC_ROUND),
                "within_limit": bool(is_within),
                "status": "ok" if is_within else "over",
            }
        )
    return report


def _r(x: float) -> float:
    """Round for output, normalizing -0.0 from float arithmetic."""
    v = round(float(x), _ROUND)
    return 0.0 if v == 0 else v
