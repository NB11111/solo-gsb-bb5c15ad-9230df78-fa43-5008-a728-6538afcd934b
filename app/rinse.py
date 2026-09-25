"""配方切换冲洗：用清水逐轮顶排管路残液，评估首批配方液的浓度变化。

切换配方时，管路中残留着上一个配方版本配出的工作液，残液体积固定为
``R`` mL。切换时按下列模型冲洗：

- **每轮冲洗**：向管路加入 ``W`` mL 清水，与 ``R`` mL 残液**完全混合**后
  排出 ``W`` mL（管内始终只剩 ``R`` mL）。混合均匀后排出，管内残液各元素
  浓度整体乘以稀释因子 ``q = R / (R + W)``，冲洗 n 轮后残液浓度为
  ``旧浓度 × q**n``——趋向 0，而不是趋向新配方浓度。

- **首批配液**：首批体积取**后一配方版本**的目标总体积 ``V`` mL，由管内
  剩余的 ``R`` mL 残液与新注入的 ``V − R`` mL 新配方液组成（要求
  ``R < V``）。元素 e 的首批浓度与相对新配方原浓度的**绝对变化**
  （mg/L）为

      C_e(n) = (旧浓度_e × q**n × R + 新浓度_e × (V − R)) / V
      Δ_e(n) = |C_e(n) − 新浓度_e| = (R/V) × |旧浓度_e × q**n − 新浓度_e|

  注意 Δ_e(n) **不一定随轮数单调下降**：残液被清水不断稀释，冲洗不足时
  残液仍带着旧配方浓度，冲洗过度时残液几乎变成清水、反过来稀释首批。
  因此每个元素的可行轮数是一个**区间** [lo_e, hi_e]（hi_e 可能为 ∞），
  而不是简单的「冲得越多越好」。

接口在 0 ~ max_rounds 内求使**全部元素** Δ_e(n) ≤ 各自上限的**最少轮数**
——各元素可行区间的交集非空时取其左端点（零轮即可时直接返回零轮），
并给出总用水量 n × W。交集为空（含超过 max_rounds）时按 max_rounds
评估，返回该轮数下的逐元素偏移与超限元素，不伪造可行结论。接口只读
两个已保存版本，不改写任何配方数据。
"""
from __future__ import annotations

import math
from typing import Any

# mL / 浓度输出统一保留 6 位小数。
_ROUND = 6
# 与上限比较时的相对容差：Δ 本身可能为 0，故取 scale-aware 容差。
_EPS = 1e-9


class RinseError(ValueError):
    """冲洗请求参数或配方版本内容不合法（体积/上限/元素覆盖等）。"""


def build_switch_rinse(
    *,
    recipe_id: int,
    from_version: int,
    to_version: int,
    from_result: dict[str, Any],
    to_result: dict[str, Any],
    residual_volume_ml: float,
    flush_water_ml: float,
    max_rounds: int,
    element_limits: dict[str, float],
) -> dict[str, Any]:
    """计算两个已保存配方版本之间切换冲洗所需的最少轮数。

    ``from_result`` / ``to_result`` 为两版本持久化的 :class:`SolveResult`
    dict，取其中 ``elements[].actual_concentration`` 作为两版工作液各元素
    的实际浓度，后一版本的 ``total_volume_ml`` 作为首批体积。
    ``element_limits`` 为已归一化元素符号 -> 首批浓度绝对变化上限
    (mg/L)，必须恰好覆盖两版元素的并集。
    """
    old_conc = _actual_concentrations(from_result, "前")
    new_conc = _actual_concentrations(to_result, "后")

    first_batch_volume_ml = _finite_positive(
        to_result.get("total_volume_ml"), "后版本结果总体积 total_volume_ml"
    )
    R = _finite_positive(residual_volume_ml, "管路残液体积 residual_volume_ml")
    W = _finite_positive(flush_water_ml, "每轮冲洗水量 flush_water_ml")
    if (
        not isinstance(max_rounds, int)
        or isinstance(max_rounds, bool)
        or max_rounds < 0
    ):
        raise RinseError("最多轮数 max_rounds 必须为非负整数。")
    if not R < first_batch_volume_ml:
        raise RinseError(
            f"管路残液体积 {R:g} mL 必须小于首批体积（后版本目标总体积）"
            f"{first_batch_volume_ml:g} mL。"
        )

    # 两版元素并集；仅出现在一版中的元素，另一版按浓度 0 处理（冲洗前
    # 残液不含该元素 / 新配方液不含该元素）。
    elements: list[str] = list(old_conc)
    for e in new_conc:
        if e not in elements:
            elements.append(e)

    unknown = sorted(set(element_limits) - set(elements))
    if unknown:
        raise RinseError(
            f"元素上限包含前后版本之外的元素：{ '、'.join(unknown) }；"
            f"两版元素并集为：{ '、'.join(elements) }。"
        )
    missing = sorted(set(elements) - set(element_limits))
    if missing:
        raise RinseError(
            f"缺少元素 { '、'.join(missing) } 的首批浓度变化上限；"
            "请为前后版本的全部元素（并集）逐一提供有限非负上限 (mg/L)。"
        )

    q = R / (R + W)
    fraction = R / first_batch_volume_ml

    # 逐元素求可行轮数区间 [lo_e, hi_e]；整体可行集为各区间与
    # [0, max_rounds] 的交集，非空时最少轮数即各 lo_e 的最大值。
    n_star = 0
    min_hi = math.inf
    any_empty = False
    for e in elements:
        limit = float(element_limits[e])
        if not math.isfinite(limit) or limit < 0.0:
            raise RinseError(f"元素 {e} 的首批浓度变化上限必须为有限非负数。")
        lo, hi = _element_window(
            q, fraction, old_conc.get(e, 0.0), new_conc.get(e, 0.0), limit
        )
        if lo > hi:
            any_empty = True
            continue
        n_star = max(n_star, lo)
        min_hi = min(min_hi, hi)

    feasible = (
        not any_empty and n_star <= min_hi and n_star <= max_rounds
    )
    # 无可行轮数时按最大轮数评估，给出该轮数下的逐元素偏移与超限元素。
    rounds = n_star if feasible else max_rounds

    return _result(
        feasible=feasible,
        rounds=rounds,
        max_rounds=max_rounds,
        q=q,
        R=R,
        W=W,
        elements=elements,
        old_conc=old_conc,
        new_conc=new_conc,
        limits=element_limits,
        recipe_id=recipe_id,
        from_version=from_version,
        to_version=to_version,
        first_batch_volume=first_batch_volume_ml,
    )


# --------------------------------------------------------------------------- #
# Per-element feasible-round window
# --------------------------------------------------------------------------- #
def _element_window(
    q: float, fraction: float, a: float, b: float, limit: float
) -> tuple[int, float]:
    """Feasible integer rounds [lo, hi] for one element (``hi`` 可为 inf）。

    Δ(n) = fraction·|a·q**n − b| ≤ limit（带 scale-aware 容差，与
    :func:`_within` 一致）。``a·q**n`` 随 n 单调不增（从 a 趋向 0），可行
    集必为区间；返回 ``(1, 0)`` 表示空集（该元素任何轮数都无法满足上限）。
    解析边界直接把比较容差并入 M = (limit + tol)/fraction（恒正，也从
    源头上避免了零上限时 log(0) 的问题），snap 循环只校正浮点误差。
    """
    tol = _EPS * max(1.0, limit)
    if q >= 1.0:
        # W 相对 R 可忽略时 q 在浮点下饱和为 1：残液浓度不再变化。
        return (0, math.inf) if fraction * abs(a - b) <= limit + tol else (1, 0)
    if q <= 0.0:
        # W 远大于 R 时 q 下溢为 0：n=0 残液浓度为 a，n≥1 残液浓度为 0。
        n0_ok = fraction * abs(a - b) <= limit + tol
        npos_ok = fraction * b <= limit + tol
        if n0_ok and npos_ok:
            return (0, math.inf)
        if n0_ok:
            return (0, 0)
        if npos_ok:
            return (1, math.inf)
        return (1, 0)
    if a == 0.0:
        # 旧液不含该元素：Δ 恒为 fraction·b，与轮数无关。
        return (0, math.inf) if fraction * b <= limit + tol else (1, 0)

    M = (limit + tol) / fraction  # 容差后的允许 |a·q**n − b| (mg/L)，恒正
    # 上侧 a·q**n ≤ b + M  →  n ≥ lo_real；b + M ≥ a 时 n=0 即满足。
    lo_real = 0.0 if b + M >= a else math.log((b + M) / a) / math.log(q)
    # 下侧 a·q**n ≥ b − M  →  n ≤ hi_real；b − M ≤ 0 时无上界（含 Δ 的
    # 渐近值 fraction·b 已落在容差内的情形）。
    if b - M <= 0.0:
        hi: float = math.inf
    else:
        ratio = (b - M) / a
        if ratio > 1.0:
            return (1, 0)  # a·q**n ≤ a < b − M，任何轮数都不行
        hi_real = math.log(ratio) / math.log(q)
        if hi_real < 0.0:
            return (1, 0)
        hi = float(_snap_hi(q, fraction, a, b, limit, max(0, math.floor(hi_real))))
        if hi < 0:
            return (1, 0)
    lo = _snap_lo(q, fraction, a, b, limit, max(0, math.ceil(lo_real)), hi)
    if lo > hi:
        return (1, 0)
    return (lo, hi)


def _within(q: float, fraction: float, a: float, n: int, b: float, limit: float) -> bool:
    """Δ(n) ≤ limit（带 scale-aware 容差）。"""
    delta = fraction * abs(a * q ** n - b)
    return delta <= limit + _EPS * max(1.0, limit)


def _snap_lo(
    q: float, fraction: float, a: float, b: float, limit: float,
    n: int, hi: float,
) -> int:
    """Snap the real-valued lower bound to the smallest feasible integer."""
    n = max(0, n)
    while n > 0 and _within(q, fraction, a, n - 1, b, limit):
        n -= 1
    # 解析下界已含比较容差，这里只校正浮点误差（通常 0~1 步）；hi 为 ∞ 时
    # 渐近值 fraction·b 在容差内，q**n 终将下溢到 0，循环必然终止。
    while n <= hi and not _within(q, fraction, a, n, b, limit):
        n += 1
    return n


def _snap_hi(
    q: float, fraction: float, a: float, b: float, limit: float, n: int
) -> int:
    """Snap the real-valued upper bound to the largest feasible integer."""
    n = max(0, n)
    while n > 0 and not _within(q, fraction, a, n, b, limit):
        n -= 1
    if not _within(q, fraction, a, n, b, limit):
        return -1
    # 解析上界已含比较容差，向上只校正浮点误差；限制步数，防止在容差
    # 边缘（渐近值恰在容差内时）无限上探。
    for _ in range(3):
        if _within(q, fraction, a, n + 1, b, limit):
            n += 1
        else:
            break
    return n


# --------------------------------------------------------------------------- #
# Result assembly
# --------------------------------------------------------------------------- #
def _finite_positive(value: Any, label: str) -> float:
    """Reject None / non-numeric / NaN / Inf / non-positive volumes."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RinseError(f"{label}必须为有限正数。")
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise RinseError(f"{label}必须为有限正数。")
    return v


def _actual_concentrations(result: dict[str, Any], label: str) -> dict[str, float]:
    """Extract element -> actual_concentration (mg/L) from a saved result."""
    rows = result.get("elements")
    if not isinstance(rows, list) or not rows:
        raise RinseError(
            f"{label}配方版本的求解结果缺少 elements，无法读取元素实际浓度。"
        )
    conc: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        e = row.get("element")
        c = row.get("actual_concentration")
        if (
            not isinstance(e, str)
            or not isinstance(c, (int, float))
            or isinstance(c, bool)
        ):
            raise RinseError(
                f"{label}配方版本求解结果中的元素实际浓度数据不完整。"
            )
        cf = float(c)
        if not math.isfinite(cf) or cf < 0.0:
            raise RinseError(
                f"{label}配方版本元素 {e} 的实际浓度必须为有限非负数。"
            )
        conc[e] = cf
    if not conc:
        raise RinseError(f"{label}配方版本的求解结果不包含任何元素实际浓度。")
    return conc


def _result(
    *,
    feasible: bool,
    rounds: int,
    max_rounds: int,
    q: float,
    R: float,
    W: float,
    elements: list[str],
    old_conc: dict[str, float],
    new_conc: dict[str, float],
    limits: dict[str, float],
    recipe_id: int,
    from_version: int,
    to_version: int,
    first_batch_volume: float,
) -> dict[str, Any]:
    factor = q ** rounds
    elem_rows: list[dict[str, Any]] = []
    over_rows: list[dict[str, Any]] = []
    for e in elements:
        old_c = old_conc.get(e, 0.0)
        new_c = new_conc.get(e, 0.0)
        residual_conc = old_c * factor
        first_conc = (
            residual_conc * R + new_c * (first_batch_volume - R)
        ) / first_batch_volume
        delta = abs(first_conc - new_c)
        limit = float(limits[e])
        within = delta <= limit + _EPS * max(1.0, limit)
        elem_rows.append(
            {
                "element": e,
                "old_concentration_mg_l": _r(old_c),
                "new_concentration_mg_l": _r(new_c),
                "limit_mg_l": _r(limit),
                "residual_concentration_mg_l": _r(residual_conc),
                "first_batch_concentration_mg_l": _r(first_conc),
                "absolute_change_mg_l": _r(delta),
                "within_limit": bool(within),
            }
        )
        if not within:
            over_rows.append(
                {
                    "element": e,
                    "limit_mg_l": _r(limit),
                    "absolute_change_mg_l": _r(delta),
                    "excess_mg_l": _r(max(0.0, delta - limit)),
                }
            )

    total_water = rounds * W
    if feasible:
        message = (
            f"冲洗 {rounds} 轮（每轮 {W:g} mL 清水）后首批各元素浓度变化"
            "均不超过允许上限。"
            if rounds > 0
            else "无需冲洗：首批各元素浓度变化均在允许上限内，可直接切换。"
        )
    else:
        names = "、".join(row["element"] for row in over_rows)
        message = (
            f"冲洗到最多 {rounds} 轮后，元素 {names} 的首批浓度变化仍超过"
            "允许上限；已返回最大轮数下的逐元素偏移与超限元素。"
        )

    return {
        "recipe_id": recipe_id,
        "from_version": from_version,
        "to_version": to_version,
        "feasible": feasible,
        "rounds": int(rounds),
        "max_rounds": int(max_rounds),
        "residual_volume_ml": _r(R),
        "flush_water_ml_per_round": _r(W),
        "first_batch_volume_ml": _r(first_batch_volume),
        "total_water_ml": _r(total_water),
        "elements": elem_rows,
        "exceeded_elements": over_rows,
        "message": message,
    }


def _r(x: float) -> float:
    """Round for output, normalizing -0.0 from float arithmetic."""
    v = round(float(x), _ROUND)
    return 0.0 if v == 0 else v
