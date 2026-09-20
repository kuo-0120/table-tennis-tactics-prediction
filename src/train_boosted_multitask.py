from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold, learning_curve
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import optuna

warnings.filterwarnings("ignore")

RANDOM_SEED = 2026
SCRIPT_VERSION = "0602_full_fixed_v3_catboost_eval_target"

REQUIRED_COLS = [
    "rally_uid", "sex", "match", "numberGame", "rally_id", "strikeNumber",
    "scoreSelf", "scoreOther", "serverGetPoint", "gamePlayerId", "gamePlayerOtherId",
    "strikeId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId",
]

LAG_SOURCE_COLS = ["actionId", "pointId", "spinId", "strengthId", "handId", "positionId", "strikeId"]

CATEGORICAL_FEATURES = [
    "sex", "match", "numberGame", "rally_id",
    "gamePlayerId", "gamePlayerOtherId", "server_player_id",
    "strikeId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId",
    "is_server", "is_receiver", "rally_phase", "score_parity",
    "is_leading", "is_trailing", "is_tied", "is_game_point", "is_early_strike",
    "lag1_actionId", "lag2_actionId", "lag3_actionId",
    "lag1_pointId", "lag2_pointId", "lag3_pointId",
    "lag1_spinId", "lag2_spinId", "lag3_spinId",
    "lag1_strengthId", "lag2_strengthId", "lag3_strengthId",
    "lag1_handId", "lag2_handId", "lag3_handId",
    "lag1_positionId", "lag2_positionId", "lag3_positionId",
    "lag1_strikeId", "lag2_strikeId", "lag3_strikeId",
    "action_x_point", "action_x_spin", "action_x_strength",
    "strike_x_action", "server_x_action", "phase_x_action",
    "last2_action_pattern", "last2_point_pattern",
    "transition_action_token", "transition_point_token",
    # Empirical state-prior features. These are built fold-by-fold from training folds only.
    "heur_action_top1", "heur_point_top1",
    "heur_action_backoff_level", "heur_point_backoff_level", "heur_win_backoff_level",
]
NUMERIC_FEATURES = [
    "strikeNumber", "scoreSelf", "scoreOther",
    "score_diff", "score_sum", "abs_score_diff",
    "strikeNumber_clip", "prefix_len", "prefix_ratio",
    "remaining_len", "remaining_ratio",
    "unique_action_cnt", "unique_point_cnt", "unique_spin_cnt", "unique_position_cnt",
    "current_action_seen_cnt", "current_point_seen_cnt", "current_spin_seen_cnt",
    "same_action_streak", "same_point_streak", "same_spin_streak",
    "server_prefix_hits", "receiver_prefix_hits", "server_hit_ratio", "receiver_hit_ratio",
    "last_score_diff_change", "score_diff_ma3", "score_sum_ma3", "score_self_ma3", "score_other_ma3",
    "heur_action_conf", "heur_point_conf", "heur_server_win_rate",
    "heur_action_support", "heur_point_support", "heur_win_support",
    "heur_action_margin", "heur_point_margin",
    "heur_server_win_logit",
]


TASKS = {
    "action": {"target": "target_actionId", "is_binary": False},
    "point": {"target": "target_pointId", "is_binary": False},
    "win": {"target": "target_serverGetPoint", "is_binary": True},
}


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_table(path: str) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位: {missing}")


def load_raw(train_path: str, test_path: str, sample_path: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    train_df = read_table(train_path)
    test_df = read_table(test_path)
    sample_df = read_table(sample_path) if sample_path else None
    validate_columns(train_df, REQUIRED_COLS)
    validate_columns(test_df, [c for c in REQUIRED_COLS if c != "serverGetPoint"])
    return train_df, test_df, sample_df


def basic_preprocess(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    out = df.copy()

    if not is_train and "serverGetPoint" not in out.columns:
        out["serverGetPoint"] = np.nan

    for c in REQUIRED_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    non_target_cols = [c for c in out.columns if c != "serverGetPoint"]
    out[non_target_cols] = out[non_target_cols].fillna(0)
    out["serverGetPoint"] = out["serverGetPoint"].fillna(0)

    int_cols = [c for c in REQUIRED_COLS if c != "serverGetPoint" and c in out.columns]
    out[int_cols] = out[int_cols].astype(int)
    out["serverGetPoint"] = out["serverGetPoint"].astype(int)

    out = out.sort_values(["match", "numberGame", "rally_id", "strikeNumber", "rally_uid"]).reset_index(drop=True)

    first_idx = out.groupby("rally_uid")["strikeNumber"].idxmin()
    server_map = out.loc[first_idx, ["rally_uid", "gamePlayerId"]].rename(columns={"gamePlayerId": "server_player_id"})
    rally_len = out.groupby("rally_uid").size().rename("rally_total_len").reset_index()

    out = out.merge(server_map, on="rally_uid", how="left")
    out = out.merge(rally_len, on="rally_uid", how="left")

    out["server_player_id"] = out["server_player_id"].fillna(0).astype(int)
    out["rally_total_len"] = out["rally_total_len"].fillna(0).astype(int)
    out["is_server"] = (out["gamePlayerId"] == out["server_player_id"]).astype(int)
    out["is_receiver"] = 1 - out["is_server"]
    return out


def _safe_mode(series: pd.Series, default_value: Any) -> Any:
    if len(series) == 0:
        return default_value
    m = series.mode(dropna=True)
    if len(m) == 0:
        return default_value
    return m.iloc[0]


def _make_token(values: List[int], default_value: int = -1) -> str:
    if len(values) == 0:
        return str(default_value)
    return "_".join(map(str, values))


def build_prefix_frame(raw_df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for rally_uid, g in raw_df.groupby("rally_uid", sort=False):
        g = g.sort_values("strikeNumber").reset_index(drop=True)
        if len(g) == 0:
            continue

        final_server_point = int(_safe_mode(g["serverGetPoint"], 0)) if is_train else -1
        action_hist, point_hist, spin_hist, pos_hist = [], [], [], []
        score_diff_hist, score_sum_hist, score_self_hist, score_other_hist = [], [], [], []
        server_hits, receiver_hits = 0, 0

        prefix_indices = range(len(g) - 1) if is_train else [len(g) - 1]

        for i in prefix_indices:
            cur = g.iloc[i]
            nxt = g.iloc[i + 1] if (is_train and i + 1 < len(g)) else None

            score_diff = int(cur["scoreSelf"] - cur["scoreOther"])
            score_sum = int(cur["scoreSelf"] + cur["scoreOther"])

            action_hist.append(int(cur["actionId"]))
            point_hist.append(int(cur["pointId"]))
            spin_hist.append(int(cur["spinId"]))
            pos_hist.append(int(cur["positionId"]))

            score_diff_hist.append(score_diff)
            score_sum_hist.append(score_sum)
            score_self_hist.append(int(cur["scoreSelf"]))
            score_other_hist.append(int(cur["scoreOther"]))

            if int(cur["is_server"]) == 1:
                server_hits += 1
            else:
                receiver_hits += 1

            feat = {
                "rally_uid": int(rally_uid),
                "match": int(cur["match"]),
                "numberGame": int(cur["numberGame"]),
                "rally_id": int(cur["rally_id"]),
                "sex": int(cur["sex"]),
                "server_player_id": int(cur["server_player_id"]),
                "gamePlayerId": int(cur["gamePlayerId"]),
                "gamePlayerOtherId": int(cur["gamePlayerOtherId"]),
                "strikeId": int(cur["strikeId"]),
                "handId": int(cur["handId"]),
                "strengthId": int(cur["strengthId"]),
                "spinId": int(cur["spinId"]),
                "pointId": int(cur["pointId"]),
                "actionId": int(cur["actionId"]),
                "positionId": int(cur["positionId"]),
                "is_server": int(cur["is_server"]),
                "is_receiver": int(cur["is_receiver"]),
                "strikeNumber": int(cur["strikeNumber"]),
                "scoreSelf": int(cur["scoreSelf"]),
                "scoreOther": int(cur["scoreOther"]),
                "score_diff": score_diff,
                "score_sum": score_sum,
                "abs_score_diff": abs(score_diff),
                "score_parity": score_sum % 2,
                "is_leading": int(score_diff > 0),
                "is_trailing": int(score_diff < 0),
                "is_tied": int(score_diff == 0),
                "is_game_point": int((cur["scoreSelf"] >= 20) or (cur["scoreOther"] >= 20)),
                "is_early_strike": int(cur["strikeNumber"] <= 2),
                "rally_phase": 0 if cur["strikeNumber"] <= 2 else (1 if cur["strikeNumber"] <= 6 else 2),
                "strikeNumber_clip": int(min(cur["strikeNumber"], 20)),
                "prefix_len": i + 1,
                "prefix_ratio": (i + 1) / max(int(cur["rally_total_len"]), 1),
                "remaining_len": max(int(cur["rally_total_len"]) - (i + 1), 0),
                "remaining_ratio": max(int(cur["rally_total_len"]) - (i + 1), 0) / max(int(cur["rally_total_len"]), 1),
                "unique_action_cnt": len(set(action_hist)),
                "unique_point_cnt": len(set(point_hist)),
                "unique_spin_cnt": len(set(spin_hist)),
                "unique_position_cnt": len(set(pos_hist)),
                "current_action_seen_cnt": action_hist.count(int(cur["actionId"])),
                "current_point_seen_cnt": point_hist.count(int(cur["pointId"])),
                "current_spin_seen_cnt": spin_hist.count(int(cur["spinId"])),
                "same_action_streak": 1,
                "same_point_streak": 1,
                "same_spin_streak": 1,
                "server_prefix_hits": server_hits,
                "receiver_prefix_hits": receiver_hits,
                "server_hit_ratio": server_hits / max(i + 1, 1),
                "receiver_hit_ratio": receiver_hits / max(i + 1, 1),
                "last_score_diff_change": 0 if len(score_diff_hist) < 2 else score_diff_hist[-1] - score_diff_hist[-2],
                "score_diff_ma3": float(np.mean(score_diff_hist[-3:])),
                "score_sum_ma3": float(np.mean(score_sum_hist[-3:])),
                "score_self_ma3": float(np.mean(score_self_hist[-3:])),
                "score_other_ma3": float(np.mean(score_other_hist[-3:])),
                "target_actionId": int(nxt["actionId"]) if nxt is not None else -1,
                "target_pointId": int(nxt["pointId"]) if nxt is not None else -1,
                "target_serverGetPoint": int(final_server_point) if is_train else -1,
            }

            for lag in [1, 2, 3]:
                for c in LAG_SOURCE_COLS:
                    hist = g.loc[max(0, i - lag):i - 1, c].tolist() if i - 1 >= 0 else []
                    feat[f"lag{lag}_{c}"] = int(hist[-1]) if len(hist) > 0 else -1

            for seq, out_name in [
                (action_hist, "same_action_streak"),
                (point_hist, "same_point_streak"),
                (spin_hist, "same_spin_streak"),
            ]:
                streak = 1
                for j in range(len(seq) - 2, -1, -1):
                    if seq[j] == seq[-1]:
                        streak += 1
                    else:
                        break
                feat[out_name] = streak

            feat["action_x_point"] = f"{feat['actionId']}_{feat['pointId']}"
            feat["action_x_spin"] = f"{feat['actionId']}_{feat['spinId']}"
            feat["action_x_strength"] = f"{feat['actionId']}_{feat['strengthId']}"
            feat["strike_x_action"] = f"{feat['strikeId']}_{feat['actionId']}"
            feat["server_x_action"] = f"{feat['is_server']}_{feat['actionId']}"
            feat["phase_x_action"] = f"{feat['rally_phase']}_{feat['actionId']}"
            feat["last2_action_pattern"] = _make_token(action_hist[-3:])
            feat["last2_point_pattern"] = _make_token(point_hist[-3:])
            feat["transition_action_token"] = f"{feat['lag1_actionId']}_{feat['actionId']}"
            feat["transition_point_token"] = f"{feat['lag1_pointId']}_{feat['pointId']}"

            # These are overwritten by apply_state_priors() using only the training fold.
            # Keeping placeholders here makes downstream schema stable, but the values below are not used as priors.
            feat["heur_action_top1"] = "NA"
            feat["heur_point_top1"] = "NA"
            feat["heur_action_backoff_level"] = "global"
            feat["heur_point_backoff_level"] = "global"
            feat["heur_win_backoff_level"] = "global"
            feat["heur_action_conf"] = 0.0
            feat["heur_point_conf"] = 0.0
            feat["heur_server_win_rate"] = 0.5
            feat["heur_action_support"] = 0.0
            feat["heur_point_support"] = 0.0
            feat["heur_win_support"] = 0.0
            feat["heur_action_margin"] = 0.0
            feat["heur_point_margin"] = 0.0
            feat["heur_server_win_logit"] = 0.0

            rows.append(feat)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("prefix frame 為空")
    return df


def _global_class_distribution(df: pd.DataFrame, target_col: str) -> Dict[int, float]:
    vc = df[target_col].value_counts(normalize=True)
    return {int(k): float(v) for k, v in vc.items()}


def _mode_int(series: pd.Series, default_value: int = 0) -> int:
    m = series.mode(dropna=True)
    if len(m) == 0:
        return int(default_value)
    return int(m.iloc[0])


def _safe_logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _group_class_table(
    train_prefix: pd.DataFrame,
    key_cols: List[str],
    target_col: str,
    global_prob: Dict[int, float],
    alpha: float = 8.0,
) -> pd.DataFrame:
    """Build smoothed top-1 class prior for one key level.

    The smoothing prevents tiny one-sample states from getting overconfident and makes
    the heuristic useful rather than just memorizing rare transitions.
    """
    if len(key_cols) == 0:
        raise ValueError("key_cols cannot be empty")

    grp = train_prefix.groupby(key_cols + [target_col]).size().rename("cnt").reset_index()
    if grp.empty:
        return pd.DataFrame(columns=key_cols + ["_top1", "_conf", "_support", "_margin"])

    grp["_total"] = grp.groupby(key_cols)["cnt"].transform("sum").astype(float)
    grp["_global_prob"] = grp[target_col].astype(int).map(global_prob).fillna(0.0).astype(float)
    grp["_prob"] = (grp["cnt"].astype(float) + alpha * grp["_global_prob"]) / (grp["_total"] + alpha)

    # A larger support is trusted more. Ties prefer the smaller class id for deterministic output.
    grp = grp.sort_values(
        key_cols + ["_prob", "cnt", target_col],
        ascending=[True] * len(key_cols) + [False, False, True],
    )

    grp["_rank_in_state"] = grp.groupby(key_cols).cumcount()
    top = grp.loc[grp["_rank_in_state"] == 0].copy().reset_index(drop=True)
    second = grp.loc[grp["_rank_in_state"] == 1, key_cols + ["_prob"]].copy().reset_index(drop=True)
    second = second.rename(columns={"_prob": "_second_prob"})

    top = top.merge(second, on=key_cols, how="left")
    top["_second_prob"] = top["_second_prob"].fillna(0.0)
    top["_margin"] = (top["_prob"] - top["_second_prob"]).clip(lower=0.0)
    top = top.rename(columns={target_col: "_top1", "_prob": "_conf", "_total": "_support"})

    return top[key_cols + ["_top1", "_conf", "_support", "_margin"]]


def _group_win_table(
    train_prefix: pd.DataFrame,
    key_cols: List[str],
    target_col: str,
    default_value: float,
    alpha: float = 30.0,
) -> pd.DataFrame:
    """Build smoothed server-win prior for one key level."""
    if len(key_cols) == 0:
        raise ValueError("key_cols cannot be empty")

    grp = train_prefix.groupby(key_cols)[target_col].agg(["mean", "count"]).reset_index()
    if grp.empty:
        return pd.DataFrame(columns=key_cols + ["_rate", "_support"])

    grp["_rate"] = (grp["mean"].astype(float) * grp["count"].astype(float) + default_value * alpha) / (
        grp["count"].astype(float) + alpha
    )
    grp["_support"] = grp["count"].astype(float)
    return grp[key_cols + ["_rate", "_support"]]


def build_state_priors(train_prefix: pd.DataFrame) -> Dict[str, Any]:
    """Build fold-safe empirical priors.

    These priors are deliberately created from the current training fold only. In CV,
    validation rows never participate in their own heuristic features, reducing leakage.
    """
    action_levels = [
        ["actionId", "pointId", "spinId", "positionId", "rally_phase", "is_server"],
        ["actionId", "pointId", "spinId", "rally_phase", "is_server"],
        ["actionId", "pointId", "rally_phase", "is_server"],
        ["actionId", "spinId", "rally_phase", "is_server"],
        ["actionId", "rally_phase", "is_server"],
        ["actionId", "pointId"],
        ["actionId"],
    ]
    point_levels = [
        ["actionId", "pointId", "spinId", "positionId", "rally_phase", "is_server"],
        ["actionId", "pointId", "spinId", "rally_phase", "is_server"],
        ["actionId", "pointId", "rally_phase", "is_server"],
        ["pointId", "spinId", "rally_phase", "is_server"],
        ["pointId", "rally_phase", "is_server"],
        ["actionId", "pointId"],
        ["pointId"],
    ]
    win_levels = [
        ["sex", "score_diff", "score_sum", "rally_phase", "is_server", "actionId", "pointId", "spinId", "positionId"],
        ["sex", "score_diff", "rally_phase", "is_server", "actionId", "pointId", "spinId"],
        ["score_diff", "rally_phase", "is_server", "actionId", "pointId"],
        ["score_diff", "rally_phase", "is_server"],
        ["rally_phase", "is_server", "actionId", "pointId"],
        ["is_server", "actionId", "pointId"],
        ["is_server", "rally_phase"],
    ]

    priors: Dict[str, Any] = {
        "action_levels": action_levels,
        "point_levels": point_levels,
        "win_levels": win_levels,
    }

    priors["action_default"] = _mode_int(train_prefix["target_actionId"], 0)
    priors["point_default"] = _mode_int(train_prefix["target_pointId"], 0)
    priors["win_default"] = float(train_prefix["target_serverGetPoint"].mean())

    action_global_prob = _global_class_distribution(train_prefix, "target_actionId")
    point_global_prob = _global_class_distribution(train_prefix, "target_pointId")
    priors["action_default_conf"] = float(max(action_global_prob.values()) if action_global_prob else 0.0)
    priors["point_default_conf"] = float(max(point_global_prob.values()) if point_global_prob else 0.0)

    priors["action_tables"] = [
        _group_class_table(train_prefix, key_cols, "target_actionId", action_global_prob, alpha=8.0)
        for key_cols in action_levels
    ]
    priors["point_tables"] = [
        _group_class_table(train_prefix, key_cols, "target_pointId", point_global_prob, alpha=8.0)
        for key_cols in point_levels
    ]
    priors["win_tables"] = [
        _group_win_table(train_prefix, key_cols, "target_serverGetPoint", priors["win_default"], alpha=30.0)
        for key_cols in win_levels
    ]

    return priors


def _drop_existing_prior_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Remove placeholder/generated prior columns before merging.

    Without this, pandas creates *_x / *_y suffixes during merge, which was the real
    cause of KeyError: 'heur_action_top1'. We intentionally overwrite the heuristic
    columns with fold-safe empirical values instead of leaving placeholders.
    """
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")


def _attach_multilevel_top1_conf(
    df: pd.DataFrame,
    tables: List[pd.DataFrame],
    levels: List[List[str]],
    default_class: int,
    default_conf: float,
    top1_col: str,
    conf_col: str,
    support_col: str,
    margin_col: str,
    backoff_col: str,
) -> pd.DataFrame:
    out = _drop_existing_prior_columns(df.copy(), [top1_col, conf_col, support_col, margin_col, backoff_col])

    out[top1_col] = int(default_class)
    out[conf_col] = float(default_conf)
    out[support_col] = 0.0
    out[margin_col] = 0.0
    out[backoff_col] = "global"

    filled = pd.Series(False, index=out.index)

    for level_idx, (key_cols, table) in enumerate(zip(levels, tables)):
        if table.empty or any(c not in out.columns for c in key_cols):
            continue

        tmp_cols = {
            "_top1": f"__{top1_col}_lvl{level_idx}",
            "_conf": f"__{conf_col}_lvl{level_idx}",
            "_support": f"__{support_col}_lvl{level_idx}",
            "_margin": f"__{margin_col}_lvl{level_idx}",
        }
        tmp = table.rename(columns=tmp_cols)
        out = out.merge(tmp, on=key_cols, how="left")

        top_tmp = tmp_cols["_top1"]
        conf_tmp = tmp_cols["_conf"]
        sup_tmp = tmp_cols["_support"]
        margin_tmp = tmp_cols["_margin"]

        hit = (~filled) & out[top_tmp].notna()
        if hit.any():
            out.loc[hit, top1_col] = out.loc[hit, top_tmp].astype(int).values
            out.loc[hit, conf_col] = out.loc[hit, conf_tmp].astype(float).values
            out.loc[hit, support_col] = out.loc[hit, sup_tmp].astype(float).values
            out.loc[hit, margin_col] = out.loc[hit, margin_tmp].astype(float).values
            out.loc[hit, backoff_col] = f"L{level_idx}"
            filled.loc[hit.index[hit]] = True

        out = out.drop(columns=[top_tmp, conf_tmp, sup_tmp, margin_tmp], errors="ignore")

    out[top1_col] = out[top1_col].fillna(default_class).astype(int).astype(str)
    out[conf_col] = pd.to_numeric(out[conf_col], errors="coerce").fillna(default_conf).astype(float)
    out[support_col] = pd.to_numeric(out[support_col], errors="coerce").fillna(0.0).astype(float)
    out[margin_col] = pd.to_numeric(out[margin_col], errors="coerce").fillna(0.0).astype(float)
    out[backoff_col] = out[backoff_col].fillna("global").astype(str)
    return out


def _attach_multilevel_win_rate(
    df: pd.DataFrame,
    tables: List[pd.DataFrame],
    levels: List[List[str]],
    default_value: float,
    out_col: str,
    support_col: str,
    logit_col: str,
    backoff_col: str,
) -> pd.DataFrame:
    out = _drop_existing_prior_columns(df.copy(), [out_col, support_col, logit_col, backoff_col])

    out[out_col] = float(default_value)
    out[support_col] = 0.0
    out[logit_col] = _safe_logit(default_value)
    out[backoff_col] = "global"

    filled = pd.Series(False, index=out.index)

    for level_idx, (key_cols, table) in enumerate(zip(levels, tables)):
        if table.empty or any(c not in out.columns for c in key_cols):
            continue

        tmp_cols = {
            "_rate": f"__{out_col}_lvl{level_idx}",
            "_support": f"__{support_col}_lvl{level_idx}",
        }
        tmp = table.rename(columns=tmp_cols)
        out = out.merge(tmp, on=key_cols, how="left")

        rate_tmp = tmp_cols["_rate"]
        sup_tmp = tmp_cols["_support"]

        hit = (~filled) & out[rate_tmp].notna()
        if hit.any():
            out.loc[hit, out_col] = out.loc[hit, rate_tmp].astype(float).values
            out.loc[hit, support_col] = out.loc[hit, sup_tmp].astype(float).values
            out.loc[hit, backoff_col] = f"L{level_idx}"
            filled.loc[hit.index[hit]] = True

        out = out.drop(columns=[rate_tmp, sup_tmp], errors="ignore")

    out[out_col] = pd.to_numeric(out[out_col], errors="coerce").fillna(default_value).clip(1e-6, 1 - 1e-6).astype(float)
    out[support_col] = pd.to_numeric(out[support_col], errors="coerce").fillna(0.0).astype(float)
    out[logit_col] = out[out_col].map(_safe_logit).astype(float)
    out[backoff_col] = out[backoff_col].fillna("global").astype(str)
    return out


def apply_state_priors(df: pd.DataFrame, priors: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()

    out = _attach_multilevel_top1_conf(
        out,
        priors["action_tables"],
        priors["action_levels"],
        priors["action_default"],
        priors["action_default_conf"],
        "heur_action_top1",
        "heur_action_conf",
        "heur_action_support",
        "heur_action_margin",
        "heur_action_backoff_level",
    )

    out = _attach_multilevel_top1_conf(
        out,
        priors["point_tables"],
        priors["point_levels"],
        priors["point_default"],
        priors["point_default_conf"],
        "heur_point_top1",
        "heur_point_conf",
        "heur_point_support",
        "heur_point_margin",
        "heur_point_backoff_level",
    )

    out = _attach_multilevel_win_rate(
        out,
        priors["win_tables"],
        priors["win_levels"],
        priors["win_default"],
        "heur_server_win_rate",
        "heur_win_support",
        "heur_server_win_logit",
        "heur_win_backoff_level",
    )

    return out


def finalize_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CATEGORICAL_FEATURES:
        if c not in out.columns:
            out[c] = "NA"
        out[c] = out[c].fillna("NA").astype(str)
    for c in NUMERIC_FEATURES:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
    return out


def prepare_xy(df: pd.DataFrame, target_name: Optional[str], label_encoder: Optional[LabelEncoder] = None):
    feat_df = finalize_feature_types(df)
    use_cat = [c for c in CATEGORICAL_FEATURES if c in feat_df.columns]
    use_num = [c for c in NUMERIC_FEATURES if c in feat_df.columns]
    X = feat_df[use_cat + use_num].copy()

    y = None
    le = label_encoder
    if target_name is not None:
        raw_y = feat_df[target_name].values
        if label_encoder is None:
            le = LabelEncoder()
            y = le.fit_transform(raw_y)
        else:
            y = label_encoder.transform(raw_y)
    return X, y, le, use_cat, use_num


def make_sample_weight(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    cnt_map = {c: n for c, n in zip(classes, counts)}
    w = np.array([1.0 / math.sqrt(cnt_map[v]) for v in y], dtype=float)
    return w / np.mean(w)


def task_metric(task: str, y_true: np.ndarray, prob: np.ndarray) -> float:
    if task == "win":
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, prob[:, 1]))
    pred = np.argmax(prob, axis=1)
    return float(f1_score(y_true, pred, average="macro", zero_division=0))


def score_components(action_true, action_prob, point_true, point_prob, win_true, win_prob) -> Dict[str, float]:
    action_f1 = float(f1_score(action_true, np.argmax(action_prob, axis=1), average="macro", zero_division=0))
    point_f1 = float(f1_score(point_true, np.argmax(point_prob, axis=1), average="macro", zero_division=0))
    win_auc = 0.5 if len(np.unique(win_true)) < 2 else float(roc_auc_score(win_true, win_prob))
    return {
        "action_macro_f1": action_f1,
        "point_macro_f1": point_f1,
        "win_auc": win_auc,
        "proxy_score": 0.4 * action_f1 + 0.4 * point_f1 + 0.2 * win_auc,
    }


def align_proba(prob: np.ndarray, model_classes: np.ndarray, global_classes: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    if prob.ndim == 1:
        prob = np.vstack([1 - prob, prob]).T

    model_classes = np.asarray(model_classes)
    global_classes = np.asarray(global_classes)

    if len(model_classes) == len(global_classes) and np.array_equal(model_classes, global_classes):
        return prob

    aligned = np.zeros((prob.shape[0], len(global_classes)), dtype=float)
    cls_to_idx = {c: i for i, c in enumerate(global_classes)}
    for j, cls in enumerate(model_classes):
        if cls in cls_to_idx:
            aligned[:, cls_to_idx[cls]] = prob[:, j]

    row_sum = aligned.sum(axis=1, keepdims=True)
    zero_rows = row_sum.squeeze() <= 0
    if np.any(zero_rows):
        aligned[zero_rows] = 1.0 / len(global_classes)
        row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / np.clip(row_sum, 1e-12, None)


def convert_for_lgbm(X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    out = X.copy()
    for c in cat_cols:
        out[c] = out[c].astype("category")
    return out


def sanitize_catboost_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Make CatBoost bootstrap parameters legal.

    CatBoost rules used here:
    - Bayesian bootstrap can use bagging_temperature, but not subsample.
    - Bernoulli / MVS bootstrap can use subsample, but not bagging_temperature.

    This is not a skip-fix; it preserves regularization intent while preventing
    Optuna from creating illegal CatBoost parameter combinations.
    """
    out = dict(params)
    bootstrap_type = str(out.get("bootstrap_type", "Bernoulli"))

    if bootstrap_type == "Bayesian":
        out.pop("subsample", None)
        out.setdefault("bagging_temperature", 1.0)
    elif bootstrap_type in {"Bernoulli", "MVS"}:
        out.pop("bagging_temperature", None)
        out.setdefault("subsample", 0.85)
    else:
        # Conservative fallback for unknown / version-specific bootstrap choices.
        out.pop("bagging_temperature", None)
        out.pop("subsample", None)

    return out


def fit_predict_catboost(X_tr, y_tr, X_va, y_va, task, cat_cols, sample_weight, params, seed):
    is_binary = TASKS[task]["is_binary"]
    obj = "Logloss" if is_binary else "MultiClass"
    metric = "AUC" if is_binary else "TotalF1:average=Macro"

    default_params = {
        "loss_function": obj,
        "eval_metric": metric,
        "iterations": 1400,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.5,
        "subsample": 0.9,
        "bootstrap_type": "Bernoulli",
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
    }
    default_params.update(params or {})
    default_params = sanitize_catboost_params(default_params)

    train_pool = Pool(X_tr, y_tr, cat_features=cat_cols, weight=sample_weight)
    valid_pool = Pool(X_va, y_va, cat_features=cat_cols)
    model = CatBoostClassifier(**default_params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=False)
    prob = model.predict_proba(valid_pool)
    return model, np.asarray(prob)


def fit_predict_lightgbm(X_tr, y_tr, X_va, y_va, task, cat_cols, sample_weight, params, seed):
    is_binary = TASKS[task]["is_binary"]
    n_classes = int(np.max(y_tr) + 1)

    default_params = {
        "n_estimators": 1400,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
    }
    default_params.update(params or {})

    if is_binary:
        model = lgb.LGBMClassifier(objective="binary", **default_params)
    else:
        model = lgb.LGBMClassifier(objective="multiclass", num_class=n_classes, **default_params)

    Xtr = convert_for_lgbm(X_tr, cat_cols)
    Xva = convert_for_lgbm(X_va, cat_cols)
    model.fit(
        Xtr,
        y_tr,
        sample_weight=sample_weight,
        eval_set=[(Xva, y_va)],
        eval_metric="auc" if is_binary else "multi_logloss",
        callbacks=[lgb.early_stopping(120, verbose=False)],
        categorical_feature=cat_cols,
    )
    prob = model.predict_proba(Xva)
    return model, np.asarray(prob)


def predict_model(model_name: str, model, X, cat_cols):
    if model_name == "catboost":
        return np.asarray(model.predict_proba(Pool(X, cat_features=cat_cols)))
    if model_name == "lightgbm":
        return np.asarray(model.predict_proba(convert_for_lgbm(X, cat_cols)))
    raise ValueError(model_name)


def tune_params(model_name: str, task: str, train_df: pd.DataFrame, seed: int, n_trials: int) -> Dict[str, Any]:
    if n_trials <= 0:
        return {}

    target_col = TASKS[task]["target"]
    groups = train_df["match"].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=seed)
    tr_idx, va_idx = next(splitter.split(train_df, groups=groups))
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    priors = build_state_priors(tr_df)
    tr_df = apply_state_priors(tr_df, priors)
    va_df = apply_state_priors(va_df, priors)

    global_le = LabelEncoder()
    global_le.fit(train_df[target_col].values)
    global_classes = global_le.classes_

    X_tr, y_tr, _, cat_cols, _ = prepare_xy(tr_df, target_col, global_le)
    X_va, y_va, _, _, _ = prepare_xy(va_df, target_col, global_le)
    sample_weight = make_sample_weight(y_tr)

    def objective(trial: optuna.Trial) -> float:
        if model_name == "catboost":
            bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bernoulli", "Bayesian"])
            params = {
                "iterations": trial.suggest_int("iterations", 700, 1800),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
                "depth": trial.suggest_int("depth", 6, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 1e-3, 3.0, log=True),
                "bootstrap_type": bootstrap_type,
            }
            if bootstrap_type == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 3.0)
            else:
                params["subsample"] = trial.suggest_float("subsample", 0.65, 1.0)

            model, prob = fit_predict_catboost(X_tr, y_tr, X_va, y_va, task, cat_cols, sample_weight, params, seed)
        elif model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 700, 1800),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 20.0, log=True),
                "max_depth": trial.suggest_int("max_depth", -1, 12),
            }
            model, prob = fit_predict_lightgbm(X_tr, y_tr, X_va, y_va, task, cat_cols, sample_weight, params, seed)
        else:
            raise ValueError(model_name)

        prob = align_proba(prob, getattr(model, "classes_", np.arange(prob.shape[1])), global_classes)
        return task_metric(task, y_va, prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


class Calibrator:
    def __init__(self, method: str = "none"):
        self.method = method
        self.model = None

    def fit(self, y_true: np.ndarray, prob: np.ndarray):
        prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
        y_true = np.asarray(y_true, dtype=int)

        if self.method == "none":
            self.model = None
            return self

        if self.method == "sigmoid":
            logit = np.log(prob / (1 - prob)).reshape(-1, 1)
            lr = LogisticRegression(max_iter=1000, solver="lbfgs")
            lr.fit(logit, y_true)
            self.model = lr
            return self

        if self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(prob, y_true)
            self.model = iso
            return self

        raise ValueError(self.method)

    def predict(self, prob: np.ndarray):
        prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "none" or self.model is None:
            return prob
        if self.method == "sigmoid":
            logit = np.log(prob / (1 - prob)).reshape(-1, 1)
            return self.model.predict_proba(logit)[:, 1]
        if self.method == "isotonic":
            return np.asarray(self.model.predict(prob), dtype=float)
        return prob


def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(w, dtype=float), 1e-12)
    return w / w.sum()


def brute_force_weights(prob_list: List[np.ndarray], y_true: np.ndarray, task: str) -> np.ndarray:
    if len(prob_list) == 1:
        return np.array([1.0])

    best_w = None
    best_score = -1e18
    for a in np.arange(0.0, 1.01, 0.05):
        b = 1.0 - a
        w = normalize_weights(np.array([a, b]))
        prob = w[0] * prob_list[0] + w[1] * prob_list[1]
        sc = task_metric(task, y_true, prob)
        if sc > best_score:
            best_score = sc
            best_w = w
    return best_w


def maybe_suppress_point0(prob: np.ndarray, class_labels: np.ndarray, factor: float) -> np.ndarray:
    if factor >= 0.999:
        return prob
    out = prob.copy()
    idx = np.where(class_labels == 0)[0]
    if len(idx) == 0:
        return out
    j = int(idx[0])
    out[:, j] *= factor
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out


def select_win_calibration(y_true: np.ndarray, prob: np.ndarray):
    best_method, best_brier, best_cal = "none", float("inf"), Calibrator("none").fit(y_true, prob)
    report = {}
    for method in ["none", "sigmoid", "isotonic"]:
        cal = Calibrator(method).fit(y_true, prob)
        pp = cal.predict(prob)
        brier = float(brier_score_loss(y_true, pp))
        report[method] = {
            "brier": brier,
            "logloss": float(log_loss(y_true, np.clip(pp, 1e-6, 1 - 1e-6))),
            "auc": 0.5 if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, pp)),
        }
        if brier < best_brier:
            best_brier = brier
            best_method = method
            best_cal = cal
    return best_method, best_cal, report


def save_json(obj: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def export_feature_importance(fitted_models: Dict[str, List[Any]], feature_names: List[str], out_png: str):
    import matplotlib.pyplot as plt

    scores = {f: [] for f in feature_names}
    for model_name, models in fitted_models.items():
        for model in models:
            if model_name == "catboost":
                imp = model.get_feature_importance()
            else:
                imp = model.feature_importances_
            for f, v in zip(feature_names, imp):
                scores[f].append(float(v))

    avg_imp = pd.Series({k: float(np.mean(v)) if len(v) else 0.0 for k, v in scores.items()})
    top = avg_imp.sort_values(ascending=False).head(25).sort_values(ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top.index, top.values)
    plt.title("特徵重要度 Top 25")
    plt.xlabel("平均 importance")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def export_win_curves(y_true: np.ndarray, prob: np.ndarray, out_dir: str):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    prob = np.clip(prob, 1e-6, 1 - 1e-6)

    fpr, tpr, _ = roc_curve(y_true, prob)
    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else 0.5
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Rally Winner ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "win_roc_curve.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, prob)
    ap = average_precision_score(y_true, prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Rally Winner PR Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "win_pr_curve.png", dpi=160)
    plt.close()

    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Rally Winner Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "win_calibration.png", dpi=160)
    plt.close()


def export_learning_curve_plot(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cat_cols: List[str], params: Dict[str, Any], out_png: str, seed: int):
    import matplotlib.pyplot as plt

    if len(np.unique(y)) < 2:
        return

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(params.get("n_estimators", 500)),
        learning_rate=float(params.get("learning_rate", 0.03)),
        num_leaves=int(params.get("num_leaves", 63)),
        min_child_samples=int(params.get("min_child_samples", 30)),
        subsample=float(params.get("subsample", 0.85)),
        colsample_bytree=float(params.get("colsample_bytree", 0.85)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=seed,
        n_jobs=-1,
    )

    X2 = convert_for_lgbm(X, cat_cols)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    train_sizes, train_scores, test_scores = learning_curve(
        model, X2, y, groups=groups, cv=gkf, scoring="roc_auc",
        n_jobs=1, train_sizes=np.linspace(0.2, 1.0, 5),
    )

    plt.figure(figsize=(6, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="train")
    plt.plot(train_sizes, test_scores.mean(axis=1), marker="o", label="validation")
    plt.xlabel("Training size")
    plt.ylabel("ROC AUC")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def run_eda_plots(train_raw: pd.DataFrame, prefix_train: pd.DataFrame, out_dir: str):
    import matplotlib.pyplot as plt

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    rally_len = train_raw.groupby("rally_uid").size()
    plt.figure(figsize=(7, 4))
    plt.hist(rally_len.values, bins=min(50, max(int(rally_len.max()), 10)))
    plt.title("Rally 長度分布")
    plt.xlabel("每個 rally 的拍數")
    plt.ylabel("數量")
    plt.tight_layout()
    plt.savefig(outp / "eda_rally_length.png", dpi=160)
    plt.close()

    action_counts = prefix_train["target_actionId"].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    plt.bar(action_counts.index.astype(str), action_counts.values)
    plt.title("下一拍 action 分布")
    plt.xlabel("actionId")
    plt.ylabel("數量")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(outp / "eda_action_dist.png", dpi=160)
    plt.close()

    point_counts = prefix_train["target_pointId"].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    plt.bar(point_counts.index.astype(str), point_counts.values)
    plt.title("下一拍 point 分布")
    plt.xlabel("pointId")
    plt.ylabel("數量")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(outp / "eda_point_dist.png", dpi=160)
    plt.close()

    strike_win = prefix_train.groupby("strikeNumber_clip")["target_serverGetPoint"].mean()
    plt.figure(figsize=(7, 4))
    plt.plot(strike_win.index, strike_win.values, marker="o")
    plt.title("不同拍次的發球者得分率")
    plt.xlabel("strikeNumber_clip")
    plt.ylabel("serverGetPoint mean")
    plt.tight_layout()
    plt.savefig(outp / "eda_strike_vs_win.png", dpi=160)
    plt.close()


def fit_one_task(task: str, train_prefix: pd.DataFrame, test_prefix: pd.DataFrame, n_splits: int, n_trials: int, seed: int, out_dir: str, point0_suppress: float = 1.0):
    target_col = TASKS[task]["target"]
    groups = train_prefix["match"].values
    actual_splits = min(n_splits, len(np.unique(groups)))
    if actual_splits < 2:
        raise ValueError("至少需要 2 個不同 match 才能做 group CV")

    # 以全資料建立一致 label space
    global_le = LabelEncoder()
    global_le.fit(train_prefix[target_col].values)
    global_classes = global_le.classes_

    y_for_split = train_prefix[target_col].values
    try:
        splitter = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=seed).split(train_prefix, y_for_split, groups=groups)
    except Exception:
        splitter = GroupKFold(n_splits=actual_splits).split(train_prefix, groups=groups)

    tuned_params = {
        "catboost": tune_params("catboost", task, train_prefix, seed, n_trials),
        "lightgbm": tune_params("lightgbm", task, train_prefix, seed + 73, n_trials),
    }

    fitted_models = {"catboost": [], "lightgbm": []}
    oof_prob_dict = {}
    test_prob_sum_dict = {}
    fold_scores = {"catboost": [], "lightgbm": []}

    for fold, (tr_idx, va_idx) in enumerate(splitter, start=1):
        tr_df = train_prefix.iloc[tr_idx].reset_index(drop=True)
        va_df = train_prefix.iloc[va_idx].reset_index(drop=True)

        priors = build_state_priors(tr_df)
        tr2 = apply_state_priors(tr_df, priors)
        va2 = apply_state_priors(va_df, priors)
        te2 = apply_state_priors(test_prefix.copy(), priors)

        X_tr, y_tr, _, cat_cols, _ = prepare_xy(tr2, target_col, global_le)
        X_va, y_va, _, _, _ = prepare_xy(va2, target_col, global_le)
        X_te, _, _, _, _ = prepare_xy(te2, None)

        sample_weight = make_sample_weight(y_tr)

        for model_name in ["catboost", "lightgbm"]:
            if model_name not in oof_prob_dict:
                oof_prob_dict[model_name] = np.zeros((len(train_prefix), len(global_classes)), dtype=float)
                test_prob_sum_dict[model_name] = np.zeros((len(test_prefix), len(global_classes)), dtype=float)

            if model_name == "catboost":
                model, va_prob = fit_predict_catboost(
                    X_tr, y_tr, X_va, y_va, task, cat_cols, sample_weight, tuned_params[model_name], seed + fold
                )
            else:
                model, va_prob = fit_predict_lightgbm(
                    X_tr, y_tr, X_va, y_va, task, cat_cols, sample_weight, tuned_params[model_name], seed + fold
                )

            model_classes = getattr(model, "classes_", np.arange(va_prob.shape[1]))
            va_prob = align_proba(va_prob, model_classes, global_classes)
            te_prob = align_proba(predict_model(model_name, model, X_te, cat_cols), model_classes, global_classes)

            oof_prob_dict[model_name][va_idx] = va_prob
            test_prob_sum_dict[model_name] += te_prob / actual_splits
            fitted_models[model_name].append(model)
            fold_scores[model_name].append(task_metric(task, y_va, va_prob))

    model_names = ["catboost", "lightgbm"]
    y_full = global_le.transform(train_prefix[target_col].values)
    oof_list = [oof_prob_dict[m] for m in model_names]
    weights = brute_force_weights(oof_list, y_full, task)

    oof_ens = weights[0] * oof_list[0] + weights[1] * oof_list[1]
    test_ens = weights[0] * test_prob_sum_dict["catboost"] + weights[1] * test_prob_sum_dict["lightgbm"]

    calibration_report = {}
    if task == "win":
        _, calibrator, calibration_report = select_win_calibration(y_full, oof_ens[:, 1])
        oof_ens[:, 1] = calibrator.predict(oof_ens[:, 1])
        oof_ens[:, 0] = 1 - oof_ens[:, 1]
        test_ens[:, 1] = calibrator.predict(test_ens[:, 1])
        test_ens[:, 0] = 1 - test_ens[:, 1]
    elif task == "point":
        oof_ens = maybe_suppress_point0(oof_ens, global_classes, point0_suppress)
        test_ens = maybe_suppress_point0(test_ens, global_classes, point0_suppress)

    X_example = finalize_feature_types(apply_state_priors(train_prefix.copy(), build_state_priors(train_prefix.copy())))
    X_example = X_example[[c for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES if c in X_example.columns]]
    export_feature_importance(fitted_models, list(X_example.columns), str(Path(out_dir) / task / f"{task}_feature_importance.png"))

    joblib.dump(
        {
            "classes_": global_classes,
            "weights": weights,
            "fold_scores": fold_scores,
            "params": tuned_params,
        },
        Path(out_dir) / task / f"{task}_artifacts.joblib",
    )

    return {
        "task": task,
        "classes_": global_classes,
        "oof_prob": oof_ens,
        "test_prob": test_ens,
        "y_true": y_full,
        "weights": weights,
        "fold_scores": fold_scores,
        "params": tuned_params,
        "calibration_report": calibration_report,
    }


def build_submission(sample_df, test_prefix, action_res, point_res, win_res):
    action_labels = action_res["classes_"]
    point_labels = point_res["classes_"]

    pred = pd.DataFrame({
        "rally_uid": test_prefix["rally_uid"].values,
        "actionId": action_labels[np.argmax(action_res["test_prob"], axis=1)].astype(int),
        "pointId": point_labels[np.argmax(point_res["test_prob"], axis=1)].astype(int),
        "serverGetPoint": np.clip(win_res["test_prob"][:, 1], 1e-6, 1 - 1e-6),
    }).sort_values("rally_uid").drop_duplicates("rally_uid").reset_index(drop=True)

    if sample_df is not None and "rally_uid" in sample_df.columns:
        sub = sample_df[["rally_uid"]].merge(pred, on="rally_uid", how="left")
        sub["actionId"] = sub["actionId"].fillna(0).astype(int)
        sub["pointId"] = sub["pointId"].fillna(0).astype(int)
        sub["serverGetPoint"] = sub["serverGetPoint"].fillna(0.5)
        return sub
    return pred


def compare_significance(base_scores: List[float], new_scores: List[float]) -> Dict[str, Any]:
    if len(base_scores) != len(new_scores) or len(base_scores) == 0:
        return {"statistic": None, "pvalue": None}
    res = wilcoxon(np.asarray(new_scores) - np.asarray(base_scores))
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--sample", type=str, default="")
    parser.add_argument("--outdir", type=str, default="outputs_tt_boosted")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--point0-suppress", type=float, default=0.55)
    args = parser.parse_args()

    print(f"[INFO] Running {SCRIPT_VERSION}")
    seed_everything(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw, sample_df = load_raw(args.train, args.test, args.sample if args.sample else None)
    train_raw = basic_preprocess(train_raw, is_train=True)
    test_raw = basic_preprocess(test_raw, is_train=False)

    prefix_train = build_prefix_frame(train_raw, is_train=True)
    prefix_test = build_prefix_frame(test_raw, is_train=False)

    run_eda_plots(train_raw, prefix_train, str(outdir / "eda"))

    action_res = fit_one_task("action", prefix_train, prefix_test, args.n_splits, args.n_trials, args.seed, str(outdir), args.point0_suppress)
    point_res = fit_one_task("point", prefix_train, prefix_test, args.n_splits, args.n_trials, args.seed + 17, str(outdir), args.point0_suppress)
    win_res = fit_one_task("win", prefix_train, prefix_test, args.n_splits, args.n_trials, args.seed + 31, str(outdir), args.point0_suppress)

    metrics = score_components(
        action_res["y_true"], action_res["oof_prob"],
        point_res["y_true"], point_res["oof_prob"],
        win_res["y_true"], win_res["oof_prob"][:, 1],
    )

    save_json(metrics, str(outdir / "cv_metrics.json"))
    export_win_curves(win_res["y_true"], win_res["oof_prob"][:, 1], str(outdir / "plots"))

    X_win = finalize_feature_types(apply_state_priors(prefix_train.copy(), build_state_priors(prefix_train.copy())))
    X_win = X_win[[c for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES if c in X_win.columns]]
    export_learning_curve_plot(
        X_win,
        win_res["y_true"],
        groups=prefix_train["match"].values,
        cat_cols=[c for c in CATEGORICAL_FEATURES if c in X_win.columns],
        params=win_res["params"]["lightgbm"],
        out_png=str(outdir / "plots" / "learning_curve_win.png"),
        seed=args.seed,
    )

    submission = build_submission(sample_df, prefix_test, action_res, point_res, win_res)
    submission.to_csv(outdir / "submission.csv", index=False, encoding="utf-8-sig")

    summary = {
        "assumptions": {
            "dataset_path": "unspecified",
            "metric": "proxy_score = 0.4*macro_f1(action)+0.4*macro_f1(point)+0.2*auc(win)",
            "compute": "single GPU / 8 CPU assumed; CPU fallback available",
        },
        "metrics": metrics,
        "action_weights": {"catboost": float(action_res["weights"][0]), "lightgbm": float(action_res["weights"][1])},
        "point_weights": {"catboost": float(point_res["weights"][0]), "lightgbm": float(point_res["weights"][1])},
        "win_weights": {"catboost": float(win_res["weights"][0]), "lightgbm": float(win_res["weights"][1])},
        "win_calibration": win_res["calibration_report"],
    }
    save_json(summary, str(outdir / "summary.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
