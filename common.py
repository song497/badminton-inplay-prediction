# -*- coding: utf-8 -*-
"""
experiments_v2 / common.py
==========================
Shared infrastructure for the revised (v2) experiment framework.

Design goals
------------
* Faithful re-implementation of the original model architectures (see
  ``notes`` in README.md for the exact source files and line ranges).
* Fixed evaluation protocol: per-run seed, 3-way train/val/test split,
  val-based EarlyStopping, test evaluated exactly once.
* Absolute paths derived from ``__file__`` -- the working directory does not
  matter (the original scripts only worked with CWD == D:\\data).
* Every run writes one JSON record + one npz curve file whose name carries a
  condition suffix (the original CNN.py and CNN_no_location.py overwrote each
  other's ``auc_curves.npz``).

Only files under D:\\data\\experiments_v2\\ are ever written.
"""
from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from config import ROOT, DATA, RESULTS, FIGURES  # noqa: E402

# ============================================================================
# 0. paths (all absolute, all rooted at this file's directory)
# ============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
CURVES_DIR = os.path.join(HERE, "results", "curves")
LOG_DIR = os.path.join(HERE, "results", "logs")
REPORT_DIR = os.path.join(HERE, "results", "tables")

BLSR_V2 = os.path.join(DATA_DIR, "dataset_blsr_v2.csv")
MACRO_V2 = os.path.join(DATA_DIR, "macro_features_v2.csv")
LINDAN_BLSR = None  # reserved for an optional cross-dataset check; data not included in this release
SPLIT_FILES = {
    "train": os.path.join(DATA_DIR, "train_match_ids_v2.txt"),
    "val": os.path.join(DATA_DIR, "val_match_ids_v2.txt"),
    "test": os.path.join(DATA_DIR, "test_match_ids_v2.txt"),
}

SHOT_COLS = ["is_target_turn", "aroundhead", "backhand", "time_proportion",
             "hit_area", "player_location_area", "opponent_location_area"]
LOCATION_COLS = ["hit_area", "player_location_area", "opponent_location_area"]
CTX_COLS = ["roundscore_diff", "continuous_score"]


def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, CURVES_DIR, LOG_DIR, REPORT_DIR):
        os.makedirs(d, exist_ok=True)


class Tee:
    """Duplicate stdout into results/logs/<name>.log (UTF-8, no BOM)."""

    def __init__(self, path: str):
        self.path = path
        self.fh = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.fh.write(s)

    def flush(self):
        self.stdout.flush()
        self.fh.flush()
        try:
            self.fh.flush()
        except Exception:
            pass


def start_log(name: str) -> Tee:
    ensure_dirs()
    tee = Tee(os.path.join(LOG_DIR, name + ".log"))
    sys.stdout = tee
    return tee


def stop_log(tee: Tee) -> None:
    try:
        sys.stdout = tee.stdout
        tee.fh.close()
    except Exception:
        pass


# ============================================================================
# 1. CLI / env configuration
# ============================================================================
def cli_parser(description: str):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--conditions", default=None,
                   help="comma/space separated subset of conditions (default: all)")
    p.add_argument("--n-runs", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", default=None, help="extra tag used in output file names")
    p.add_argument("--list", action="store_true", help="list conditions and exit")
    return p


def resolve_conditions(all_conditions: Sequence[str], arg: str | None) -> List[str]:
    if not arg:
        return list(all_conditions)
    want = [c.strip() for c in arg.replace(",", " ").split() if c.strip()]
    bad = [c for c in want if c not in all_conditions]
    if bad:
        raise SystemExit("unknown condition(s): %s\nknown: %s"
                         % (", ".join(bad), ", ".join(all_conditions)))
    return want


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return v.strip().lower() in ("1", "true", "yes", "on", "y")


def config(n_runs_cli=None, epochs_cli=None) -> Dict[str, int]:
    """N_RUNS / EPOCHS resolve: CLI > env > default."""
    n_runs = n_runs_cli if n_runs_cli is not None else env_int("N_RUNS", 5)
    epochs = epochs_cli if epochs_cli is not None else env_int("EPOCHS", 100)
    return {"n_runs": int(n_runs), "epochs": int(epochs)}


def set_seed(seed: int) -> None:
    """Per-run independent seed (original scripts set 42 once, outside the loop)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ============================================================================
# 2. data loading
# ============================================================================
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_split_ids() -> Dict[str, List[int]]:
    out = {}
    for name, path in SPLIT_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        out[name] = [int(x) for x in np.loadtxt(path, dtype=int).ravel().tolist()]
    return out


def load_macro() -> Tuple[pd.DataFrame, List[str]]:
    macro = pd.read_csv(MACRO_V2)
    feat_cols = [c for c in macro.columns if c not in ("game_id", "match_id", "label")]
    return macro, feat_cols


def load_blsr(path: str = BLSR_V2, drop_location: bool = False):
    """Load the shot-level table and build the v2 encodings.

    v2 fix (F2, encoding leakage): the type vocabulary and the area vocabulary
    are fitted on the *training* split only.  The originals ran ``pd.factorize``
    over the full table (including test) and derived ``num_types``/``num_areas``
    from the full table, so the embedding-table size depended on the test set.

    Returns
    -------
    df        : DataFrame with ``type_code`` and integer area columns
    meta      : dict with num_types / num_areas / vocabularies
    """
    df = pd.read_csv(path)
    for c in LOCATION_COLS:
        df[c] = df[c].fillna(0).astype(int)
    if drop_location:
        for c in LOCATION_COLS:
            df[c] = 0

    ids = load_split_ids()
    train_mask = df["match_id"].isin(ids["train"])
    if train_mask.sum() == 0:                      # e.g. a small external check set
        train_mask = pd.Series(True, index=df.index)

    tr_types = sorted(df.loc[train_mask, "type"].dropna().unique().tolist())
    type_to_code = {t: i + 1 for i, t in enumerate(tr_types)}   # 0 == padding/unknown
    df["type_code"] = df["type"].map(type_to_code).fillna(0).astype(np.int32)

    areas = df.loc[train_mask, LOCATION_COLS].to_numpy()
    area_max = int(areas.max()) if areas.size else 0
    num_areas = max(area_max, 1) + 1
    for c in LOCATION_COLS:
        df[c] = np.clip(df[c].to_numpy(), 0, num_areas - 1).astype(int)

    meta = {
        "num_types": len(tr_types) + 1,
        "num_areas": num_areas,
        "type_vocab": tr_types,
        "n_unknown_type_rows": int((df["type_code"] == 0).sum()),
    }
    return df, meta


# ============================================================================
# 3. sequence construction
# ============================================================================
def build_rally_sequences(df: pd.DataFrame, matches: Sequence[int],
                          min_len: int = 2, label_from: str = "first") -> List[dict]:
    """One sample per rally (micro models), original grouping/ordering."""
    sub = df[df["match_id"].isin(list(matches))]
    seqs = []
    for (game_id, rally_id), g in sub.groupby(["game_id", "rally_id"], sort=True):
        if len(g) < min_len:
            continue
        g = g.sort_values("time_proportion")
        row = g.iloc[0] if label_from == "first" else g.iloc[-1]
        seqs.append({
            "game_id": int(game_id),
            "rally_id": int(rally_id),
            "match_id": int(g["match_id"].iloc[0]),
            "shots": g[SHOT_COLS].to_numpy(np.float32),
            "types": g["type_code"].to_numpy(np.int32),
            "ctx": g[CTX_COLS].iloc[0].to_numpy(np.float32),
            "label": float(row["is_target_win"]),
        })
    return seqs


def build_game_sequences(df: pd.DataFrame, matches: Sequence[int]) -> List[dict]:
    """One sample per game = concatenated shot sequence (end-to-end models)."""
    sub = df[df["match_id"].isin(list(matches))]
    seqs = []
    for game_id, g in sub.groupby("game_id", sort=True):
        g = g.sort_values("time_proportion")
        seqs.append({
            "game_id": int(game_id),
            "match_id": int(g["match_id"].iloc[0]),
            "shots": g[SHOT_COLS].to_numpy(np.float32),
            "label": float(g["is_target_win"].iloc[-1]),
        })
    return seqs


# ============================================================================
# 4. tf.data pipelines
# ============================================================================
def _micro_generator(seqs, shuffle, rng):
    idx = list(range(len(seqs)))
    while True:
        if shuffle:
            rng.shuffle(idx)
        for i in idx:
            s = seqs[i]
            yield (s["shots"], s["types"], s["ctx"],
                   np.array(len(s["shots"]), dtype=np.int32)), s["label"]
        if not shuffle:
            return


_MICRO_SPEC = (
    (tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
     tf.TensorSpec(shape=(None,), dtype=tf.int32),
     tf.TensorSpec(shape=(2,), dtype=tf.float32),
     tf.TensorSpec(shape=(), dtype=tf.int32)),
    tf.TensorSpec(shape=(), dtype=tf.float32),
)


def make_micro_dataset(seqs, batch_size=32, shuffle=False, seed=42, repeat=False):
    rng = np.random.RandomState(seed)
    ds = tf.data.Dataset.from_generator(
        lambda: _micro_generator(seqs, shuffle, rng), output_signature=_MICRO_SPEC)
    if repeat:
        ds = ds.repeat()
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=((tf.TensorShape([None, 7]), tf.TensorShape([None]),
                        tf.TensorShape([2]), tf.TensorShape([])), tf.TensorShape([])),
        padding_values=((0.0, 0, 0.0, 0), 0.0))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _e2e_generator(seqs, rng, shuffle, macro=None, batch_size=8):
    """Original e2e pattern: an *infinite* generator (steps_per_epoch is always
    passed explicitly, see common.steps_for)."""
    idx = list(range(len(seqs)))
    while True:
        if shuffle:
            rng.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            batch = [seqs[i] for i in batch_idx]
            max_len = max(s["shots"].shape[0] for s in batch)
            padded = np.zeros((len(batch), max_len, 7), dtype=np.float32)
            for i, s in enumerate(batch):
                padded[i, :s["shots"].shape[0], :] = s["shots"]
            y = np.array([s["label"] for s in batch], dtype=np.float32)
            if macro is None:
                yield padded, y
            else:
                m = np.stack([macro[s["game_id"]] for s in batch]).astype(np.float32)
                yield (padded, m), y
        if not shuffle:
            return


def make_e2e_dataset(seqs, batch_size=8, shuffle=False, seed=42, macro=None):
    """The generator already emits *padded batches* (original design), so the
    element spec is rank-3 and no ``padded_batch`` is applied on top.

    Bug found while building v2: stacking ``padded_batch`` on top of the
    already-batched generator produced shape (8, 471, 7) where (None, 7) was
    declared, so ``model.fit`` failed with
    "generator yielded an element of shape (8, 471, 7) ...".
    """
    rng = np.random.RandomState(seed)
    spec = (tf.TensorSpec(shape=(None, None, 7), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32))
    if macro is not None:
        macro_dim = len(next(iter(macro.values())))
        spec = ((tf.TensorSpec(shape=(None, None, 7), dtype=tf.float32),
                 tf.TensorSpec(shape=(None, macro_dim), dtype=tf.float32)),
                tf.TensorSpec(shape=(None,), dtype=tf.float32))
    ds = tf.data.Dataset.from_generator(
        lambda: _e2e_generator(seqs, rng, shuffle, macro, batch_size),
        output_signature=spec)
    return ds.prefetch(tf.data.AUTOTUNE)


def steps_for(seqs, batch_size: int) -> int:
    """Steps per epoch for the from_generator pipelines.

    ``tf.data.Dataset.from_generator`` reports UNKNOWN cardinality, and Keras
    then treats the dataset as an infinite stream (this stall was observed while
    building v2).  Always pass steps_per_epoch/validation_steps explicitly.
    """
    return max(1, int(np.ceil(len(seqs) / float(batch_size))))


def padded_matrix(seqs, feat_dim=7) -> np.ndarray:
    max_len = max(len(s["shots"]) for s in seqs)
    out = np.zeros((len(seqs), max_len, feat_dim), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i, :len(s["shots"]), :] = s["shots"]
    return out


# ============================================================================
# 5. model architectures (verbatim reproductions)
# ============================================================================
L2_REG = 0.01
EPS = 1e-9


class TemporalScore(layers.Layer):
    """delta = sigmoid(theta + mu * time_prop)   (CNN.py lines 134-136)."""

    def call(self, inputs):
        theta, mu, tau = inputs
        return tf.sigmoid(theta + mu * tau)


class StaggeredMerge(layers.Layer):
    """Two Conv1D branches interleaved even/odd (CNN.py lines 145-153).

    NOTE (see README 'original bugs'): the reshape length is ``L//2`` where L is
    the *padded* sequence length, so on an odd-length batch the last position is
    dropped and the output length is batch-dependent.  Reproduced as-is.
    """

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def call(self, inputs):
        c1, c2 = inputs
        seq_len = tf.shape(c1)[1]
        new_len = seq_len - (seq_len % 2)
        c1_even = c1[:, :new_len:2, :]
        c2_odd = c2[:, 1:new_len:2, :]
        stacked = tf.stack([c1_even, c2_odd], axis=2)
        merged = tf.reshape(stacked, (tf.shape(c1)[0], new_len // 2, 2 * self.filters))
        return merged


class WeightedSum(layers.Layer):
    def call(self, inputs):
        w, x = inputs
        return tf.reduce_sum(w * x, axis=1)


def build_cnn_gru_attention(num_types: int, num_areas: int, embed_dim: int = 15,
                            area_embed_dim: int = 10, rnn_units: int = 32,
                            filters: int = 32, kernel: int = 3,
                            l2: float = L2_REG):
    """Micro CNN + BiGRU + attention (D:\\data\\微观模型\\CNN.py lines 115-169).

    Faithfully reproduces the original, *including* the fact that the
    ``StaggeredMerge`` output has a batch-dependent length and therefore loses
    the embedding mask, so the attention softmax also spreads mass over the
    padded positions (see README, problem B6/B7).  A masked variant was
    attempted in v2 and rejected: the staggered length is ``ceil(L/2)`` while
    the mask has length ``L``, so a correct mask needs the same reference length
    the original does not expose.  Keeping the original behaviour makes the v2
    numbers directly comparable to the paper's.
    """
    input_shots = tf.keras.Input(shape=(None, 7), name="shots")
    input_types = tf.keras.Input(shape=(None,), dtype=tf.int32, name="types")
    input_ctx = tf.keras.Input(shape=(2,), name="ctx")
    input_len = tf.keras.Input(shape=(), dtype=tf.int32, name="len")

    type_embed = layers.Embedding(num_types, embed_dim, mask_zero=True, name="type_embed")
    theta_embed = layers.Embedding(num_types, 1, name="theta_embed")
    mu_embed = layers.Embedding(num_types, 1, name="mu_embed")
    area_embed = layers.Embedding(num_areas, area_embed_dim, mask_zero=True, name="area_embed")

    hit_area = layers.Lambda(lambda x: tf.cast(x[:, :, 4], tf.int32))(input_shots)
    player_area = layers.Lambda(lambda x: tf.cast(x[:, :, 5], tf.int32))(input_shots)
    opp_area = layers.Lambda(lambda x: tf.cast(x[:, :, 6], tf.int32))(input_shots)
    hit_area_emb = area_embed(hit_area)
    player_area_emb = area_embed(player_area)
    opp_area_emb = area_embed(opp_area)

    time_prop = input_shots[:, :, 3:4]
    theta = theta_embed(input_types)
    mu = mu_embed(input_types)
    delta = TemporalScore()([theta, mu, time_prop])
    type_embedded = type_embed(input_types)
    enhanced_type = layers.Multiply()([delta, type_embedded])

    other_feat = input_shots[:, :, :3]
    combined = layers.Concatenate(axis=-1)(
        [other_feat, hit_area_emb, player_area_emb, opp_area_emb, enhanced_type])

    cnn1 = layers.Conv1D(filters, kernel, padding="same", activation="relu")(combined)
    cnn2 = layers.Conv1D(filters, kernel, padding="same", activation="relu")(combined)
    staggered = StaggeredMerge(filters)([cnn1, cnn2])

    gru = layers.Bidirectional(
        layers.GRU(rnn_units, return_sequences=True,
                   kernel_regularizer=tf.keras.regularizers.l2(l2)))(staggered)
    attn_input = layers.Concatenate(axis=-1)([staggered, gru])
    attn_scores = layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(l2))(attn_input)
    attn_weights = layers.Softmax(axis=1)(attn_scores)   # original behaviour (see docstring)

    context_vec = WeightedSum()([attn_weights, attn_input])
    merged = layers.Concatenate()([context_vec, input_ctx])
    output = layers.Dense(1, activation="sigmoid",
                          kernel_regularizer=tf.keras.regularizers.l2(l2))(merged)
    return tf.keras.Model(inputs=[input_shots, input_types, input_ctx, input_len],
                          outputs=output)


def build_bilstm(num_types: int, embed_dim: int = 15, rnn_units: int = 32):
    """Micro BiLSTM (D:\\data\\微观模型\\BiLSTM.py lines 84-96)."""
    input_shots = tf.keras.Input(shape=(None, 7), name="shots")
    input_types = tf.keras.Input(shape=(None,), dtype=tf.int32, name="types")
    input_ctx = tf.keras.Input(shape=(2,), name="ctx")
    input_len = tf.keras.Input(shape=(), dtype=tf.int32, name="len")

    type_embed = layers.Embedding(num_types, embed_dim, mask_zero=True)(input_types)
    combined = layers.Concatenate(axis=-1)([input_shots, type_embed])
    lstm_out = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=False))(combined)
    merged = layers.Concatenate()([lstm_out, input_ctx])
    output = layers.Dense(1, activation="sigmoid")(merged)
    return tf.keras.Model(inputs=[input_shots, input_types, input_ctx, input_len],
                          outputs=output)


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len=200, d_model=32, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(seq_len)
        return x + self.pos_emb(positions)


def _transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    attn_out = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads,
                                         dropout=dropout)(inputs, inputs)
    attn_out = layers.Dropout(dropout)(attn_out)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_out)
    ffn = tf.keras.Sequential([layers.Dense(ff_dim, activation="relu"),
                               layers.Dense(inputs.shape[-1])])
    ffn_out = ffn(out1)
    ffn_out = layers.Dropout(dropout)(ffn_out)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_out)


def build_transformer(num_types: int, embed_dim: int = 32, num_heads: int = 4,
                      ff_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
    """Micro Transformer (D:\\data\\微观模型\\Transformer.py lines 74-115)."""
    input_shots = tf.keras.Input(shape=(None, 7), name="shots")
    input_types = tf.keras.Input(shape=(None,), dtype=tf.int32, name="types")
    input_ctx = tf.keras.Input(shape=(2,), name="ctx")
    input_len = tf.keras.Input(shape=(), dtype=tf.int32, name="len")

    type_embed = layers.Embedding(num_types, embed_dim, mask_zero=True)(input_types)
    shot_proj = layers.Dense(embed_dim)(input_shots)
    combined = shot_proj + type_embed
    combined = PositionalEncoding(max_len=200, d_model=embed_dim)(combined)
    x = combined
    head_size = embed_dim // num_heads
    for _ in range(num_layers):
        x = _transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    pooled = layers.GlobalAveragePooling1D()(x)
    merged = layers.Concatenate()([pooled, input_ctx])
    output = layers.Dense(1, activation="sigmoid")(merged)
    return tf.keras.Model(inputs=[input_shots, input_types, input_ctx, input_len],
                          outputs=output)


# ---- end-to-end (game-level) architectures -------------------------------
MACRO_DIM = 18


def build_e2e_gru(macro_dim: int | None = None):
    """端到端 GRU (D:\\data\\端到端实验\\train_GRU_model.py / _macro_model.py).

    The original passes ``dropout=0.5`` *and* ``recurrent_dropout=0.5`` to a GRU
    on CPU.  TF 2.15 implements this combination but emits a crash-prone
    fallback warning on some builds; ``V2_E2E_RECURRENT_DROPOUT=0`` disables it.
    """
    use_rd = env_flag("V2_E2E_RECURRENT_DROPOUT", True)
    rd = 0.5 if use_rd else 0.0
    seq_input = layers.Input(shape=(None, 7))
    if macro_dim is None:
        x = layers.GRU(16, dropout=0.5, recurrent_dropout=rd)(seq_input)
    else:
        macro_input = layers.Input(shape=(macro_dim,))
        g = layers.GRU(16, dropout=0.5, recurrent_dropout=rd)(seq_input)
        x = layers.Concatenate()([g, macro_input])
        x = layers.Dense(16, activation="relu",
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(1, activation="sigmoid")(x)
        return tf.keras.models.Model([seq_input, macro_input], output)
    x = layers.Dense(16, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.models.Model(seq_input, output)


def build_e2e_lstm(macro_dim: int | None = None):
    """端到端 LSTM.

    pure   : train_LSTM_model.py lines 57-63
    +macro : train_LSTM_macro_model.py lines 77-83 (LSTM -> concat macro -> Dense)
    """
    use_rd = env_flag("V2_E2E_RECURRENT_DROPOUT", True)
    rd = 0.5 if use_rd else 0.0
    seq_input = layers.Input(shape=(None, 7))
    if macro_dim is not None:
        macro_input = layers.Input(shape=(macro_dim,))
        g = layers.LSTM(16, dropout=0.5, recurrent_dropout=rd)(seq_input)
        x = layers.Concatenate()([g, macro_input])
        x = layers.Dense(16, activation="relu",
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(1, activation="sigmoid")(x)
        return tf.keras.models.Model([seq_input, macro_input], output)
    x = layers.LSTM(16, dropout=0.5, recurrent_dropout=rd)(seq_input)
    x = layers.Dense(16, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.models.Model(seq_input, output)


def _e2e_transformer_encoder(inputs, num_heads=2, ff_dim=32, dropout=0.3):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads,
                                            key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = layers.Dropout(dropout)(attn_output)
    out1 = layers.LayerNormalization()(inputs + attn_output)
    ffn = layers.Dense(ff_dim, activation="relu")(out1)
    ffn = layers.Dense(inputs.shape[-1])(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    return layers.LayerNormalization()(out1 + ffn)


def build_e2e_transformer(macro_dim: int | None = None):
    """端到端 Transformer.

    pure   : train_Transformer_model.py lines 57-81
    +macro : train_Transformer_macro_model.py lines 87-102
             (project -> 2 encoders -> GAP -> concat macro -> Dense)
    """
    seq_input = layers.Input(shape=(None, 7), name="seq")
    x = layers.Dense(32)(seq_input)
    for _ in range(2):
        x = _e2e_transformer_encoder(x)
    x = layers.GlobalAveragePooling1D()(x)
    if macro_dim is not None:
        macro_input = layers.Input(shape=(macro_dim,), name="macro")
        x = layers.Concatenate()([x, macro_input])
        x = layers.Dense(16, activation="relu",
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(1, activation="sigmoid")(x)
        return tf.keras.models.Model([seq_input, macro_input], output)
    x = layers.Dense(16, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.models.Model(seq_input, output)


# ============================================================================
# 6. training + evaluation helpers
# ============================================================================
def compile_model(model, lr: float = 1e-3):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


def _hist(history, key: str):
    h = history.history
    if key in h:
        return [float(v) for v in h[key]]
    for alt in (key.upper(), key.lower()):
        if alt in h:
            return [float(v) for v in h[alt]]
    return []


def train_with_early_stopping(model, train_ds, val_ds, epochs: int, patience: int = 15,
                              verbose: int = 2, steps_per_epoch: int | None = None,
                              validation_steps: int | None = None):
    """val_auc EarlyStopping, restore_best_weights.  Test is never touched."""
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=patience,
        restore_best_weights=True, verbose=1)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        verbose=verbose, callbacks=[es])
    train_auc = _hist(history, "auc")
    val_auc = _hist(history, "val_auc")
    epochs_run = len(train_auc)
    if val_auc:
        best_epoch = int(np.argmax(val_auc)) + 1          # 1-based
        val_auc_best = float(np.max(val_auc))
    else:
        best_epoch, val_auc_best = epochs_run, float("nan")
    return {
        "history": history,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "train_auc_last": float(train_auc[-1]) if train_auc else float("nan"),
        "val_auc_best": val_auc_best,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
    }


def predict_micro(model, seqs, batch_size: int = 32) -> np.ndarray:
    ds = make_micro_dataset(seqs, batch_size=batch_size, shuffle=False)
    return model.predict(ds, verbose=0).ravel()


def predict_e2e(model, seqs, batch_size: int = 8, macro: Dict[int, np.ndarray] | None = None):
    X = padded_matrix(seqs)
    if macro is None:
        return model.predict(X, batch_size=batch_size, verbose=0).ravel()
    M = np.stack([macro[s["game_id"]] for s in seqs]).astype(np.float32)
    return model.predict([X, M], batch_size=batch_size, verbose=0).ravel()


def safe_auc(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).ravel()
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


# ============================================================================
# 7. result persistence
# ============================================================================
def slug(*parts) -> str:
    return "__".join(str(p).replace(" ", "_").replace("/", "-") for p in parts if p)


def save_record(record: Dict, tag: str | None = None) -> str:
    ensure_dirs()
    name = slug(tag, record.get("condition"), "run%d" % record.get("run", 0)) + ".json"
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def save_curves(fit: Dict, condition: str, run: int, tag: str | None = None) -> str:
    """Curve file name carries the condition suffix (fixes the shared
    ``auc_curves.npz`` overwrite between CNN.py and CNN_no_location.py)."""
    ensure_dirs()
    name = slug(tag, condition, "run%d" % run) + ".npz"
    path = os.path.join(CURVES_DIR, name)
    np.savez(path,
             train_auc=np.asarray(fit["train_auc"], dtype=np.float32),
             val_auc=np.asarray(fit["val_auc"], dtype=np.float32),
             condition=np.array(condition))
    return path


def base_record(condition: str, run: int, seed: int, epochs: int,
                data_path: str, extra: Dict | None = None) -> Dict:
    rec = {
        "condition": condition,
        "run": run,
        "seed": seed,
        "epochs_requested": epochs,
        "data_path": data_path,
        "data_sha256_16": sha256_file(data_path) if os.path.exists(data_path) else None,
        "framework": {"tensorflow": tf.__version__, "numpy": np.__version__,
                      "pandas": pd.__version__},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        rec.update(extra)
    return rec


class Timer:
    def __init__(self):
        self.t0 = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0
