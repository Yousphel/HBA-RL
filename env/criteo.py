"""
Criteo Display Advertising data -> workload attributes (revision 1)
===================================================================

IMPORTANT (issues A9, A10): what this mapping is and is not
------------------------------------------------------------
The Criteo labels are CLICKS (not conversions) and the integer features I1-I13
are ANONYMISED: their meaning is not public. The assignment below (I2 -> value,
I1 -> duration, ...) is therefore an ARBITRARY, FIXED mapping chosen only to
obtain heavy-tailed, correlated workload attributes from real logs. It makes no
claim that I2 "is" revenue or that I5 "is" interaction complexity.

Mapping used (formulas, clip ranges, missing-value rule), per row
-----------------------------------------------------------------
  label (click 0/1)    priority boost
  spend   = clip(|I2| / (|I2| + 1000), 0.05, 0.90)              (0.3 if I2 = 0)
  priority p = clip(spend + 0.15 * label, 0.05, 1.0)
  value  v = clip(|I2| * (1 + p), 50, 5000)                     (draw Exponential(500) if I2 = 0)
  penalty ratio = clip(|I3| / (|I3| + 500), 0.1, 0.4)           (0.2 if I3 = 0)
  penalty kappa = v * penalty ratio
  duration D = clip(|I1| / 20, 3, 20) steps                     (U(3, 15) if I1 = 0)
  GPU = clip(|I5| / 5, 0.5, 4.0)                                (U(0.5, 2.0) if I5 = 0)
  CPU = clip(|I6| / 2, 2.0, 16.0)                               (U(2, 8) if I6 = 0)
  memory = clip(|I7| / 4, 8.0, 64.0)                            (U(8, 32) if I7 = 0)
  bandwidth ~ U(1, 5)
  deadline tau = D * U(1.5, 2.5) if p > 0.6 else D * U(2.5, 4.0)
  model type = stable hash of C1 modulo 4 (stored, not used by the environment)

Missing-value rule (D4): a MISSING integer feature is replaced by a constant before the
formulas are applied: I1 = 10, I2 = 500, I3 = 100, I5 = 1, I6 = 4, I7 = 16. A value that
is EXACTLY ZERO triggers the random fallback shown in brackets above. Negative values
(I2 and others can be negative) are mapped through |x|. The random draws use the
episode seed, so they are reproducible. Because |I2| is small for most rows, many jobs sit at the lower clip
bound of 50; Table M3 (scripts/workload_stats.py) reports the share at each bound.

Episode sampling (revision 1 fix)
---------------------------------
Before this revision every episode replayed the SAME shuffled rows from index 0
(reset() re-shuffled with a fixed seed), so an episode saw only about 30 distinct
rows out of 160,000, and different methods saw different rows because the shuffle
seed was the training seed. Now each episode draws its rows without replacement
from the split using a generator seeded by the episode seed. Same episode seed =>
same rows for every method. Different episode seeds => different rows.

Data (D4)
---------
The experiments use a 200,000-row extract: the FIRST 200,000 rows of Criteo's
train.txt, produced by data/make_criteo_extract.py (SHA-256 recorded). Split by row
order: first 160,000 = train, last 40,000 = evaluation. Criteo's test.txt is not
used (its labels are withheld).

Reference: Criteo Display Advertising Challenge (2014),
https://www.kaggle.com/c/criteo-display-ad-challenge
"""

import random
import zlib
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

# ── Column definitions ────────────────────────────────────────────────
LABEL_COL = "label"
INT_COLS  = [f"I{i}" for i in range(1, 14)]   # I1-I13
CAT_COLS  = [f"C{i}" for i in range(1, 27)]   # C1-C26
ALL_COLS  = [LABEL_COL] + INT_COLS + CAT_COLS  # 40 columns

# C1 bucket -> model type. Arbitrary fixed assignment (C1 is anonymised). Not used by the environment.
MODEL_TYPE_MAP = {0: 'Transformer', 1: 'CNN', 2: 'RNN', 3: 'LinearModel'}

# Clip bounds, exposed so scripts/workload_stats.py can report the share of rows at each bound
CLIP = {
    'value':    (50.0, 5000.0),
    'priority': (0.05, 1.0),
    'duration': (3.0, 20.0),
    'gpu':      (0.5, 4.0),
    'cpu':      (2.0, 16.0),
    'memory':   (8.0, 64.0),
    'penalty_ratio': (0.1, 0.4),
}

# Rows used per episode: an episode has at most episode_length arrivals
ROWS_PER_EPISODE = 200


# ==================== CriteoLoader ====================

class CriteoLoader:
    """
    Loads the first `n_rows` of train.txt and splits them by row order.

    Parameters
    ----------
    criteo_folder : folder containing train.txt (the 200,000-row extract)
    split         : 'train' (first train_ratio) or 'eval' (the remaining rows)
    train_ratio   : 0.8
    max_rows      : safety cap when reading chunks
    """

    def __init__(self, criteo_folder: str, split: str = 'train', train_ratio: float = 0.8,
                 chunk_size: int = 50_000, max_rows: Optional[int] = 500_000, seed: int = 42):
        if split not in ('train', 'eval'):
            raise ValueError(f"split must be 'train' or 'eval', got '{split}'")
        folder = Path(criteo_folder).resolve()
        for candidate in [folder / 'train.txt', folder / 'train']:
            if candidate.exists():
                self._train_path = str(candidate)
                break
        else:
            raise FileNotFoundError(
                f"\nCriteo train file not found in: {folder}\n"
                f"Expected {folder}/train.txt (run data/make_criteo_extract.py to create it).")

        self.criteo_folder = str(folder)
        self.split         = split
        self.train_ratio   = train_ratio
        self.chunk_size    = chunk_size
        self.max_rows      = max_rows
        self.seed          = seed          # kept for API compatibility; no longer used for shuffling
        self._cache        = None
        print(f"[CriteoLoader] split='{split}' train_ratio={train_ratio} file={self._train_path}")

    # ------------------------------------------------------------------
    def _iter_chunks(self) -> Iterator[pd.DataFrame]:
        rows_read = 0
        reader = pd.read_csv(self._train_path, sep='\t', header=None, names=ALL_COLS,
                             dtype=str, chunksize=self.chunk_size)
        for chunk in reader:
            chunk[INT_COLS]  = chunk[INT_COLS].apply(pd.to_numeric, errors='coerce')
            chunk[LABEL_COL] = pd.to_numeric(chunk[LABEL_COL], errors='coerce')
            yield chunk
            rows_read += len(chunk)
            if self.max_rows and rows_read >= self.max_rows:
                break

    # ------------------------------------------------------------------
    def load_sample(self, n_rows: int = 200_000) -> pd.DataFrame:
        """Return the requested split of the first n_rows rows (cached)."""
        if self._cache is not None:
            return self._cache
        frames, collected = [], 0
        for chunk in self._iter_chunks():
            frames.append(chunk)
            collected += len(chunk)
            if collected >= n_rows:
                break
        full_df   = pd.concat(frames, ignore_index=True).iloc[:n_rows]
        split_idx = int(len(full_df) * self.train_ratio)
        train_df  = full_df.iloc[:split_idx].reset_index(drop=True)
        eval_df   = full_df.iloc[split_idx:].reset_index(drop=True)
        df = train_df if self.split == 'train' else eval_df
        self._cache = df
        self.rows_total = len(full_df)
        print(f"[CriteoLoader] rows read: {len(full_df):,} | train {len(train_df):,} | eval {len(eval_df):,} | "
              f"using '{self.split}' ({len(df):,} rows) | click rate {df[LABEL_COL].mean():.4f} | "
              f"missing I1 {df['I1'].isna().mean():.1%}")
        return df

    # ------------------------------------------------------------------
    def get_statistics(self) -> dict:
        df = self.load_sample()
        return {
            'split'        : self.split,
            'total_rows'   : len(df),
            'click_rate'   : float(df[LABEL_COL].mean()),
            'missing_rates': {c: float(df[c].isna().mean()) for c in INT_COLS},
            'zero_rates'   : {c: float((df[c] == 0).mean()) for c in INT_COLS},
        }


# ==================== CriteoWorkloadMapper ====================

class CriteoWorkloadMapper:
    """Turns Criteo rows into job attributes using the arbitrary fixed mapping in the module docstring."""

    def __init__(self, loader: CriteoLoader):
        self.loader = loader
        self._df    = None
        self._order = None
        self._idx   = 0
        self._DURATION_SCALE = 20.0
        self._GPU_SCALE      = 5.0
        self._CPU_SCALE      = 2.0
        self._MEM_SCALE      = 4.0

    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self._df is None:
            self._df = self.loader.load_sample()
        if self._order is None:
            self.reset()

    @staticmethod
    def _safe(val, default=0.0) -> float:
        try:
            v = float(val)
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    def reset(self, episode_seed: Optional[int] = None):
        """Start a new episode: draw this episode's rows from the split with the episode seed."""
        if self._df is None:
            self._df = self.loader.load_sample()
        rng = np.random.default_rng(episode_seed)   # None => fresh OS entropy
        n = len(self._df)
        self._order = rng.choice(n, size=min(n, ROWS_PER_EPISODE), replace=False)
        self._idx = 0

    # ------------------------------------------------------------------
    def next_workload_params(self, current_step: int = 0) -> dict:
        self._ensure_loaded()
        row = self._df.iloc[int(self._order[self._idx % len(self._order)])]
        self._idx += 1

        label = int(self._safe(row[LABEL_COL], 0))
        # Missing value rule: constant defaults I1=10, I2=500, I3=100, I5=1, I6=4, I7=16.
        # An exact zero is treated differently: a seeded random draw (see docstring).
        i1 = self._safe(row['I1'], 10.0)
        i2 = self._safe(row['I2'], 500.0)
        i3 = self._safe(row['I3'], 100.0)
        i5 = self._safe(row['I5'], 1.0)
        i6 = self._safe(row['I6'], 4.0)
        i7 = self._safe(row['I7'], 16.0)
        c1 = str(row.get('C1', '00000000'))

        duration = float(np.clip(abs(i1) / self._DURATION_SCALE, *CLIP['duration'])) \
            if i1 != 0 else random.uniform(3, 15)

        spend_signal = float(np.clip(abs(i2) / (abs(i2) + 1000.0), 0.05, 0.90)) if i2 != 0 else 0.3
        priority = float(np.clip(spend_signal + 0.15 * label, *CLIP['priority']))

        revenue = float(np.clip(abs(i2) * (1.0 + priority), *CLIP['value'])) \
            if i2 != 0 else random.expovariate(1 / 500)

        penalty_ratio = float(np.clip(abs(i3) / (abs(i3) + 500.0), *CLIP['penalty_ratio'])) if i3 != 0 else 0.2
        penalty = revenue * penalty_ratio

        gpu_req = float(np.clip(abs(i5) / self._GPU_SCALE, *CLIP['gpu'])) if i5 != 0 else random.uniform(0.5, 2.0)
        cpu_req = float(np.clip(abs(i6) / self._CPU_SCALE, *CLIP['cpu'])) if i6 != 0 else random.uniform(2.0, 8.0)
        mem_req = float(np.clip(abs(i7) / self._MEM_SCALE, *CLIP['memory'])) if i7 != 0 else random.uniform(8.0, 32.0)

        # Stable across processes (Python's built-in hash() of a str is randomised per process)
        model_type = MODEL_TYPE_MAP[zlib.crc32(c1.encode()) % 4]

        slack = random.uniform(1.5, 2.5) if priority > 0.6 else random.uniform(2.5, 4.0)
        deadline = duration * slack

        return dict(
            workload_id           = f"criteo_{self._idx}",
            model_type            = model_type,
            priority              = priority,
            deadline              = deadline,
            original_deadline     = deadline,
            resource_requirements = {
                'cpu': cpu_req, 'gpu': gpu_req, 'memory': mem_req,
                'bandwidth': random.uniform(1.0, 5.0),
            },
            estimated_duration = duration,
            revenue_value      = revenue,
            penalty_cost       = penalty,
            arrival_time       = current_step,
        )


# ==================== Environment Factory ====================

def make_criteo_environment(criteo_folder: str,
                            split:          str   = 'train',
                            train_ratio:    float = 0.8,
                            num_nodes:      int   = 5,
                            max_workloads:  int   = 20,
                            episode_length: int   = 100,
                            max_rows:       int   = 500_000,
                            seed:           int   = 42,
                            arrival_prob:   float = 0.3,
                            observe_all_nodes: bool = True):
    """
    Build an MLSchedulingEnvironment whose jobs come from Criteo rows.

    `seed` is kept for API compatibility. Rows are now chosen per episode from the
    episode seed passed to env.reset(seed=...), so it no longer affects the data.
    """
    import types
    from hrl_scheduler_simulation import MLSchedulingEnvironment, MLWorkload

    loader = CriteoLoader(criteo_folder, split=split, train_ratio=train_ratio,
                          max_rows=max_rows, seed=seed)
    mapper = CriteoWorkloadMapper(loader)
    loader.load_sample()

    env = MLSchedulingEnvironment(num_nodes=num_nodes, max_workloads=max_workloads,
                                  episode_length=episode_length, arrival_prob=arrival_prob,
                                  observe_all_nodes=observe_all_nodes)

    def _criteo_generate_workload(self_env):
        return MLWorkload(**mapper.next_workload_params(self_env.current_step))
    env._generate_workload = types.MethodType(_criteo_generate_workload, env)

    _orig_reset = env.reset
    def _criteo_reset(self_env, seed=None, options=None):
        mapper.reset(seed)                       # rows depend on the episode seed only
        return _orig_reset(seed=seed, options=options)
    env.reset = types.MethodType(_criteo_reset, env)

    env.criteo_loader = loader
    env.criteo_mapper = mapper
    print(f"[make_criteo_environment] split='{split}' nodes={num_nodes} episode_length={episode_length} "
          f"arrival_prob={arrival_prob}")
    return env, loader, mapper


if __name__ == '__main__':
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else './criteo'
    for split in ('train', 'eval'):
        loader = CriteoLoader(folder, split=split)
        stats = loader.get_statistics()
        print(f"{split}: rows={stats['total_rows']:,} click rate={stats['click_rate']:.4f}")
        print("  missing rate  :", {k: round(v, 3) for k, v in list(stats['missing_rates'].items())[:7]})
        print("  zero rate     :", {k: round(v, 3) for k, v in list(stats['zero_rates'].items())[:7]})
