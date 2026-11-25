"""
equivalence.py - equivalence test (TOST) and Holm correction (elevation additions).

  TOST  two one-sided Welch t-tests on seed means (Schuirmann, 1987; Lakens, 2017).
        The variant is equivalent to the reference when the 90% Welch CI of
        (variant mean - reference mean) lies inside -margin .. +margin.
  Holm  step-down adjustment of a family of p-values (Holm, 1979).

The margin is set HERE, in one place. It is a percentage of the reference (HBA-RL) mean profit.
State in the paper that the margin was set after the main results were seen (post hoc).
"""
import numpy as np
from scipy import stats

EQ_MARGIN_PCT = 2.0      # +/- 2% of HBA-RL mean profit; change only here


def tost_from_summary(mean_v, sd_v, n_v, mean_r, sd_r, n_r, margin):
    """TOST from summary statistics: variant (v) minus reference (r), Welch standard error and df."""
    va, vb = sd_v ** 2 / n_v, sd_r ** 2 / n_r
    se = np.sqrt(va + vb)
    diff = mean_v - mean_r
    df = (va + vb) ** 2 / (va ** 2 / (n_v - 1) + vb ** 2 / (n_r - 1))
    half = stats.t.ppf(0.95, df) * se
    p_low = stats.t.sf((diff + margin) / se, df)     # H0: diff <= -margin
    p_high = stats.t.cdf((diff - margin) / se, df)   # H0: diff >= +margin
    lo, hi = diff - half, diff + half
    return dict(diff=diff, se=se, df=df, ci90_low=lo, ci90_high=hi, margin=margin,
                p_tost=max(p_low, p_high), equivalent=bool(-margin < lo and hi < margin))


def tost(variant, reference, margin_pct=EQ_MARGIN_PCT):
    """TOST on two arrays of seed means. The margin is margin_pct % of the reference mean."""
    v, r = np.asarray(variant, float), np.asarray(reference, float)
    margin = abs(r.mean()) * margin_pct / 100
    out = tost_from_summary(v.mean(), v.std(ddof=1), len(v), r.mean(), r.std(ddof=1), len(r), margin)
    out["margin_pct"] = margin_pct
    return out


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order. NaN p-values are left as NaN."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    m = len(q)
    if m == 0:
        return out
    order = np.argsort(q)
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * q[order]))
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out
