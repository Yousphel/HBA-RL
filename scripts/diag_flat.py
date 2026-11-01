"""Diagnostic: are Flat PPO checkpoints of different seeds really different policies?"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
for d in sorted(glob.glob('businessAware/flat_ppo/*/seed_*')):
    ck, sm = os.path.join(d, 'checkpoint_best.pt'), os.path.join(d, 'training_summary.json')
    if not os.path.exists(ck): continue
    c = torch.load(ck, map_location='cpu', weights_only=False)
    s = json.load(open(sm)) if os.path.exists(sm) else {}
    w = c['policy']; k = [n for n in w if 'action_mean.weight' in n][0]
    x = torch.randn(200, 50)
    print(f"{d}: best_ep={s.get('best_checkpoint_episode')} episodes_run={s.get('episodes_run')} "
          f"best_reward={c.get('best_reward'):.2f} |action_mean.w|={w[k].abs().mean():.4f} "
          f"wsum={sum(v.float().sum() for v in w.values()):.4f}")
