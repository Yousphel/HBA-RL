"""Fig. 1 (A14): architecture redrawn from the code. Vector PDF.
Matches: 50-dim state (queue summary + all five nodes), macro outputs 12-dim subgoal and a 5-way routing choice,
routing happens BEFORE job selection, micro of the routed node picks the queue index, nodes numbered 0-4
(0-1 high-performance, 2-4 standard), no tenants, no pending-job business value in the macro input.
    python scripts/make_fig1.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.set_xlim(0, 100); ax.set_ylim(-1, 64); ax.axis('off')


def box(x, y, w, h, text, fc='#ffffff', ec='#333333', bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontweight='bold' if bold else 'normal')


def arrow(x1, y1, x2, y2, text=None, dy=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', lw=1.1, color='#333333'))
    if text:
        ax.text((x1 + x2) / 2 + 0.6, (y1 + y2) / 2 + dy, text, fontsize=7, ha='left')


box(1, 49, 31, 12, 'Cluster state s (50-dim)\nqueue summary and all five\nnodes: CPU, GPU, memory use,\ncost', '#eef3f8')
box(1, 27, 25, 9, 'Global job queue\n(up to 20 pending jobs)', '#f4f4f4')
box(38, 46, 32, 15, 'Macro-controller (PPO)\n1  subgoal g (12-dim)\n2  routing node m in {0,...,4}', '#d9e6f2', bold=False)
arrow(32.5, 55, 37.7, 55)
arrow(54, 45.7, 54, 40.3, 'routing first', dy=0.0)
box(38, 24, 32, 16, 'Micro-controller of node m\n(one PPO policy per node)\ninput: s and g (62-dim)\n1  queue index of the job\n2  allocation fraction', '#e3efdb')
arrow(26.3, 31.5, 37.7, 31.5)
ax.text(32, 33, 'job picked\nafter routing', fontsize=7, ha='center')
ax.text(72, 59, 'g is passed to the micro-controller\nas a vector (probabilities);\nit is not executed as an action', fontsize=7, va='top')

names = ['Node 0', 'Node 1', 'Node 2', 'Node 3', 'Node 4']
for i, n in enumerate(names):
    hp = i < 2
    box(3 + i * 19, 4.5, 17, 9, f'{n}\n{"high-performance" if hp else "standard"}', '#fbe5d6' if hp else '#fff2cc')
arrow(54, 23.7, 54, 20)
ax.plot([13, 89], [20, 20], color='#333333', lw=1)
for i in range(5):
    arrow(12.5 + i * 19, 20, 12.5 + i * 19, 14.2)
ax.text(56, 21.0, 'job runs on node m', fontsize=7, ha='left')
ax.text(99, 0.3, 'reward r_t and SLA events feed both levels', fontsize=7, ha='right', style='italic')
os.makedirs(os.path.join('results', 'figures'), exist_ok=True)
plt.savefig(os.path.join('results', 'figures', 'fig1_architecture.pdf'), bbox_inches='tight')
print('wrote results/figures/fig1_architecture.pdf')
