import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, NullFormatter

pts = [(41.3, 90, '1'), (6.5, 43, '2'), (2.2, 16, '3'), (2.2, 0, '4')]

fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.plot([x for x, _, _ in pts], [y for _, y, _ in pts],
        marker='o', color=NAVY, zorder=3)

off = {'1': (-26, -3), '2': (9, -12), '3': (-5, 10), '4': (11, -3)}
for x, y, l in pts:
    ax.annotate(r'$k$=%s' % l, (x, y), textcoords='offset points',
                xytext=off[l], fontsize=7)

ax.scatter([2.2], [0], s=70, facecolor='none', edgecolor=GOLD, lw=1.5, zorder=4)
ax.annotate('silent detector:\n0.11 FA/24 h on healthy subjects',
            (2.2, 0), textcoords='offset points', xytext=(34, 14),
            fontsize=6.5, color=GOLD, ha='left',
            arrowprops=dict(arrowstyle='->', color=GOLD, lw=0.8,
                            shrinkA=0, shrinkB=4))

ax.set_xscale('log')
ax.set_xlim(1.6, 70)
ax.set_ylim(-12, 104)

# kill every automatic log tick, major and minor, then place our own
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50]))
ax.xaxis.set_major_formatter(FixedFormatter(['2', '5', '10', '20', '50']))
ax.yaxis.set_major_locator(FixedLocator([0, 20, 40, 60, 80, 100]))

ax.set_xlabel('false alarms per 24 h  (DS2)')
ax.set_ylabel('episode sensitivity (%)')
ax.grid(True, axis='y')
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'fig_pair.pdf'))
print('fig_pair rebuilt')
