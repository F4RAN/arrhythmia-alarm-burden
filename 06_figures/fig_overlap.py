import sys,os; sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from style import *
import matplotlib.pyplot as plt, numpy as np
pairs=['RF\nET','MLP\nRF','MLP\nET','MLP\nLR','RF\nLR','ET\nLR']
jac=[0.474,0.107,0.126,0.059,0.047,0.035]
fig,ax=plt.subplots(figsize=(3.5,2.05))
b=ax.bar(np.arange(6),jac,color=[GOLD]+[NAVY]*5,width=0.6)
ax.set_xticks(np.arange(6)); ax.set_xticklabels(pairs,fontsize=6.5)
ax.set_ylabel('false-positive overlap\n(Jaccard)')
ax.set_ylim(0,0.58); ax.grid(True,axis='y')
ax.annotate('same model family',(0,0.474),textcoords='offset points',xytext=(2,7),fontsize=6.5,color=GOLD)
for r,v in zip(b,jac):
    if v<0.2: ax.text(r.get_x()+r.get_width()/2,v+0.012,f'{v:.3f}',ha='center',fontsize=5.8,color=GREY)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'fig_overlap.pdf'))
print('fig_overlap ok')
