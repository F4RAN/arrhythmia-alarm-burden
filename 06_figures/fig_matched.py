import sys,os; sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from style import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
tim=[(41.3,90,'1'),(6.5,43,'2'),(2.2,16,'3'),(2.2,0,'4')]
epi=[(0.0,64,'1'),(4.4,26,'2')]
pb =[(0.0,64,'1'),(0.0,7,'2')]
fig,ax=plt.subplots(figsize=(3.5,2.45))
ax.plot([x for x,_,_ in tim],[y for _,y,_ in tim],marker='o',color=GREY,ls='--',label='temporal gating')
ax.plot([x for x,_,_ in epi],[y for _,y,_ in epi],marker='D',color=GOLD,ls='-',label='episode-gated module')
ax.plot([x for x,_,_ in pb],[y for _,y,_ in pb],marker='v',color=NAVY,ls=':',label='per-beat activation')
for x,y,l in tim:
    ax.annotate(r'$k$=%s'%l,(x,y),textcoords='offset points',xytext=(6,4),fontsize=6.5,color=GREY)
ax.annotate(r'$k$=1',(0.0,64),textcoords='offset points',xytext=(7,5),fontsize=6.5,color='k')
ax.annotate(r'$k$=2',(4.4,26),textcoords='offset points',xytext=(5,6),fontsize=6.5,color=GOLD)
ax.annotate(r'$k$=2',(0.0,7),textcoords='offset points',xytext=(7,-2),fontsize=6.5,color=NAVY)
ax.set_xscale('symlog',linthresh=1.0,linscale=0.45)
ax.set_xlim(-0.25,90); ax.set_ylim(-8,100)
ax.xaxis.set_major_locator(FixedLocator([0,1,10,100]))
ax.xaxis.set_major_formatter(FixedFormatter(['0','1','10','100']))
ax.set_xlabel('false alarms per 24 h  (DS2)')
ax.set_ylabel(r'episode sensitivity (\%)')
ax.grid(True,axis='y'); ax.legend(loc='lower right',handlelength=2.2)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'fig_matched.pdf'))
print('fig_matched ok')
