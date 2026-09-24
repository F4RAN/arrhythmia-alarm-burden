import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

PRIM="#E8EEF5"; NEUT="#F2F2F2"; HIGH="#F6EFDF"
LW=1.3; FS=5.5

def box(ax,x,y,w,h,t,fc):
    ax.add_patch(Rectangle((x,y),w,h,linewidth=LW,edgecolor='black',
                           facecolor=fc,zorder=2))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=FS,
            fontweight='bold',zorder=3,linespacing=1.4)

def arrow(ax,p0,p1):
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle='-|>',mutation_scale=7,
        linewidth=LW,color='black',shrinkA=0,shrinkB=0,zorder=4))

fig,ax=plt.subplots(figsize=(3.5,1.30))
W,H,G=1.18,0.40,0.13
ys=0.62
items=[("Per-beat\nmetrics",NEUT),("Prevalence\ncorrection",PRIM),
       ("Episode rule\n$k$ in a row",PRIM),("Refractory\nwindow",PRIM),
       ("Alarms\nper 24 h",HIGH)]
xs=[];x=0.0
for t,fc in items:
    box(ax,x,ys,W,H,t,fc); xs.append(x); x+=W+G
for i in range(4):
    arrow(ax,(xs[i]+W,ys+H/2),(xs[i+1],ys+H/2))

bx=xs[2]
box(ax,bx,0.04,W,H,"Episode\nsensitivity",NEUT)
arrow(ax,(bx+W/2,ys),(bx+W/2,0.04+H))
ax.plot([bx+W,xs[4]+W/2],[0.04+H/2,0.04+H/2],color='black',lw=LW,zorder=1)
arrow(ax,(xs[4]+W/2,0.04+H/2),(xs[4]+W/2,ys))

ax.add_patch(Rectangle((xs[1]-0.09,ys-0.09),(xs[3]+W)-xs[1]+0.18,H+0.18,
    fill=False,linewidth=1.0,edgecolor='black',linestyle=(0,(4,2.5)),zorder=1))
ax.text((xs[1]+xs[3]+W)/2,ys+H+0.13,"Conversion",ha='center',va='bottom',
        fontsize=FS+0.6,fontweight='bold')

ax.set_xlim(-0.12,xs[4]+W+0.12); ax.set_ylim(-0.02,1.26)
ax.axis('off')
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'fig_pipeline.pdf'))
print('fig_pipeline ok')
