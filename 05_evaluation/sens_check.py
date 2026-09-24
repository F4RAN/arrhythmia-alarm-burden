"""Does the k=6 rule silence TRUE detections as well as false ones?"""
import pickle,numpy as np,torch,torch.nn as nn,json,warnings,sys; warnings.filterwarnings('ignore')
sys.path.insert(0,'code')
from papers import BusiaTinyTransformer
FS=360.
T=json.load(open('experiments/fixed_thr.json'))
bu=BusiaTinyTransformer(d=20,heads=4); bu.load_state_dict(torch.load('code/models/f_busia.pt')); bu.eval()
d2=pickle.load(open('data/mitbih/ds2.pkl','rb'))
X=torch.tensor(d2['X']); Y=d2['Y']; R=d2['REC']; P=d2['POS']
p=P/FS
pre=np.zeros(len(P),np.float32); post=np.zeros(len(P),np.float32)
for rc in np.unique(R):
    m=np.where(R==rc)[0]; q=P[m]/FS
    dd=np.diff(q,prepend=q[0]); d2_=np.diff(q,append=q[-1]); mu=np.median(dd[dd>0])
    pre[m]=np.clip(dd/mu,0,3); post[m]=np.clip(d2_/mu,0,3)
Rt=torch.tensor(np.stack([pre,post],1))
with torch.no_grad():
    sc=torch.cat([1-torch.softmax(bu(X[i:i+4096],Rt[i:i+4096]),1)[:,0] for i in range(0,len(X),4096)]).numpy()
flag=(sc>T['Busia']['thr']).astype(int)
trueV=(Y=='V').astype(int); trueAbn=(Y!='N').astype(int)

def runs(a,k,rec,t,gap=1.2):
    ev=[]
    for r in np.unique(rec):
        m=np.where(rec==r)[0]; f=a[m]; tt=t[m]; n=len(f); i=0
        while i<n:
            if f[i]:
                j=i
                while j+1<n and f[j+1] and (tt[j+1]-tt[j])<=gap: j+=1
                if j-i+1>=k: ev.append((r,tt[i],j-i+1))
                i=j+1
            else: i+=1
    return ev

print("MIT-BIH DS2 (arrhythmia PATIENTS, 11.0 h) -- Busia")
print(f"  beat-level: flags {flag.mean()*100:.2f}% | true abnormal {trueAbn.mean()*100:.2f}%")
print()
print(f"{'k':<4}{'TRUE V-runs':>13}{'Busia alarms':>14}{'detected':>10}{'episode Se':>12}")
print("-"*56)
for k in [1,2,3,4,5,6]:
    tv=runs(trueV,k,R,p); ev=runs(flag,k,R,p)
    det=sum(1 for r,t,_ in tv if any(r2==r and abs(t2-t)<10 for r2,t2,_ in ev))
    se=det/max(len(tv),1)*100
    print(f"{k:<4}{len(tv):>13}{len(ev):>14}{det:>10}{se:>11.0f}%")
print()
print("Same for ANY abnormal beat (V, S or F), which is what the model predicts:")
print(f"{'k':<4}{'TRUE runs':>13}{'Busia alarms':>14}{'detected':>10}{'episode Se':>12}")
print("-"*56)
for k in [1,2,3,4,5,6]:
    tv=runs(trueAbn,k,R,p); ev=runs(flag,k,R,p)
    det=sum(1 for r,t,_ in tv if any(r2==r and abs(t2-t)<10 for r2,t2,_ in ev))
    print(f"{k:<4}{len(tv):>13}{len(ev):>14}{det:>10}{det/max(len(tv),1)*100:>11.0f}%")
