"""THE decisive test for paper 2:
   at MATCHED episode sensitivity, does the voting module beat plain timing logic?"""
import pickle,numpy as np,torch,torch.nn as nn,json,warnings,sys; warnings.filterwarnings('ignore')
sys.path.insert(0,'code')
from papers import BusiaTinyTransformer, FaragSTFTCNN, ArrythMLAutoencoder
FS=360.; HOURS=11.03
T=json.load(open('experiments/fixed_thr.json'))
bu=BusiaTinyTransformer(d=20,heads=4); bu.load_state_dict(torch.load('code/models/f_busia.pt')); bu.eval()
fa=FaragSTFTCNN(ch=14); fa.load_state_dict(torch.load('code/models/f_farag.pt')); fa.eval()
ae=ArrythMLAutoencoder()
ae.enc=nn.Sequential(nn.Linear(180,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,32))
ae.dec=nn.Sequential(nn.Linear(32,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,180))
ae.load_state_dict(torch.load('code/models/f_ae.pt')); ae.eval()

d=pickle.load(open('data/mitbih/ds2.pkl','rb'))
X=torch.tensor(d['X']); Y=d['Y']; R=d['REC']; P=d['POS']; t=P/FS
pre=np.zeros(len(P),np.float32); post=np.zeros(len(P),np.float32)
for rc in np.unique(R):
    m=np.where(R==rc)[0]; q=P[m]/FS
    a=np.diff(q,prepend=q[0]); b=np.diff(q,append=q[-1]); mu=np.median(a[a>0])
    pre[m]=np.clip(a/mu,0,3); post[m]=np.clip(b/mu,0,3)
Rt=torch.tensor(np.stack([pre,post],1))
def sc(m,isae=False,stft=False):
    o=[]
    with torch.no_grad():
        for i in range(0,len(X),4096):
            if isae: o.append(m.score(X[i:i+4096]))
            elif stft: o.append(1-torch.softmax(m(X[i:i+4096]),1)[:,0])
            else: o.append(1-torch.softmax(m(X[i:i+4096],Rt[i:i+4096]),1)[:,0])
    return torch.cat(o).numpy()
sB,sF,sA = sc(bu), sc(fa,stft=True), sc(ae,isae=True)
fB=(sB>T['Busia']['thr']).astype(int); fF=(sF>T['Farag']['thr']).astype(int); fA=(sA>T['ArrythML']['thr']).astype(int)
trueV=(Y=='V').astype(int)

def runs(a,k,gap=1.2):
    ev=[]
    for r in np.unique(R):
        m=np.where(R==r)[0]; f=a[m]; tt=t[m]; n=len(f); i=0
        while i<n:
            if f[i]:
                j=i
                while j+1<n and f[j+1] and (tt[j+1]-tt[j])<=gap: j+=1
                if j-i+1>=k: ev.append((r,tt[i]))
                i=j+1
            else: i+=1
    return ev
TV={k:runs(trueV,k) for k in [1,2,3,4]}
def score(flag,k):
    ev=runs(flag.astype(int),k); tv=TV[k]
    # sensitivity: true episodes that have a matching alarm
    det=sum(1 for r,x in tv if any(r2==r and abs(x2-x)<10 for r2,x2 in ev))
    # false alarms: alarms that match NO true episode (each alarm counted once)
    fp=sum(1 for r,x in ev if not any(r2==r and abs(x2-x)<10 for r2,x2 in tv))
    return fp*24/HOURS, det/max(len(tv),1)*100, len(tv)

print("="*72)
print("PAPER 2 DECISIVE TEST -- Busia as main model, MIT-BIH DS2")
print("Both mechanisms only SUPPRESS. Fair question: at equal sensitivity,")
print("which gives fewer false alarms?")
print("="*72)
print(f"{'mechanism':<26}{'k':>3}{'falseAl/day':>13}{'episode Se':>12}{'trueEps':>9}")
print("-"*72)
rows=[]
for k in [1,2,3,4]:
    fp,se,n=score(fB,k); rows.append(('timing only',k,fp,se)); 
    print(f"{'timing rule only':<26}{k:>3}{fp:>13.1f}{se:>11.0f}%{n:>9}")
print()
for k in [1,2]:
    for lab,f in [('+1 other agrees',(fB&(fF|fA))),('+both agree',(fB&fF&fA))]:
        fp,se,n=score(f,k); rows.append((lab,k,fp,se))
        print(f"{'module: '+lab:<26}{k:>3}{fp:>13.1f}{se:>11.0f}%{n:>9}")
print()
print("="*72); print("MATCHED-SENSITIVITY COMPARISON"); print("="*72)
for target in [70,50,30]:
    best={}
    for lab,k,fp,se in rows:
        fam='module' if lab.startswith('+') else 'timing'
        if se>=target and (fam not in best or fp<best[fam][0]): best[fam]=(fp,lab,k,se)
    if len(best)==2:
        tm,md=best['timing'],best['module']
        w='MODULE' if md[0]<tm[0] else 'TIMING'
        print(f"Se >= {target}%:  timing {tm[0]:.1f}/day ({tm[1]} k={tm[2]}, Se {tm[3]:.0f}%)"
              f"  vs  module {md[0]:.1f}/day ({md[1]} k={md[2]}, Se {md[3]:.0f}%)   -> {w} WINS")
    else:
        print(f"Se >= {target}%:  no matched pair available")
