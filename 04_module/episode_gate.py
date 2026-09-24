"""Compare WHEN the module is woken.
   per-beat  : module runs on every flagged beat        (what we built)
   episode   : module runs once per candidate alarm     (what the prof meant)
   Measures accuracy AND number of activations."""
import pickle,numpy as np,torch,torch.nn as nn,json,warnings,sys; warnings.filterwarnings('ignore')
sys.path.insert(0,'code')
from papers import BusiaTinyTransformer, FaragSTFTCNN, ArrythMLAutoencoder
FS=360.; T=json.load(open('experiments/fixed_thr.json'))
bu=BusiaTinyTransformer(d=20,heads=4); bu.load_state_dict(torch.load('code/models/f_busia.pt')); bu.eval()
fa=FaragSTFTCNN(ch=14); fa.load_state_dict(torch.load('code/models/f_farag.pt')); fa.eval()
ae=ArrythMLAutoencoder()
ae.enc=nn.Sequential(nn.Linear(180,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,32))
ae.dec=nn.Sequential(nn.Linear(32,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,180))
ae.load_state_dict(torch.load('code/models/f_ae.pt')); ae.eval()

def prep(d):
    X=torch.tensor(d['X']); P=d['POS'] if 'POS' in d else d['P']; R=d['REC']
    pre=np.zeros(len(P),np.float32); post=np.zeros(len(P),np.float32)
    for rc in np.unique(R):
        m=np.where(R==rc)[0]; q=P[m]/FS
        a=np.diff(q,prepend=q[0]); b=np.diff(q,append=q[-1]); mu=np.median(a[a>0])
        pre[m]=np.clip(a/mu,0,3); post[m]=np.clip(b/mu,0,3)
    return X, torch.tensor(np.stack([pre,post],1)), R, P/FS

d=pickle.load(open('data/mitbih/ds2.pkl','rb')); Y=d['Y']
X,Rt,R,t=prep(d); HOURS=11.03
def sc(m,isae=False,stft=False):
    o=[]
    with torch.no_grad():
        for i in range(0,len(X),4096):
            if isae: o.append(m.score(X[i:i+4096]))
            elif stft: o.append(1-torch.softmax(m(X[i:i+4096]),1)[:,0])
            else: o.append(1-torch.softmax(m(X[i:i+4096],Rt[i:i+4096]),1)[:,0])
    return torch.cat(o).numpy()
fB=(sc(bu)>T['Busia']['thr']).astype(int)
fF=(sc(fa,stft=True)>T['Farag']['thr']).astype(int)
fA=(sc(ae,isae=True)>T['ArrythML']['thr']).astype(int)
trueV=(Y=='V').astype(int)

def cand(a,k,gap=1.2):
    """candidate episodes from the MAIN model only -> list of (rec, t0, idx array)"""
    out=[]
    for r in np.unique(R):
        m=np.where(R==r)[0]; f=a[m]; tt=t[m]; n=len(f); i=0
        while i<n:
            if f[i]:
                j=i
                while j+1<n and f[j+1] and (tt[j+1]-tt[j])<=gap: j+=1
                if j-i+1>=k: out.append((r,tt[i],m[i:j+1]))
                i=j+1
            else: i+=1
    return out
def truth(k): return [(r,t0) for r,t0,_ in cand(trueV,k)]
def sens_fa(ev,tv):
    det=sum(1 for r,x in tv if any(r2==r and abs(x2-x)<10 for r2,x2 in ev))
    fp=sum(1 for r,x in ev if not any(r2==r and abs(x2-x)<10 for r2,x2 in tv))
    return fp*24/HOURS, det/max(len(tv),1)*100

print("="*74)
print("WHEN TO WAKE THE MODULE -- Busia main, DS2")
print("="*74)
print(f"{'strategy':<30}{'k':>3}{'falseAl/d':>11}{'Se':>7}{'module runs':>13}")
print("-"*74)
for k in [1,2]:
    tv=truth(k)
    # A) per-beat AND, then episode formed from the survivors  (what we built)
    per=(fB&(fF|fA)).astype(int)
    ev=[(r,t0) for r,t0,_ in cand(per,k)]
    fp,se=sens_fa(ev,tv)
    print(f"{'per-beat (current build)':<30}{k:>3}{fp:>11.1f}{se:>6.0f}%{int(fB.sum()):>13,}")
    # B) episode gate: candidate from MAIN only, module confirms the episode
    cs=cand(fB,k); runs_=0; ev=[]
    for r,t0,idx in cs:
        runs_+=1                                  # ONE module activation per candidate
        agree=((fF[idx]|fA[idx]).mean()>=0.5)     # majority of beats in the episode
        if agree: ev.append((r,t0))
    fp,se=sens_fa(ev,tv)
    print(f"{'episode gate (prof design)':<30}{k:>3}{fp:>11.1f}{se:>6.0f}%{runs_:>13,}")
    # C) episode gate, Farag only (drop the 163 kB ArrythML)
    runs_=0; ev=[]
    for r,t0,idx in cs:
        runs_+=1
        if fF[idx].mean()>=0.5: ev.append((r,t0))
    fp,se=sens_fa(ev,tv)
    print(f"{'episode gate, Farag only':<30}{k:>3}{fp:>11.1f}{se:>6.0f}%{runs_:>13,}")
    print()
print("module runs = number of times the side module is activated over 11 h")
