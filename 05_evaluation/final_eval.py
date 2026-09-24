import pickle,numpy as np,torch,torch.nn as nn,json,warnings; warnings.filterwarnings('ignore')
from papers import BusiaTinyTransformer, FaragSTFTCNN, ArrythMLAutoencoder
FS=360.; KS=[1,3,5,6]
def P(*a): print(*a,flush=True)
T=json.load(open('fixed_thr.json'))
bu=BusiaTinyTransformer(d=20,heads=4); bu.load_state_dict(torch.load('f_busia.pt')); bu.eval()
fa=FaragSTFTCNN(ch=14); fa.load_state_dict(torch.load('f_farag.pt')); fa.eval()
ae=ArrythMLAutoencoder()
ae.enc=nn.Sequential(nn.Linear(180,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,32))
ae.dec=nn.Sequential(nn.Linear(32,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,180))
ae.load_state_dict(torch.load('f_ae.pt')); ae.eval()
N=['Busia','Farag','ArrythML']
def flags(X,Pos):
    Xt=torch.tensor(X); p=Pos/FS
    d=np.diff(p,prepend=p[0]); d2=np.diff(p,append=p[-1]); mu=np.median(d[d>0]) if (d>0).any() else 1.
    Rt=torch.tensor(np.stack([np.clip(d/mu,0,3),np.clip(d2/mu,0,3)],1).astype(np.float32))
    o=[]
    with torch.no_grad():
        for m,nm,isae in [(bu,'Busia',0),(fa,'Farag',0),(ae,'ArrythML',1)]:
            s=[]
            for i in range(0,len(Xt),4096):
                s.append(m.score(Xt[i:i+4096]) if isae else 1-torch.softmax(m(Xt[i:i+4096],Rt[i:i+4096]),1)[:,0])
            o.append((torch.cat(s).numpy()>T[nm]['thr']).astype(int))
    return np.stack(o)
def eps(f,t,k,g=1.2,r=300.):
    n=len(f);i=0;last=-1e9;c=0
    while i<n:
        if f[i]:
            j=i
            while j+1<n and f[j+1] and (t[j+1]-t[j])<=g: j+=1
            if j-i+1>=k and t[i]-last>=r: c+=1; last=t[i]
            i=j+1
        else: i+=1
    return c
nsr=pickle.load(open('data2/nsrdb.pkl','rb')); H=sum(v['dur'] for v in nsr.values())/3600.
PR={}; FR=np.zeros(3); TOT=0
for rc,v in nsr.items():
    f=flags(v['X'].astype(np.float32),v['P']); PR[rc]=(f,v['P']/FS)
    FR+=f.sum(1); TOT+=f.shape[1]
P("="*76); P(f"PAPER MODELS (calibrated) on NSRDB: {H:.1f} h, {TOT:,} healthy beats"); P("="*76)
for i,nm in enumerate(N): P(f"  {nm:<10} flags {FR[i]/TOT*100:5.2f}% of healthy beats")
P("")
P(f"{'model':<11}{'rule':<9}"+"".join(f"{'k='+str(k):>10}" for k in KS)); P("-"*76)
R={}
for i,nm in enumerate(N):
    R[nm]={}; oth=[j for j in range(3) if j!=i]
    for need,lab in [(0,'alone'),(1,'+1 agree'),(2,'+2 agree')]:
        R[nm][lab]={}; row=f"{nm:<11}{lab:<9}"
        for k in KS:
            tot=sum(eps((((f[i]==1)) if need==0 else ((f[i]==1)&(f[oth].sum(0)>=need))).astype(int),t,k)
                    for f,t in PR.values())
            r=tot*24./H; R[nm][lab][k]=round(r,3); row+=f"{r:>10.2f}"
        P(row)
    P("")
P("="*76); P("SUMMARY at k=6"); P("="*76)
P(f"{'model':<11}{'alone':>10}{'+1 agree':>11}{'+2 agree':>11}{'cut':>9}")
for nm in N:
    a=R[nm]['alone'][6]; v2=R[nm]['+2 agree'][6]
    P(f"{nm:<11}{a:>10.2f}{R[nm]['+1 agree'][6]:>11.2f}{v2:>11.2f}{(0 if a==0 else 100*(1-v2/a)):>8.0f}%")
P("\ntarget <1.0/day | ASSURE 0.29/day")
json.dump(R,open('final_alarms.json','w'),indent=1)
