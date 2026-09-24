import pickle,numpy as np,torch,json,warnings; warnings.filterwarnings('ignore')
import torch.nn as nn
from papers import BusiaTinyTransformer, FaragSTFTCNN, ArrythMLAutoencoder
FS=360.
bu=BusiaTinyTransformer(d=20,heads=4); bu.load_state_dict(torch.load('m_busia.pt')); bu.eval()
fa=FaragSTFTCNN(ch=14); fa.load_state_dict(torch.load('m_farag.pt')); fa.eval()
ae=ArrythMLAutoencoder()
ae.enc=nn.Sequential(nn.Linear(180,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,32))
ae.dec=nn.Sequential(nn.Linear(32,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,180))
ae.load_state_dict(torch.load('m_arrythml.pt')); ae.eval()
THR=json.load(open('ae_thr.json'))['thr']
N=['Busia','Farag','ArrythML']
nsr=pickle.load(open('data2/nsrdb.pkl','rb'))
# use 6 records for speed - still ~145 h
keys=list(nsr)[:6]
FP=[set(),set(),set()]; off=0
for rc in keys:
    v=nsr[rc]; X=torch.tensor(v['X'].astype(np.float32))
    p=v['P']/FS; d=np.diff(p,prepend=p[0]); d2=np.diff(p,append=p[-1])
    mu=np.median(d[d>0]); R=torch.tensor(np.stack([np.clip(d/mu,0,3),np.clip(d2/mu,0,3)],1).astype(np.float32))
    with torch.no_grad():
        b=torch.cat([(bu(X[i:i+4096],R[i:i+4096]).argmax(1)!=0).int() for i in range(0,len(X),4096)]).numpy()
        f=torch.cat([(fa(X[i:i+4096]).argmax(1)!=0).int() for i in range(0,len(X),4096)]).numpy()
        a=torch.cat([(ae.score(X[i:i+4096])>THR).int() for i in range(0,len(X),4096)]).numpy()
    for k,arr in enumerate([b,f,a]):
        FP[k] |= set(np.where(arr==1)[0]+off)
    off+=len(X)
print(f"beats scored: {off:,} across {len(keys)} healthy records")
print("\nALL these flags are FALSE POSITIVES (healthy subjects)")
for i,n in enumerate(N): print(f"  {n:<10} {len(FP[i]):>8,} false flags")
print("\nJaccard overlap of false positives:")
print(f"{'':<11}"+"".join(f"{n:>11}" for n in N))
for i,a in enumerate(N):
    row=f"{a:<11}"
    for j,b in enumerate(N):
        u=len(FP[i]|FP[j]); row+=f"{len(FP[i]&FP[j])/u if u else 0:>11.3f}"
    print(row)
U=set.union(*FP); I=set.intersection(*FP)
print(f"\nunion {len(U):,}   all-three-wrong {len(I):,}  ({len(I)/len(U)*100:.1f}% of union)")
print("compare: 4 diverse sklearn families gave 0.5% all-wrong")
