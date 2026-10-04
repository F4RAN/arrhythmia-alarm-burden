"""False-positive overlap of the three deployed detectors on healthy subjects.
Uses the calibrated models (f_*.pt) and the thresholds in fixed_thr.json, the same
operating point as every other result. Scores the first six NSRDB records."""
import pickle,numpy as np,torch,json,warnings; warnings.filterwarnings('ignore')
import torch.nn as nn
from papers import BusiaTinyTransformer, FaragSTFTCNN, ArrythMLAutoencoder
FS=360.; T=json.load(open('fixed_thr.json'))
bu=BusiaTinyTransformer(d=20,heads=4); bu.load_state_dict(torch.load('f_busia.pt')); bu.eval()
fa=FaragSTFTCNN(ch=14); fa.load_state_dict(torch.load('f_farag.pt')); fa.eval()
ae=ArrythMLAutoencoder()
ae.enc=nn.Sequential(nn.Linear(180,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,32))
ae.dec=nn.Sequential(nn.Linear(32,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,180))
ae.load_state_dict(torch.load('f_ae.pt')); ae.eval()
N=['Busia','Farag','ArrythML']
nsr=pickle.load(open('data2/nsrdb.pkl','rb')); keys=list(nsr)[:6]
FP=[set(),set(),set()]; off=0
for rc in keys:
    v=nsr[rc]; X=torch.tensor(v['X'].astype(np.float32))
    p=v['P']/FS; d=np.diff(p,prepend=p[0]); d2=np.diff(p,append=p[-1])
    mu=np.median(d[d>0]); R=torch.tensor(np.stack([np.clip(d/mu,0,3),np.clip(d2/mu,0,3)],1).astype(np.float32))
    with torch.no_grad():
        b=torch.cat([1-torch.softmax(bu(X[i:i+4096],R[i:i+4096]),1)[:,0] for i in range(0,len(X),4096)]).numpy()>T['Busia']['thr']
        f=torch.cat([1-torch.softmax(fa(X[i:i+4096]),1)[:,0] for i in range(0,len(X),4096)]).numpy()>T['Farag']['thr']
        a=torch.cat([ae.score(X[i:i+4096]) for i in range(0,len(X),4096)]).numpy()>T['ArrythML']['thr']
    for k,arr in enumerate([b,f,a]): FP[k] |= set(np.where(arr)[0]+off)
    off+=len(X)
print(f"beats scored: {off:,} across {len(keys)} healthy records (every flag is a false positive)")
for i,n in enumerate(N): print(f"  {n:<10} {len(FP[i]):>8,} false flags ({len(FP[i])/off*100:.2f}%)")
print("\nJaccard overlap of false positives:")
print(f"{'':<11}"+"".join(f"{n:>11}" for n in N))
for i,a_ in enumerate(N):
    print(f"{a_:<11}"+"".join(f"{len(FP[i]&FP[j])/max(len(FP[i]|FP[j]),1):>11.3f}" for j in range(3)))
U=set.union(*FP); I=set.intersection(*FP)
print(f"\nunion {len(U):,}   all-three-wrong {len(I):,}  ({len(I)/len(U)*100:.1f}% of union)")
print("note: ArrythML flags most healthy beats, so Jaccard values involving it are small partly because its set is large")
