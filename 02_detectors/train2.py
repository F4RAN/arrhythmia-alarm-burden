"""Fixed training: mild class weighting + operating point calibrated on
   NORMAL beats, which is what a deployed device actually needs."""
import pickle,numpy as np,torch,torch.nn as nn,json,warnings; warnings.filterwarnings('ignore')
from papers import BusiaTinyTransformer, FaragSTFTCNN, ArrythMLAutoencoder, nparams
torch.manual_seed(0); np.random.seed(0)
FS=360.; CL=['N','S','V','F','Q']

def rr(POS,REC):
    pre=np.zeros(len(POS),np.float32); post=np.zeros(len(POS),np.float32)
    for rc in np.unique(REC):
        m=np.where(REC==rc)[0]; p=POS[m]/FS
        d=np.diff(p,prepend=p[0]); d2=np.diff(p,append=p[-1])
        mu=np.median(d[d>0]) if (d>0).any() else 1.
        pre[m]=np.clip(d/mu,0,3); post[m]=np.clip(d2/mu,0,3)
    return np.stack([pre,post],1)

d1=pickle.load(open('data/ds1.pkl','rb')); d2=pickle.load(open('data/ds2.pkl','rb'))
X1=torch.tensor(d1['X']); y1=np.array([CL.index(c) for c in d1['Y']]); R1=torch.tensor(rr(d1['POS'],d1['REC']))
X2=torch.tensor(d2['X']); y2=np.array([CL.index(c) for c in d2['Y']]); R2=torch.tensor(rr(d2['POS'],d2['REC']))
y1t=torch.tensor(y1)
# hold out 20% of DS1 records for threshold calibration
recs=np.unique(d1['REC']); cal=set(recs[::5])
isc=np.array([x in cal for x in d1['REC']])
print(f"DS1 train {(~isc).sum():,}  calib {isc.sum():,}  DS2 {len(y2):,}")

def train(model,name,ep=25):
    opt=torch.optim.Adam(model.parameters(),lr=2e-3)
    cnt=np.bincount(y1[~isc],minlength=5).astype(np.float32)
    w=np.sqrt(cnt.sum()/(5*np.maximum(cnt,1)))          # MILD, was linear before
    w=torch.tensor(np.clip(w/w.min(),1,8).astype(np.float32))
    lf=nn.CrossEntropyLoss(weight=w)
    idx=np.where(~isc)[0]
    for e in range(ep):
        perm=np.random.permutation(idx); model.train()
        for i in range(0,len(perm),256):
            b=torch.tensor(perm[i:i+256]); opt.zero_grad()
            lf(model(X1[b],R1[b]),y1t[b]).backward(); opt.step()
    return model

def abn_score(model,X,R,ae=False,bs=4096):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            if ae: out.append(model.score(X[i:i+bs]))
            else:
                p=torch.softmax(model(X[i:i+bs],R[i:i+bs]),1); out.append(1-p[:,0])
    return torch.cat(out).numpy()

def calibrate(s_cal_normal,target_fpr):
    """threshold so only target_fpr of NORMAL beats are flagged"""
    return float(np.percentile(s_cal_normal,100*(1-target_fpr)))

def report(name,s2,thr):
    pred=(s2>thr).astype(int); true=(y2!=0).astype(int)
    tp=((pred==1)&(true==1)).sum(); fp=((pred==1)&(true==0)).sum()
    fn=((pred==0)&(true==1)).sum()
    se=tp/max(tp+fn,1); pv=tp/max(tp+fp,1)
    print(f"  {name}: DS2 Se {se*100:.1f}%  PPV {pv*100:.1f}%  F1 {200*se*pv/max(se+pv,1e-9):.1f}%  flagrate {pred.mean()*100:.2f}%")
    return dict(se=float(se),ppv=float(pv),thr=thr)

TARGET_FPR=0.01     # flag 1% of normal beats - a sane device operating point
out={}
print("\n=== Busia 2024 ===")
bu=BusiaTinyTransformer(d=20,heads=4); print(f"  {nparams(bu):,} params (paper 6,643)")
train(bu,'Busia'); sc=abn_score(bu,X1,R1); thr=calibrate(sc[isc&(y1==0)],TARGET_FPR)
out['Busia']=report('Busia',abn_score(bu,X2,R2),thr); torch.save(bu.state_dict(),'f_busia.pt')

print("=== Farag 2023 ===")
fa=FaragSTFTCNN(ch=14); print(f"  {nparams(fa):,} params (paper 1,267-1,619)")
train(fa,'Farag'); sc=abn_score(fa,X1,R1); thr=calibrate(sc[isc&(y1==0)],TARGET_FPR)
out['Farag']=report('Farag',abn_score(fa,X2,R2),thr); torch.save(fa.state_dict(),'f_farag.pt')

print("=== ArrythML 2026 ===")
ae=ArrythMLAutoencoder()
ae.enc=nn.Sequential(nn.Linear(180,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,32))
ae.dec=nn.Sequential(nn.Linear(32,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,180))
print(f"  {nparams(ae):,} params (paper ~180 kB)")
norm=X1[(~isc)&(y1==0)]; opt=torch.optim.Adam(ae.parameters(),lr=2e-3); mse=nn.MSELoss()
for e in range(25):
    perm=torch.randperm(len(norm))
    for i in range(0,len(norm),256):
        b=perm[i:i+256]; opt.zero_grad(); mse(ae(norm[b]),norm[b]).backward(); opt.step()
sc=abn_score(ae,X1,R1,ae=True); thr=calibrate(sc[isc&(y1==0)],TARGET_FPR)   # FIXED: on NORMAL beats
out['ArrythML']=report('ArrythML',abn_score(ae,X2,R2,ae=True),thr); torch.save(ae.state_dict(),'f_ae.pt')

json.dump(out,open('fixed_thr.json','w'),indent=1)
print("\nthresholds calibrated so each model flags 1% of NORMAL beats")
