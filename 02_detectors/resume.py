import pickle,numpy as np,torch,torch.nn as nn,json,warnings,sys; warnings.filterwarnings('ignore')
from papers import BusiaTinyTransformer, FaragSTFTCNN, ArrythMLAutoencoder
torch.manual_seed(0); np.random.seed(0)
FS=360.; CL=['N','S','V','F','Q']
def P(*a): print(*a,flush=True)
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
recs=np.unique(d1['REC']); cal=set(recs[::5]); isc=np.array([x in cal for x in d1['REC']])

ae=ArrythMLAutoencoder()
ae.enc=nn.Sequential(nn.Linear(180,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,32))
ae.dec=nn.Sequential(nn.Linear(32,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),nn.Linear(256,180))
norm=X1[(~isc)&(y1==0)]; P(f"AE training on {len(norm):,} normal beats")
opt=torch.optim.Adam(ae.parameters(),lr=2e-3); mse=nn.MSELoss()
for e in range(15):
    perm=torch.randperm(len(norm)); t=0;n=0
    for i in range(0,len(norm),512):
        b=perm[i:i+512]; opt.zero_grad(); l=mse(ae(norm[b]),norm[b]); l.backward(); opt.step(); t+=l.item(); n+=1
    P(f"  ep{e+1}/15 loss {t/n:.5f}")
torch.save(ae.state_dict(),'f_ae.pt'); P("AE saved")

bu=BusiaTinyTransformer(d=20,heads=4); bu.load_state_dict(torch.load('f_busia.pt')); bu.eval()
fa=FaragSTFTCNN(ch=14); fa.load_state_dict(torch.load('f_farag.pt')); fa.eval(); ae.eval()
def sc(m,X,R,is_ae=False,bs=4096):
    o=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            o.append(m.score(X[i:i+bs]) if is_ae else 1-torch.softmax(m(X[i:i+bs],R[i:i+bs]),1)[:,0])
    return torch.cat(o).numpy()
TARGET=0.01
out={}
P("\nCALIBRATED so each model flags 1% of NORMAL beats on held-out DS1 records\n")
for nm,m,isae in [('Busia',bu,False),('Farag',fa,False),('ArrythML',ae,True)]:
    s1=sc(m,X1,R1,isae); thr=float(np.percentile(s1[isc&(y1==0)],99))
    s2=sc(m,X2,R2,isae); pred=(s2>thr).astype(int); true=(y2!=0).astype(int)
    tp=((pred==1)&(true==1)).sum(); fp=((pred==1)&(true==0)).sum(); fn=((pred==0)&(true==1)).sum()
    se=tp/max(tp+fn,1); pv=tp/max(tp+fp,1)
    P(f"{nm:<10} DS2  Se {se*100:5.1f}%  PPV {pv*100:5.1f}%  flagrate {pred.mean()*100:5.2f}%  thr {thr:.5f}")
    out[nm]=dict(thr=thr,se=float(se),ppv=float(pv))
json.dump(out,open('fixed_thr.json','w'),indent=1); P("\nwrote fixed_thr.json")
