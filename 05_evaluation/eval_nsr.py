"""False alarm rate on NSRDB: 18 healthy subjects, ~24h each.
   Every alarm is FALSE by definition. No sensitivity ambiguity."""
import pickle, numpy as np, json
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

d1=pickle.load(open('data/ds1.pkl','rb'))
X1=d1['X']; y1=(d1['Y']!='N').astype(int)
models={
 'MLP-raw'   : make_pipeline(StandardScaler(),MLPClassifier((64,32),max_iter=60,random_state=0)),
 'RandForest': RandomForestClassifier(n_estimators=120,min_samples_leaf=3,n_jobs=-1,random_state=0),
 'ExtraTrees': ExtraTreesClassifier(n_estimators=120,min_samples_leaf=3,n_jobs=-1,random_state=0),
 'LogReg'    : make_pipeline(StandardScaler(),LogisticRegression(max_iter=400)),
}
for m in models.values(): m.fit(X1,y1)
print('models trained on DS1',flush=True)

nsr=pickle.load(open('data2/nsrdb.pkl','rb'))
FS=360.
def episodes(flag,t,k,max_gap=1.2,refr=300.):
    n=len(flag); i=0; last=-1e9; c=0
    while i<n:
        if flag[i]:
            j=i
            while j+1<n and flag[j+1] and (t[j+1]-t[j])<=max_gap: j+=1
            if j-i+1>=k and t[i]-last>=refr: c+=1; last=t[i]
            i=j+1
        else: i+=1
    return c

TOTH=sum(v['dur'] for v in nsr.values())/3600
print(f'NSRDB: {len(nsr)} records, {TOTH:.1f} hours total\n',flush=True)
print(f"{'model':<12}"+"".join(f"{'k='+str(k):>11}" for k in [1,3,5,6]))
print(f"{'':12}"+"".join(f"{'falseAl/24h':>11}" for k in [1,3,5,6]))
res={}
for nm,m in models.items():
    tot={k:0 for k in [1,3,5,6]}
    for rc,v in nsr.items():
        X=v['X'].astype(np.float32); t=v['P']/FS
        p=m.predict(X)
        for k in tot: tot[k]+=episodes(p,t,k)
    row=f"{nm:<12}"
    res[nm]={}
    for k in [1,3,5,6]:
        rate=tot[k]*24.0/TOTH; res[nm][k]=rate
        row+=f"{rate:>11.2f}"
    print(row,flush=True)
json.dump(res,open('nsr_results.json','w'),indent=1)
print("\nASSURE bar 0.29/day | target <1.0/day")
