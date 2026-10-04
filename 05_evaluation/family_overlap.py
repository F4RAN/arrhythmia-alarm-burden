"""False-positive overlap between four classical detector families (Fig. 5).
Trained on DS1, tested on DS2; a false positive is a normal DS2 beat flagged abnormal."""
import pickle,numpy as np,os
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
V='.'
d1=pickle.load(open('data/ds1.pkl','rb')); d2=pickle.load(open('data/ds2.pkl','rb'))
X1=d1['X']; y1=(d1['Y']!='N').astype(int); X2=d2['X']; y2=(d2['Y']!='N').astype(int)
models={'MLP':make_pipeline(StandardScaler(),MLPClassifier((64,32),max_iter=60,random_state=0)),
 'RF':RandomForestClassifier(n_estimators=120,min_samples_leaf=3,n_jobs=-1,random_state=0),
 'ET':ExtraTreesClassifier(n_estimators=120,min_samples_leaf=3,n_jobs=-1,random_state=0),
 'LR':make_pipeline(StandardScaler(),LogisticRegression(max_iter=400))}
fps={}
for n,m in models.items():
    m.fit(X1,y1); pr=m.predict(X2); fps[n]=set(np.where((pr==1)&(y2==0))[0]); print(n,'FP beats',len(fps[n]),flush=True)
print(f"evaluated on DS2: {len(y2):,} beats, {(y2==0).sum():,} normal (FP = flagged normal beat)")
pairs=[('RF','ET'),('MLP','RF'),('MLP','ET'),('MLP','LR'),('RF','LR'),('ET','LR')]
for a,b in pairs: print(f"Jaccard {a}-{b}: {len(fps[a]&fps[b])/len(fps[a]|fps[b]):.3f}")
U=set.union(*fps.values()); I=set.intersection(*fps.values()); print(f"all-four-wrong {len(I)} of union {len(U)} = {len(I)/len(U)*100:.2f}%")
