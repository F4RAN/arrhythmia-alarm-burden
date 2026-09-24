import wfdb, os, numpy as np, pickle
DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]
AAMI = {}
for s in ['N','L','R','e','j']: AAMI[s]='N'
for s in ['A','a','J','S']:     AAMI[s]='S'
for s in ['V','E']:             AAMI[s]='V'
AAMI['F']='F'
for s in ['/','f','Q']:         AAMI[s]='Q'
OUT=os.path.expanduser('~/arrhythmia_exp/data'); os.makedirs(OUT,exist_ok=True)
FS=360; W_L=W_R=90
X,Y,REC,POS=[],[],[],[]; dur={}
for r in DS2:
    try:
        rec=wfdb.rdrecord(str(r),pn_dir='mitdb',channels=[0])
        ann=wfdb.rdann(str(r),'atr',pn_dir='mitdb')
    except Exception as e:
        print('FAIL',r,e,flush=True); continue
    sig=rec.p_signal[:,0].astype(np.float32); dur[r]=len(sig)/FS
    for pos,sym in zip(ann.sample,ann.symbol):
        if sym not in AAMI: continue
        if pos-W_L<0 or pos+W_R>=len(sig): continue
        seg=sig[pos-W_L:pos+W_R]; sd=seg.std()
        X.append(((seg-seg.mean())/(sd if sd>1e-6 else 1.0)).astype(np.float32))
        Y.append(AAMI[sym]); REC.append(r); POS.append(pos)
    print('rec',r,'cum',len(X),flush=True)
d=dict(X=np.array(X),Y=np.array(Y),REC=np.array(REC),POS=np.array(POS),dur_s=dur)
pickle.dump(d,open(f'{OUT}/ds2.pkl','wb'))
print('DONE',d['X'].shape,{k:int((d['Y']==k).sum()) for k in ['N','S','V','F','Q']},flush=True)
