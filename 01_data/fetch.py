import wfdb, os, numpy as np, pickle, sys

# de Chazal inter-patient split (paced records 102,104,107,217 excluded per AAMI)
DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]

# AAMI mapping
AAMI = {}
for s in ['N','L','R','e','j']: AAMI[s]='N'
for s in ['A','a','J','S']:     AAMI[s]='S'
for s in ['V','E']:             AAMI[s]='V'
AAMI['F']='F'
for s in ['/','f','Q']:         AAMI[s]='Q'

OUT = os.path.expanduser('~/arrhythmia_exp/data')
os.makedirs(OUT, exist_ok=True)
FS = 360
W_L, W_R = 90, 90   # 0.25 s either side of R peak -> 180 samples

def grab(recs, name):
    X, Y, REC, POS = [], [], [], []
    dur_s = {}
    for r in recs:
        try:
            rec = wfdb.rdrecord(str(r), pn_dir='mitdb', channels=[0])
            ann = wfdb.rdann(str(r), 'atr', pn_dir='mitdb')
        except Exception as e:
            print('FAIL', r, e); continue
        sig = rec.p_signal[:,0].astype(np.float32)
        dur_s[r] = len(sig)/FS
        for pos, sym in zip(ann.sample, ann.symbol):
            if sym not in AAMI: continue
            if pos-W_L < 0 or pos+W_R >= len(sig): continue
            seg = sig[pos-W_L:pos+W_R]
            sd = seg.std()
            seg = (seg - seg.mean())/(sd if sd>1e-6 else 1.0)
            X.append(seg.astype(np.float32)); Y.append(AAMI[sym])
            REC.append(r); POS.append(pos)
        print(f'{name} rec {r}: {len(ann.sample)} anns, cum beats {len(X)}', flush=True)
    d = dict(X=np.array(X), Y=np.array(Y), REC=np.array(REC), POS=np.array(POS), dur_s=dur_s)
    with open(f'{OUT}/{name}.pkl','wb') as f: pickle.dump(d, f)
    print(name, d['X'].shape, {k:int((d['Y']==k).sum()) for k in ['N','S','V','F','Q']})

grab(DS1,'ds1'); grab(DS2,'ds2')
