"""Stream-process episode databases. Never hold a whole DB in memory."""
import wfdb, numpy as np, pickle, os, sys
from scipy.signal import resample_poly
FS=360; W=90
os.makedirs('data2',exist_ok=True)

def beats_from(db, rec, want_fs=360):
    r=wfdb.rdrecord(rec, pn_dir=db, channels=[0])
    a=wfdb.rdann(rec,'atr',pn_dir=db)
    sig=r.p_signal[:,0].astype(np.float32); fs=r.fs
    samp=a.sample.copy()
    if fs!=want_fs:
        from math import gcd
        g=gcd(int(fs),want_fs); up=want_fs//g; dn=int(fs)//g
        sig=resample_poly(sig,up,dn).astype(np.float32)
        samp=(samp*(want_fs/fs)).astype(int)
    X=[];P=[]
    for p in samp:
        if p-W<0 or p+W>=len(sig): continue
        s=sig[p-W:p+W]; sd=s.std()
        X.append(((s-s.mean())/(sd if sd>1e-6 else 1.0)).astype(np.float32)); P.append(p)
    return np.array(X), np.array(P), len(sig)/want_fs, a

# ---------- NSRDB: 18 healthy subjects, 24 h each. EVERY alarm is false ----------
def do_nsrdb():
    recs=wfdb.get_record_list('nsrdb'); out={}
    for rc in recs:
        try:
            X,P,dur,_=beats_from('nsrdb',rc)
            out[rc]=dict(X=X.astype(np.float16),P=P,dur=dur)
            print(f'nsrdb {rc}: {len(X)} beats {dur/3600:.1f} h',flush=True)
        except Exception as e: print('nsrdb FAIL',rc,e,flush=True)
    pickle.dump(out,open('data2/nsrdb.pkl','wb'))
    print('NSRDB total hours', sum(v['dur'] for v in out.values())/3600, flush=True)

# ---------- VFDB: real VT/VF episodes with rhythm onset annotations ----------
def do_rhythm(db):
    recs=wfdb.get_record_list(db); out={}
    for rc in recs:
        try:
            X,P,dur,a=beats_from(db,rc)
            ep=[]
            if a.aux_note is not None:
                sc=a.sample*(FS/ wfdb.rdheader(rc,pn_dir=db).fs)
                for s,note in zip(sc,a.aux_note):
                    n=(note or '').strip().rstrip('\x00')
                    if n.startswith('('): ep.append((float(s)/FS,n[1:]))
            out[rc]=dict(X=X.astype(np.float16),P=P,dur=dur,ep=ep)
            labs=set(l for _,l in ep)
            print(f'{db} {rc}: {len(X)} beats {dur/60:.0f} min rhythms={sorted(labs)[:6]}',flush=True)
        except Exception as e: print(db,'FAIL',rc,e,flush=True)
    pickle.dump(out,open(f'data2/{db}.pkl','wb'))
    print(db,'done',len(out),'records',flush=True)

if __name__=='__main__':
    t=sys.argv[1]
    if t=='nsrdb': do_nsrdb()
    else: do_rhythm(t)
