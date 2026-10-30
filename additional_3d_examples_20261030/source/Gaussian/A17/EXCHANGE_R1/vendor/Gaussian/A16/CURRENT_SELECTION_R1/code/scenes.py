"""Frozen continuous geometries and sample split; reuses the established DDA."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
G=ROOT.parents[1]
for p in (G/'A10/code',G/'A9/code'):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from a10_common import balanced
from experiment_support import fibonacci

FAMILIES=('gaussian','contact','shell','asymmetric')

def seed(*parts):
    return int.from_bytes(hashlib.sha256(('A16|20260930|'+'|'.join(map(str,parts))).encode()).digest()[:4],'big')

def _component(center,axes,value,angle=0.,kind='ellipsoid'):
    return dict(center=list(map(float,center)),axes=list(map(float,axes)),contrast=[float(value),float(.025+.035*value)],angle=float(angle),kind=kind)

def make_scene(family,j,split,object_id,representation):
    rng=np.random.default_rng(seed('geometry',split,family,j))
    dev=split=='development'
    scene=dict(object_id=object_id,split=split,family=family,representation=representation,
               n=12,truth_n=14,edge=1.5,wavenumber=2.,rotation=float(rng.uniform(0,2*np.pi)),
               noise_seed=seed('noise',object_id),geometry_seed=seed('geometry',split,family,j),
               background=[.1,.04],init=[.1,.04],partial=False,strength_multiplier=1.)
    if family=='gaussian':
        count=(1+j if dev else 2+(j%3))
        centers=rng.uniform(-.35,.35,(count,3))
        if not dev: centers[:,2]=np.linspace(-.27,.27,count)
        scene['components']=[_component(c,rng.uniform(.16,.27,3),rng.uniform(.7,1.6),rng.uniform(0,np.pi),'gaussian') for c in centers]
    elif family=='contact':
        count=2 if dev else (3 if j%2 else 2)
        radius=.22 if dev else .19
        gap=[0.,.06,.12][j%3] if dev else [0.,.04,.08,.14][j%4]
        centers=np.zeros((count,3)); centers[:,0]=(np.arange(count)-(count-1)/2)*(2*radius+gap)
        if not dev: centers[:,1]=[.025*(-1)**i for i in range(count)]
        scene['components']=[_component(c,[radius,radius*(1.15 if not dev else 1),radius],.9+.55*i,rng.uniform(0,.25)) for i,c in enumerate(centers)]
        scene['surface_gap_nominal']=gap
    elif family=='shell':
        center=rng.uniform(-.055,.055,3)
        if dev and j==1:
            scene['components']=[_component(center,[.38,.30,.29],1.1,0,'box'),_component(center+[-.19,0,0],[.19,.30,.29],2.1,0,'box')]
        else:
            levels=2 if dev else 3
            axes=np.array([.43,.43,.43]) if dev else np.array([.43,.35,.39])
            comps=[]
            for i in range(levels):
                offset=np.zeros(3) if dev and j==0 else np.array([.04*i,-.018*i,.012*i])
                comps.append(_component(center+offset,axes*(1-.25*i),.8+.7*i,float(rng.uniform(0,.45))))
            scene['components']=comps
    else:
        count=2 if dev else 3
        centers=np.array([[-.29,-.15,.08],[.29,.17,-.12],[.05,-.30,-.24]])[:count]
        centers=centers+rng.uniform(-.035,.035,centers.shape)
        scene['components']=[_component(c,rng.uniform(.13,.19,3),.8+.65*i,rng.uniform(0,np.pi),'box' if (not dev and j%2 and i==1) else 'ellipsoid') for i,c in enumerate(centers)]
    return scene

def material(points,scene):
    p=np.asarray(points); bg=complex(*scene['background'])
    out=np.full(len(p),bg,dtype=complex)
    for c in scene['components']:
        q=p-np.asarray(c['center']); a=c['angle']; co,si=np.cos(a),np.sin(a)
        q=q@np.array([[co,-si,0],[si,co,0],[0,0,1.]])
        q=q/np.asarray(c['axes']); val=complex(*c['contrast'])*scene['strength_multiplier']
        if c['kind']=='gaussian': out+=val*np.exp(-.5*np.sum(q*q,axis=1))
        else:
            inside=np.max(np.abs(q),axis=1)<=1 if c['kind']=='box' else np.sum(q*q,axis=1)<=1
            out[inside]=bg+val  # frozen order defines nested layers, not sum of regions
    return out

def acquisition(scene,heldout=False):
    dirs,pol,rx,obs=balanced(96 if heldout else 64,rotation=scene['rotation']+(.37 if heldout else 0),partial=False)
    a=scene['rotation']; co,si=np.cos(a),np.sin(a)
    R=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
    dirs,pol=dirs@R.T,pol@R.T
    if scene.get('partial') and not heldout:
        keep=rx[:,0]>=0; rx,obs=rx[keep],obs[keep]
    return dirs,pol,rx,obs

def noise(data0,level,object_id,realization):
    if level==0:return np.zeros_like(data0),None
    s=seed('measurement_noise',object_id,level,realization)
    r=np.random.default_rng(s); e=r.normal(size=data0.shape)+1j*r.normal(size=data0.shape)
    return level*np.linalg.norm(data0)*e/np.linalg.norm(e),s

def generate():
    cases=[]
    for f,family in enumerate(FAMILIES):
        for j in range(3):
            rep='gaussian' if (f+j)%2==0 else 'voxel'
            cases.append(make_scene(family,j,'development',1001+3*f+j,rep))
        for j in range(4):
            cases.append(make_scene(family,j,'validation',2001+4*f+j,'gaussian' if j%2==0 else 'voxel'))
    for j in range(8):
        family=FAMILIES[j%4]
        c=make_scene(family,10+j,'extension',3001+j,'gaussian' if j%2==0 else 'voxel')
        if j<2:c['partial']=True;c['regime']='limited_aperture'
        elif j<4:c['strength_multiplier']=2.;c['regime']='stronger_scattering'
        else:c.update(n=8,representation='wavelet_full',regime='representation_transfer',paired_representation='voxel')
        cases.append(c)
    return cases

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'configs/scenes.json'));args=ap.parse_args()
    p=Path(args.out); data={'master_seed':20260930,'scenes':generate(),'kind':'frozen_before_new_Maxwell_results'}
    if p.exists():raise FileExistsError('Frozen scenes already exist; no silent regeneration')
    p.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'path':str(p),'objects':len(data['scenes']),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
