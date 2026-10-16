"""CPU-only check in the real GWAK environment; no training/background/GPU access."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def verify(root, candidate_dir=None):
    import torch
    base=Path(candidate_dir or root)
    # Explicit loading also works when another installed GWAK package has __init__.py.
    load('gwak.sky_position_antenna_patterns',base/'gwak/sky_position_antenna_patterns.py')
    sky=load('gwak.sky_position',base/'gwak/sky_position.py')
    dl=load('gwak.train.dataloader',base/'gwak/train/dataloader.py')
    class BaseReached(Exception):pass
    original=dl.GwakBaseDataloader.__init__
    def stop(self,*args,**kwargs):raise BaseReached(self)
    dl.GwakBaseDataloader.__init__=stop
    def construct(**kwargs):
        try:
            dl.SignalDataloader([],[],[],[],snr_prior=torch.distributions.Uniform(4.,30.),**kwargs)
        except BaseReached as e:return e.args[0]
        raise AssertionError('Expected base initializer sentinel')
    try:
        x=construct(event_ra_deg=54.50484,event_dec_deg=-26.94637,
                    event_time_utc='2019-01-14T20:57:03')
        assert abs(x.fixed_phi_rad-6.042472390537084)<1e-6
        legacy=construct(fixed_ra=x.fixed_phi_rad,fixed_dec=x.fixed_declination_rad)
        torch.manual_seed(101);a=x.sample_sky_location(256)
        torch.manual_seed(101);b=legacy.sample_sky_location(256)
        assert all(torch.equal(i,j) for i,j in zip(a,b))
        # Mean phi moves ~180 degrees after 12 hours with RA/Dec unchanged.
        later=construct(event_ra_deg=54.50484,event_dec_deg=-26.94637,
                        event_time_utc='2019-01-15T08:57:03')
        delta=(x.fixed_phi_rad-later.fixed_phi_rad+math.pi)%(2*math.pi)-math.pi
        assert abs(delta)>3.0
        try:construct(event_ra_deg=54)
        except ValueError:pass
        else:raise AssertionError('Incomplete event accepted')
    finally:
        dl.GwakBaseDataloader.__init__=original
    # Check these angles actually pass through the local ml4gw detector projection.
    from ml4gw.gw import get_ifo_geometry,compute_observed_strain
    tensors,vertices=get_ifo_geometry('H1','L1')
    plus=torch.zeros(256,512);plus[:,256]=1
    cross=torch.zeros_like(plus);psi=torch.zeros(256)
    projected=compute_observed_strain(a[1],psi,a[0],tensors,vertices,4096,
                                     plus=plus,cross=cross)
    assert projected.shape==(256,2,512) and torch.isfinite(projected).all()
    assert projected.square().sum()>0
    factors=sky.antenna_patterns(x.sky_position)
    from ml4gw.gw import compute_antenna_responses
    ref=compute_antenna_responses(torch.tensor([math.pi/2-x.fixed_declination_rad]),
        torch.zeros(1),torch.tensor([x.fixed_phi_rad]),tensors,['plus','cross'])[0]
    for i,ifo in enumerate(('H1','L1')):
        assert abs(factors['detectors'][ifo]['F_plus']-float(ref[0,i]))<1e-6
        assert abs(factors['detectors'][ifo]['F_cross']-float(ref[1,i]))<1e-6
    result={'antenna_patterns':factors,'result':'SKY POSITION CHECK PASSED','phi_rad':x.fixed_phi_rad,
            'declination_rad':x.fixed_declination_rad,'default_window_deg':x.sky_window_deg,
            'constructor':'real dataloader; data/GPU base initialization intercepted',
            'projection':'real ml4gw H1/L1 impulse projection on CPU',
            'torch':torch.__version__,'astropy':x.sky_position['astropy_version']}
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate-dir',type=Path)
    args=p.parse_args();verify(args.root,args.candidate_dir)
