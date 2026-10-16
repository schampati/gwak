"""Scientific convention, sampler regression and event-input wiring checks (CPU)."""
import ast
import csv
import math
from pathlib import Path
import typing
import numpy as np
import pytest
import torch
from astropy.time import Time
from gwak.sky_position import (event_to_gwak, resolve_sky_position, SkyPositionMixin,
                               validate_sky_window, read_icecube_event, plot_event_position)

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('ra,dec,utc,expected',[
    (54.50484,-26.94637,'2019-01-14T20:57:03',6.042472390537084),
    (173.1362,27.7129,'2013-04-27T07:47:57',3.5026257992835097),
])
def test_archived_grb_and_independent_gmst(ra,dec,utc,expected):
    r=event_to_gwak(ra,dec,utc)
    assert abs(r['earth_phi_rad']-expected)<1e-6
    t=Time(utc,scale='utc');t.delta_ut1_utc=r['ut1_minus_utc_seconds']
    jd=t.ut1.jd;centuries=(jd-2451545.0)/36525
    gmst=math.radians((280.46061837+360.98564736629*(jd-2451545.0)
                       +.000387933*centuries**2-centuries**3/38710000)%360)
    error=(r['gmst_rad']-gmst+math.pi)%(2*math.pi)-math.pi
    assert abs(error)<2e-5
    assert sum(v*v for v in r['source_unit_vector'])==pytest.approx(1)
    assert r['earth_polar_deg']==pytest.approx(90-dec)


def test_trigger_time_changes_direction_and_mjd_matches():
    a=event_to_gwak(54,-26,'2019-01-14T20:57:03')
    b=event_to_gwak(54,-26,'2019-01-15T08:57:03')
    assert abs((a['earth_phi_rad']-b['earth_phi_rad']+math.pi)%(2*math.pi)-math.pi)>3.0
    c=event_to_gwak(54,-26,a['input']['mjd_utc'],time_format='mjd')
    assert c['earth_phi_rad']==pytest.approx(a['earth_phi_rad'],abs=1e-9)
    assert a['earth_declination_rad']==b['earth_declination_rad']


@pytest.mark.parametrize('kwargs',[
    dict(event_ra_deg=54),dict(event_dec_deg=1,event_time_utc='2019-01-01T00:00:00'),
    dict(fixed_phi_rad=1),dict(fixed_ra=1),
    dict(fixed_ra=1,fixed_dec=0,fixed_phi_rad=1,fixed_declination_rad=0),
    dict(event_ra_deg=54,event_dec_deg=-26,event_time_utc='2019-01-14T20:57:03',fixed_phi_rad=1,fixed_declination_rad=0),
    dict(fixed_phi_rad=0,fixed_declination_rad=math.pi/2),
    dict(fixed_phi_rad=float('nan'),fixed_declination_rad=0),
    dict(sky_window_deg=0),dict(sky_window_deg=float('inf')),
])
def test_invalid_sampler_input(kwargs):
    with pytest.raises(ValueError): resolve_sky_position(**kwargs)


@pytest.mark.parametrize('ra,dec,time',[(360,0,'2019-01-01T00:00:00'),
    (0,91,'2019-01-01T00:00:00'),(float('nan'),0,'2019-01-01T00:00:00'),
    (0,0,'2019-01-01'),(0,0,'2019-01-01T00:00:00-05:00'),(0,0,'2200-01-01T00:00:00')])
def test_bad_event(ra,dec,time):
    with pytest.raises(ValueError):event_to_gwak(ra,dec,time)


def test_actual_dataloader_constructor_wiring_and_sampler():
    # Execute the actual patched constructor until the data/GPU base initializer.
    # The sentinel prevents data access; the actual sampling method is retained.
    tree=ast.parse((ROOT/'gwak/train/dataloader.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SignalDataloader')
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('__init__','sample_sky_location')]
    class BaseReached(Exception):pass
    class Base:
        def __init__(self,*args,**kwargs): raise BaseReached(self)
    namespace=dict(SkyPositionMixin=SkyPositionMixin,GwakBaseDataloader=Base,
                   Optional=typing.Optional,List=typing.List,Union=typing.Union,
                   TransformedDistribution=torch.distributions.TransformedDistribution,
                   Distribution=torch.distributions.Distribution)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'actual_constructor','exec'),namespace)
    actual=namespace['SignalDataloader']
    def build(**sky):
        try:actual([],[],[],[],snr_prior=torch.distributions.Uniform(4.,30.),**sky)
        except BaseReached as e:return e.args[0]
    event=build(event_ra_deg=54.50484,event_dec_deg=-26.94637,event_time_utc='2019-01-14T20:57:03')
    r=event_to_gwak(54.50484,-26.94637,'2019-01-14T20:57:03')
    earth=build(**r['gwak_sampling'])
    legacy=build(fixed_ra=earth.fixed_phi_rad,fixed_dec=earth.fixed_declination_rad)
    for loader in [event,earth,legacy]:
        torch.manual_seed(25);phi,dec=loader.sample_sky_location(2048)
        torch.manual_seed(25)
        old_phi=(earth.fixed_phi_rad+(torch.rand(2048)-.5)*np.deg2rad(1.))%(2*torch.pi)
        old_dec=earth.fixed_declination_rad+(torch.rand(2048)-.5)*np.deg2rad(1.)
        assert torch.equal(phi,old_phi) and torch.equal(dec,old_dec)
    assert event.sky_position['input']['trigger_time_utc'].startswith('2019-01-14')
    # Archived evaluators mutate aliases after constructing an all-sky loader.
    dynamic=build();dynamic.fixed_ra=earth.fixed_phi_rad;dynamic.fixed_dec=earth.fixed_declination_rad
    assert dynamic.fixed_phi_rad==earth.fixed_phi_rad
    assert dynamic.fixed_declination_rad==earth.fixed_declination_rad
    # No unpaired/ambiguous event input reaches file I/O or CUDA initialization.
    with pytest.raises(ValueError):build(event_ra_deg=54)


def test_wrap_window_and_all_sky_regression():
    obj=SkyPositionMixin();obj.configure_sky_position(fixed_phi_rad=.0001,fixed_declination_rad=0)
    phi,dec=obj._sample_sky_location(10000)
    assert phi.min()>=0 and phi.max()<2*torch.pi
    offset=(phi-.0001+torch.pi)%(2*torch.pi)-torch.pi
    assert offset.abs().max()<math.radians(.5001)
    assert dec.abs().max()<math.radians(.5001)
    class Prior:
        def sample(self,shape):return torch.rand(shape)
    obj.configure_sky_position();obj.ra_prior=Prior();obj.dec_prior=Prior()
    torch.manual_seed(2);out=obj._sample_sky_location(20)
    torch.manual_seed(2);assert torch.equal(out[0],torch.rand(20));assert torch.equal(out[1],torch.rand(20))


def test_icecube_and_plot(tmp_path):
    csvpath=ROOT/'examples/sky_position/icecube_event.csv'
    e=read_icecube_event(csvpath,'IC110514A')
    assert e['ra_deg']==138.47 and e['dec_deg']==-1.94
    assert e['catalog']['start_mjd_difference_seconds']<.001
    r=plot_event_position(e['ra_deg'],e['dec_deg'],e['trigger_time'],output=tmp_path/'icecube',name=e['name'])
    for ext in ('png','pdf','json'):assert (tmp_path/f'icecube.{ext}').stat().st_size>100
    assert r['window']['half_width_deg']==.5
    with pytest.raises(ValueError):read_icecube_event(csvpath,'not-an-event')
    text=csvpath.read_text().replace('55695.06415108922','55696.06415108922')
    bad=tmp_path/'bad.csv';bad.write_text(text)
    with pytest.raises(ValueError,match='disagree'):read_icecube_event(bad)


def test_patch_idempotence_and_preserves_other_changes():
    import importlib.util
    spec=importlib.util.spec_from_file_location('patcher',ROOT/'scripts/patch_sky_sampler.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    patched=(ROOT/'gwak/train/dataloader.py').read_bytes()
    assert module.patch_dataloader(patched)==patched
    original=ROOT/'dataloader.original.py'
    if original.exists():
        extra=original.read_bytes()+b'\n# unrelated user change\n'
        assert module.patch_dataloader(extra).endswith(b'# unrelated user change\n')
        crlf=original.read_text().replace('\n','\r\n').encode()
        assert b'event_time_utc' in module.patch_dataloader(crlf)
