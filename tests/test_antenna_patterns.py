import math
import random
import pytest
from gwak.sky_position_antenna_patterns import (
    antenna_patterns, event_to_gwak, event_to_gwak_from_first_principles,
    plot_antenna_patterns,
)


def test_matches_ml4gw_and_lal_across_sky():
    import torch
    import lal
    from ml4gw.gw import get_ifo_geometry, compute_antenna_responses
    rng = random.Random(20261012)
    points = [(rng.uniform(0, 2*math.pi), math.asin(rng.uniform(-1, 1)),
               rng.uniform(-4*math.pi, 4*math.pi)) for _ in range(256)]
    points += [(0, d, p) for d in (-math.pi/2, 0, math.pi/2) for p in (0, math.pi/4)]
    phi, dec, psi = [torch.tensor(v, dtype=torch.float64) for v in zip(*points)]
    tensors, _ = get_ifo_geometry('H1', 'L1')
    reference = compute_antenna_responses(math.pi/2-dec, psi, phi,
                                          tensors.double(), ['plus', 'cross'])
    for j, (p, d, s) in enumerate(points):
        output = antenna_patterns({'earth_phi_rad':p, 'earth_declination_rad':d},
                                   polarization_rad=s)
        for i, ifo in enumerate(('H1','L1')):
            actual = output['detectors'][ifo]
            # ml4gw geometry and LAL response tensors use float32 constants.
            expected = lal.ComputeDetAMResponse(lal.CachedDetectors[
                lal.LHO_4K_DETECTOR if ifo == 'H1' else lal.LLO_4K_DETECTOR].response,
                p, d, s, 0.0)
            for k, key in enumerate(('F_plus', 'F_cross')):
                assert actual[key] == pytest.approx(float(reference[j,k,i]), abs=2e-7)
                assert actual[key] == pytest.approx(expected[k], abs=2e-7)
            assert 0 <= actual['response_amplitude'] <= 1.000001


def test_polarization_and_antipodal_amplitude_invariance():
    position = {'earth_phi_rad':1.3,'earth_declination_rad':-.6}
    opposite = {'earth_phi_rad':(1.3+math.pi)%(2*math.pi),'earth_declination_rad':.6}
    a, b = antenna_patterns(position), antenna_patterns(opposite)
    for psi in (.17, math.pi/4, math.pi, -math.pi/3):
        c=antenna_patterns(position, polarization_rad=psi)
        for ifo in ('H1','L1'):
            x=a['detectors'][ifo];y=c['detectors'][ifo]
            assert y['response_amplitude']==pytest.approx(x['response_amplitude'],abs=1e-14)
            assert b['detectors'][ifo]['response_amplitude']==pytest.approx(x['response_amplitude'],abs=1e-14)
            assert y['F_plus']==pytest.approx(x['F_plus']*math.cos(2*psi)+x['F_cross']*math.sin(2*psi),abs=1e-14)
            assert y['F_cross']==pytest.approx(x['F_cross']*math.cos(2*psi)-x['F_plus']*math.sin(2*psi),abs=1e-14)


def test_accepts_both_conversions_and_old_import():
    import gwak.sky_position as old
    a=event_to_gwak(138.47,-1.94,'2011-05-14T01:32:22.654109')
    b=event_to_gwak_from_first_principles(138.47,-1.94,'2011-05-14T01:32:22.654109',
                                        ut1_minus_utc_seconds=a['ut1_minus_utc_seconds'])
    for ifo in ('H1','L1'):
        assert antenna_patterns(a)['detectors'][ifo]==pytest.approx(antenna_patterns(b)['detectors'][ifo],abs=1e-13)
    assert old.event_to_gwak is event_to_gwak
    assert old.antenna_patterns is antenna_patterns


@pytest.mark.parametrize('position,psi', [
    (None,0), ({},0), ({'earth_phi_rad':0,'earth_declination_rad':2},0),
    ({'earth_phi_rad':7,'earth_declination_rad':0},0),
    ({'earth_phi_rad':float('nan'),'earth_declination_rad':0},0),
    ({'earth_phi_rad':0,'earth_declination_rad':0},float('inf')),
    ({'earth_phi_rad':0,'earth_declination_rad':0,'convention':'other'},0),
])
def test_bad_inputs(position,psi):
    with pytest.raises(ValueError):antenna_patterns(position,polarization_rad=psi)


def test_plot_outputs(tmp_path):
    position={'earth_phi_rad':1.3,'earth_declination_rad':-.6}
    output=tmp_path/'test'
    result=plot_antenna_patterns(position,output=output)
    assert result==antenna_patterns(position)
    for extension in ('.pdf','.png','.json'):
        assert output.with_suffix(extension).stat().st_size>100
