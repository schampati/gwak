import math
import pytest
from gwak.sky_position_antenna_patterns import event_to_gwak_from_first_principles as direct
from gwak.sky_position_antenna_patterns import _calendar, _tai_offset
from datetime import date


def test_no_astronomical_imports():
    # Load the combined file in an isolated interpreter while rejecting every
    # third-party import. Only the arithmetic function is invoked.
    import subprocess
    import sys
    import gwak.sky_position_antenna_patterns as module
    code = r"""
import importlib.abc
import runpy
import sys
class BlockAstronomy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'astropy', 'astropy_iers_data', 'erfa', 'numpy', 'torch', 'gwak'}:
            raise RuntimeError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockAstronomy())
namespace = runpy.run_path(sys.argv[1])
result = namespace['event_to_gwak_from_first_principles'](138.47, -1.94, '2011-05-14T01:32:22.654109')
assert result['ut1_mode'] == 'UTC approximation'
"""
    subprocess.run([sys.executable, '-I', '-c', code, module.__file__], check=True)


def test_calendar_reference_and_century_leap_rules():
    assert _calendar('2000-01-01T12:00:00')[1:]==(2451545,43200)
    assert _calendar('2000-02-29T00:00:00')[1]+1==_calendar('2000-03-01T00:00:00')[1]
    with pytest.raises(ValueError):_calendar('1900-02-29T00:00:00')


def test_leap_offsets():
    assert _tai_offset(date(2011,5,14))==34
    assert _tai_offset(date(2013,4,27))==35
    assert _tai_offset(date(2016,12,31))==36
    assert _tai_offset(date(2017,1,1))==37
    with pytest.raises(ValueError):_tai_offset(date(2028,1,1))


def test_rotation_sign_and_normalization():
    a=direct(0,0,'2019-01-14T00:00:00')
    b=direct(90,0,'2019-01-14T00:00:00')
    assert (b['earth_phi_rad']-a['earth_phi_rad'])%(2*math.pi)==pytest.approx(math.pi/2)
    later=direct(0,0,'2019-01-14T00:01:00')
    assert later['earth_phi_rad']<a['earth_phi_rad']
    assert sum(x*x for x in a['source_unit_vector'])==pytest.approx(1)
    assert a['ut1_mode']=='UTC approximation'


@pytest.mark.parametrize('kwargs',[
    {'ra_deg':360},{'dec_deg':91},{'ra_deg':float('nan')},
    {'trigger_time_utc':'2011-05-14'}, {'trigger_time_utc':'2016-12-31T23:59:60'},
    {'sky_window_deg':0},{'dec_deg':89.9},{'ut1_minus_utc_seconds':float('inf')},
])
def test_invalid_input(kwargs):
    args={'ra_deg':138.47,'dec_deg':-1.94,'trigger_time_utc':'2011-05-14T01:32:22.654109'}
    args.update(kwargs)
    with pytest.raises(ValueError):direct(**args)
