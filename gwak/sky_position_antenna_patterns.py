"""Event coordinates -> fixed Earth-relative direction and H1/L1 antenna patterns.

Two calculation functions live here:
  event_to_gwak: Astropy-based time conversion.
  event_to_gwak_from_first_principles: independent standard-library arithmetic.
Astropy is imported only when the Astropy-based path is called.

This file also runs independently of GWAK/PyTorch:
  python sky_position_antenna_patterns.py --ra-deg 54.50484 --dec-deg -26.94637 \
      --trigger-time 2019-01-14T20:57:03 --output grb190114c

Convention: phi = (RA - Greenwich mean sidereal time) mod 2*pi;
declination is unchanged. This preserves the existing GWAK study convention,
not a full ICRS-to-ITRS transformation (no precession/nutation/polar motion).
The vector points TOWARD the source; propagation is in the opposite direction.
"""
from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import warnings

DEFAULT_SKY_WINDOW_DEG = 1.0
CONVENTION = 'gwak_ra_minus_gmst_iau2006'


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number, not a boolean')
    try:
        result = float(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f'{name} must be a finite number') from e
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def validate_sky_window(phi_rad, declination_rad, sky_window_deg=DEFAULT_SKY_WINDOW_DEG):
    """Validate a full-width phi/declination box; never silently clip at poles."""
    width = _number(sky_window_deg, 'sky_window_deg')
    if not 0 < width <= 360:
        raise ValueError('sky_window_deg must be in (0, 360] (full box width)')
    if (phi_rad is None) != (declination_rad is None):
        raise ValueError('fixed_phi_rad and fixed_declination_rad must be supplied together')
    if phi_rad is None:
        return None, None, width
    phi = _number(phi_rad, 'fixed_phi_rad')
    dec = _number(declination_rad, 'fixed_declination_rad')
    if not 0 <= phi < 2 * math.pi:
        raise ValueError('fixed_phi_rad must be in [0, 2*pi) radians')
    if abs(dec) > math.pi / 2:
        raise ValueError('fixed_declination_rad must be in [-pi/2, pi/2] radians')
    if abs(dec) + math.radians(width) / 2 > math.pi / 2:
        raise ValueError('Sky box crosses a pole; reduce sky_window_deg')
    return phi, dec, width


def _utc_time(trigger_time, time_format):
    from astropy.time import Time
    if time_format == 'isot':
        if not isinstance(trigger_time, str) or not re.fullmatch(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?', trigger_time
        ):
            raise ValueError('Use a full UTC date/time: YYYY-MM-DDTHH:MM:SS[.fraction][Z]')
        value = trigger_time.rstrip('Z').replace(' ', 'T')
    elif time_format == 'mjd':
        value = _number(trigger_time, 'trigger_time (MJD UTC)')
    else:
        raise ValueError("time_format must be 'isot' or 'mjd'; both use UTC")
    return Time(value, format=time_format, scale='utc', precision=9)


def event_to_gwak(ra_deg, dec_deg, trigger_time, *, time_format='isot',
                  sky_window_deg=DEFAULT_SKY_WINDOW_DEG, temp_plot=False):
    """Convert one event to JSON-compatible angles and ready-to-use sampler kwargs.

    ra_deg / dec_deg: numeric catalog equatorial coordinates in DEGREES.
    trigger_time: full UTC string, or numeric MJD with time_format='mjd'.
    sky_window_deg: full width in each coordinate, default 1 (= +/-0.5 deg).

    Uses Astropy mean Greenwich sidereal time (IAU2006) and bundled IERS data,
    without network downloads. Unsupported epochs fail rather than extrapolate.
    Conversion happens once; background timestamps do not affect sampling.

    temp_plot=True opens an unsaved window; close it to return the same result.
    Requires a local interactive Matplotlib backend. Default False does no plotting.
    """
    import astropy
    import astropy_iers_data
    from astropy.utils import iers
    ra = _number(ra_deg, 'ra_deg')
    dec = _number(dec_deg, 'dec_deg')
    if not 0 <= ra < 360:
        raise ValueError('ra_deg must be in [0, 360) degrees')
    if not -90 <= dec <= 90:
        raise ValueError('dec_deg must be in [-90, 90] degrees')
    epoch = _utc_time(trigger_time, time_format)
    with iers.conf.set_temp('auto_download', False):
        dut1, status = epoch.get_delta_ut1_utc(return_status=True)
        if int(status) < 0:
            raise ValueError('Trigger time is outside bundled IERS coverage; update astropy-iers-data')
        # Record and use the identical UT1 correction, including predicted status.
        epoch.delta_ut1_utc = dut1.to_value('s')
        gmst = float(epoch.sidereal_time('mean', 'greenwich', model='IAU2006').rad)
    phi = (math.radians(ra) - gmst) % (2 * math.pi)
    phi, declination, width = validate_sky_window(phi, math.radians(dec), sky_window_deg)
    polar = math.pi / 2 - declination
    vector = [math.cos(declination)*math.cos(phi),
              math.cos(declination)*math.sin(phi), math.sin(declination)]
    result = {
        'input': {'ra_deg': ra, 'dec_deg': dec, 'trigger_time_utc': epoch.utc.isot,
                  'mjd_utc': float(epoch.utc.mjd)},
        'convention': CONVENTION,
        'earth_phi_rad': phi, 'earth_declination_rad': declination,
        'earth_polar_rad': polar,
        'earth_phi_deg': math.degrees(phi), 'earth_declination_deg': dec,
        'earth_polar_deg': math.degrees(polar), 'source_unit_vector': vector,
        'gmst_rad': gmst, 'ut1_minus_utc_seconds': float(dut1.to_value('s')),
        'iers_status': int(status), 'astropy_version': astropy.__version__,
        'astropy_iers_data_version': astropy_iers_data.__version__,
        'window': {'shape': 'phi_declination_box', 'full_width_deg': width,
                   'half_width_deg': width / 2},
        'gwak_sampling': {'fixed_phi_rad': phi,
                          'fixed_declination_rad': declination,
                          'sky_window_deg': width},
    }
    if temp_plot:
        _show_temporary_figure(_position_figure(result, name="Event"))
    return result


# Independent arithmetic implementation (no astronomy-library calls).
# Uses measured UT1-UTC when supplied; otherwise explicitly assumes UT1=UTC.
# Formula references:
# https://raw.githubusercontent.com/liberfa/erfa/master/src/gmst06.c
# https://raw.githubusercontent.com/liberfa/erfa/master/src/era00.c
# https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat
# Effective UTC dates and TAI-UTC, from IERS Leap_Second.dat, Bulletin C 72.
# Snapshot checked 2026-10-12. Refuse to assume future leap-second decisions.
LEAP_TABLE_EXPIRES = date(2027, 6, 28)
TAI_MINUS_UTC = (
    (1972,1,1,10), (1972,7,1,11), (1973,1,1,12), (1974,1,1,13),
    (1975,1,1,14), (1976,1,1,15), (1977,1,1,16), (1978,1,1,17),
    (1979,1,1,18), (1980,1,1,19), (1981,7,1,20), (1982,7,1,21),
    (1983,7,1,22), (1985,7,1,23), (1988,1,1,24), (1990,1,1,25),
    (1991,1,1,26), (1992,7,1,27), (1993,7,1,28), (1994,7,1,29),
    (1996,1,1,30), (1997,7,1,31), (1999,1,1,32), (2006,1,1,33),
    (2009,1,1,34), (2012,7,1,35), (2015,7,1,36), (2017,1,1,37),
)


def _finite(value, name):
    if isinstance(value, bool):raise ValueError(f'{name} must be numeric, not boolean')
    try:v=float(value)
    except (ValueError,TypeError) as e:raise ValueError(f'{name} must be numeric') from e
    if not math.isfinite(v):raise ValueError(f'{name} must be finite')
    return v


def _calendar(trigger_time_utc):
    """Gregorian calendar to integer Julian day number and UTC seconds of day."""
    if not isinstance(trigger_time_utc,str):raise ValueError('Expected a full UTC timestamp string')
    match=re.fullmatch(r'(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)Z?',trigger_time_utc)
    if match is None:raise ValueError('Use YYYY-MM-DDTHH:MM:SS[.fraction][Z] in UTC')
    year,month,day,hour,minute=map(int,match.groups()[:5]);second=float(match.group(6))
    day_date=date(year,month,day)  # validate month length / leap year; no time conversion
    if not (0<=hour<24 and 0<=minute<60 and 0<=second<60):
        raise ValueError('Invalid clock time; literal leap-second timestamps (:60) are not supported')
    # Integer Gregorian Julian-day formula; JDN labels NOON, so midnight is JDN-0.5.
    shift=(14-month)//12
    y=year+4800-shift;m=month+12*shift-3
    jdn=day+(153*m+2)//5+365*y+y//4-y//100+y//400-32045
    return day_date,jdn,3600*hour+60*minute+second


def _tai_offset(day_date):
    if not date(1972,1,1)<=day_date<LEAP_TABLE_EXPIRES:
        raise ValueError('Date outside the leap-table validity; supply a verified tai_minus_utc_seconds')
    for year,month,day,offset in reversed(TAI_MINUS_UTC):
        if day_date>=date(year,month,day):return float(offset)
    raise AssertionError('Unreachable leap-table date')


def event_to_gwak_from_first_principles(ra_deg, dec_deg, trigger_time_utc, *,
                                       ut1_minus_utc_seconds=None,
                                       tai_minus_utc_seconds=None,
                                       sky_window_deg=1.0, temp_plot=False):
    """RA/Dec in degrees + UTC -> Earth-relative GWAK angles and unit vector.

    For a precise comparison, use the same measured UT1-UTC as the reference.
    Default UT1=UTC is a documented approximation, not an exact IERS correction.
    No astronomical library is required, including for the leap-second lookup.

    temp_plot=True opens an unsaved window; close it to return the same result.
    Requires a local interactive Matplotlib backend. Default False does no plotting.
    """
    ra=_finite(ra_deg,'ra_deg');dec=_finite(dec_deg,'dec_deg')
    width=_finite(sky_window_deg,'sky_window_deg')
    if not 0<=ra<360:raise ValueError('ra_deg must be in [0,360)')
    if not -90<=dec<=90:raise ValueError('dec_deg must be in [-90,90]')
    if not 0<width<=360 or abs(dec)+width/2>90:
        raise ValueError('Window must have positive full width and stay within the poles')
    day_date,jdn,seconds=_calendar(trigger_time_utc)
    dut1=0. if ut1_minus_utc_seconds is None else _finite(ut1_minus_utc_seconds,'UT1-UTC')
    dat=_tai_offset(day_date) if tai_minus_utc_seconds is None else _finite(tai_minus_utc_seconds,'TAI-UTC')

    # 1. Convert UTC to rotation time UT1 and uniform time TT.
    #    UT1 = UTC + DUT1; TT = UTC + (TAI-UTC) + 32.184 seconds.
    ut1_fraction=(seconds+dut1)/86400.
    tt_fraction=(seconds+dat+32.184)/86400.
    midnight_from_j2000=jdn-2451545-.5
    ut1_days=midnight_from_j2000+ut1_fraction
    tt_centuries=(midnight_from_j2000+tt_fraction)/36525.

    # 2. Earth rotation angle: 2pi[0.7790572732640 + 1.00273781191135448*d].
    #    Split the integer turns out BEFORE summation to retain precision.
    #    JD at midnight has fractional part 0.5; integer turns vanish modulo 2pi.
    era_turns=.7790572732640 + .00273781191135448*ut1_days + .5 + ut1_fraction
    era=(2*math.pi*era_turns)%(2*math.pi)

    # 3. Precession-related offset from ERA to mean Greenwich sidereal time.
    #    Coefficients are ARCSECONDS; powers use TT centuries since J2000.
    coefficients=(.014506,4612.156534,1.3915817,-.00000044,-.000029956,-.0000000368)
    offset_arcsec=math.fsum(c*tt_centuries**power for power,c in enumerate(coefficients))
    gmst=(era+math.radians(offset_arcsec/3600.))%(2*math.pi)

    # 4. Rotate the catalog direction about Earth's z axis by minus GMST.
    phi=(math.radians(ra)-gmst)%(2*math.pi)
    declination=math.radians(dec);theta=math.pi/2-declination
    vector=[math.cos(declination)*math.cos(phi),
            math.cos(declination)*math.sin(phi),math.sin(declination)]
    result = {
        'input':{'ra_deg':ra,'dec_deg':dec,'trigger_time_utc':trigger_time_utc},
        'convention':'gwak_ra_minus_gmst_iau2006',
        'ut1_mode':'UTC approximation' if ut1_minus_utc_seconds is None else 'supplied UT1-UTC',
        'ut1_minus_utc_seconds':dut1,'tai_minus_utc_seconds':dat,
        'julian_date_midnight':jdn-.5,'ut1_fraction_of_day':ut1_fraction,
        'tt_fraction_of_day':tt_fraction,'tt_centuries_from_j2000':tt_centuries,
        'earth_rotation_angle_rad':era,'gmst_offset_arcsec':offset_arcsec,
        'gmst_rad':gmst,'earth_phi_rad':phi,'earth_declination_rad':declination,
        'earth_polar_rad':theta,'earth_phi_deg':math.degrees(phi),
        'earth_declination_deg':dec,'earth_polar_deg':math.degrees(theta),
        'source_unit_vector':vector,
        'gwak_sampling':{'fixed_phi_rad':phi,'fixed_declination_rad':declination,'sky_window_deg':width},
    }
    if temp_plot:
        _show_temporary_figure(_position_figure(result, name="Event"))
    return result



# Earth-fixed unit arm vectors, from LALDetectors.h; same constants as ml4gw.
# https://lscsoft.docs.ligo.org/lalsuite/7.26/lal/_l_a_l_detectors_8h_source.html
DETECTOR_ARMS = {
    'H1': ((-0.22389266154, 0.79983062746, 0.55690487831),
           (-0.91397818574, 0.02609403989, -0.40492342125)),
    'L1': ((-0.95457412153, -0.14158077340, -0.26218911324),
           (0.29774156894, -0.48791033647, -0.82054461286)),
}


def antenna_patterns(position, *, polarization_rad=0.0, temp_plot=False):
    """Return H1/L1 F_plus, F_cross and sqrt(F_plus**2 + F_cross**2).

    position: output dictionary from EITHER event_to_gwak function in this file.
    polarization_rad: psi in radians in the ml4gw/LAL polarization convention.
        Zero is a reference basis, not a measurement of the event polarization.

    Evaluates the response at the exact supplied direction (the window center),
    not an average over sky_window_deg. No additional time conversion is done.
    Uses only Python arithmetic: d=(x*x - y*y)/2, F_A = d:e_A.
    These are dimensionless long-wavelength antenna factors, not strain, SNR,
    detection efficiency, or detector-noise weighting. Arrival delays are
    separate. response_amplitude is independent of psi, unlike F_plus/F_cross.

    temp_plot=True opens an unsaved window; close it to return the same result.
    Requires a local interactive Matplotlib backend. Default False does no plotting.
    """
    if not isinstance(position, Mapping):
        raise ValueError('position must be the output dictionary of a sky conversion')
    if position.get('convention', CONVENTION) != CONVENTION:
        raise ValueError('Unsupported sky-coordinate convention')
    try:
        phi = _number(position['earth_phi_rad'], 'earth_phi_rad')
        dec = _number(position['earth_declination_rad'], 'earth_declination_rad')
    except KeyError as exc:
        raise ValueError('position needs earth_phi_rad and earth_declination_rad') from exc
    if not 0 <= phi < 2 * math.pi or not -math.pi/2 <= dec <= math.pi/2:
        raise ValueError('Earth phi must be in [0,2*pi), declination in [-pi/2,pi/2]')
    psi = _number(polarization_rad, 'polarization_rad') % math.pi
    # e_theta (southward) and e_phi (eastward), perpendicular to source direction.
    # theta = pi/2 - dec. Match ml4gw's m/n ordering and signs explicitly.
    u = (math.cos(phi)*math.sin(dec), math.sin(phi)*math.sin(dec), -math.cos(dec))
    v = (-math.sin(phi), math.cos(phi), 0.0)
    m = tuple(-a*math.sin(psi)-b*math.cos(psi) for a, b in zip(u, v))
    n = tuple(-a*math.cos(psi)+b*math.sin(psi) for a, b in zip(u, v))
    def dot(a, b):
        return math.fsum(x*y for x, y in zip(a, b))
    detectors = {}
    for ifo, (x, y) in DETECTOR_ARMS.items():
        xm, xn, ym, yn = dot(x,m), dot(x,n), dot(y,m), dot(y,n)
        f_plus = 0.5 * (xm*xm-xn*xn-ym*ym+yn*yn)
        f_cross = xm*xn-ym*yn
        detectors[ifo] = {'F_plus': f_plus, 'F_cross': f_cross,
                          'response_amplitude': math.hypot(f_plus, f_cross)}
    result = {'earth_phi_rad': phi, 'earth_declination_rad': dec,
            'polarization_rad': psi, 'convention': CONVENTION,
            'polarization_convention': 'ml4gw/LAL',
            'response_definition': 'sqrt(F_plus**2 + F_cross**2)',
            'detectors': detectors}
    if temp_plot:
        plot_antenna_patterns(position, polarization_rad=polarization_rad, temp_plot=True)
    return result


def _show_temporary_figure(fig):
    """Display an in-memory figure without saving a PNG, PDF or JSON file."""
    import matplotlib.pyplot as plt
    if fig.canvas.required_interactive_framework is None:
        plt.close(fig)
        raise RuntimeError(
            'temp_plot=True needs an interactive Matplotlib backend (MacOSX, TkAgg or QtAgg). '
            'Run in a local Python session with GUI support; Agg/inline and a plain SSH '
            'terminal cannot open a desktop window. No plot files were saved.')
    try:
        plt.show(block=True)
    finally:
        plt.close(fig)


def _antenna_figure(position, *, polarization_rad=0.0, name='Event'):
    """Build the antenna figure without writing files or changing the backend."""
    import matplotlib.pyplot as plt
    import numpy as np
    response = antenna_patterns(position, polarization_rad=polarization_rad)
    phis = np.linspace(0, 2*math.pi, 145)
    decs = np.linspace(-math.pi/2, math.pi/2, 73)
    keys = ('F_plus', 'F_cross', 'response_amplitude')
    labels = (r'$F_+$', r'$F_\times$', r'$\sqrt{F_+^2+F_\times^2}$')
    maps = np.empty((2, 3, len(decs), len(phis)))
    for j, dec in enumerate(decs):
        for i, phi in enumerate(phis):
            values = antenna_patterns({'earth_phi_rad': float(phi) % (2*math.pi),
                                       'earth_declination_rad': float(dec)},
                                      polarization_rad=polarization_rad)
            for row, ifo in enumerate(('H1', 'L1')):
                for col, key in enumerate(keys):
                    maps[row, col, j, i] = values['detectors'][ifo][key]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for row, ifo in enumerate(('H1', 'L1')):
        for col, key in enumerate(keys):
            ax = axes[row, col]
            im = ax.pcolormesh(np.degrees(phis), np.degrees(decs), maps[row,col],
                               shading='auto', cmap='viridis' if col == 2 else 'RdBu_r',
                               vmin=0 if col == 2 else -1, vmax=1)
            ax.scatter(math.degrees(response['earth_phi_rad']),
                       math.degrees(response['earth_declination_rad']),
                       marker='*', s=160, c='white', edgecolors='black', linewidths=.8)
            event_phi = math.degrees(response['earth_phi_rad'])
            event_dec = math.degrees(response['earth_declination_rad'])
            # Put exact center values beside the star, independent of map resolution.
            # Move inward for sources near the edges of the map.
            dx = 12 if event_phi < 180 else -12
            dy = 22 if event_dec < 50 else -28
            value = response['detectors'][ifo][key]
            ax.annotate(f'{labels[col]} = {value:+.6f}' if col < 2 else
                        f'{labels[col]} = {value:.6f}',
                        xy=(event_phi, event_dec), xytext=(dx, dy),
                        textcoords='offset points', ha='left' if dx > 0 else 'right',
                        va='bottom' if dy > 0 else 'top', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                                  edgecolor='0.3', alpha=0.95),
                        arrowprops=dict(arrowstyle='-', color='black', lw=0.8))
            ax.set(xlabel='Earth-relative azimuth phi (deg)', ylabel='Declination (deg)',
                   title=f'{ifo}: {labels[col]}',
                   xlim=(0,360), ylim=(-90,90))
            fig.colorbar(im, ax=ax, label='Dimensionless response')
    fig.suptitle(f'{name} | H1 / L1 antenna patterns | psi = '
                 f'{math.degrees(response["polarization_rad"]):.2f} deg\n'
                 'Star: event direction. Amplitude is independent of polarization angle.')
    return fig, response


def plot_antenna_patterns(position, *, polarization_rad=0.0,
                          output='antenna_patterns', name='Event', temp_plot=False):
    """Plot H1/L1 maps, saving PNG/PDF/JSON unless temp_plot=True.

    Temporary mode ignores output, opens an unsaved window, and waits until it
    is closed. Both modes return the same response dictionary.
    """
    import matplotlib.pyplot as plt
    fig, response = _antenna_figure(position, polarization_rad=polarization_rad, name=name)
    if temp_plot:
        _show_temporary_figure(fig)
        return response
    base = Path(output)
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf'):
        fig.savefig(str(base)+suffix, dpi=180)
    plt.close(fig)
    Path(str(base)+'.json').write_text(json.dumps(response, indent=2)+'\n')
    return response


def resolve_sky_position(*, fixed_phi_rad=None, fixed_declination_rad=None,
                         event_ra_deg=None, event_dec_deg=None, event_time_utc=None,
                         sky_window_deg=DEFAULT_SKY_WINDOW_DEG,
                         fixed_ra=None, fixed_dec=None):
    """Resolve exactly one input mode: event, Earth-relative, legacy, or all-sky."""
    legacy = fixed_ra is not None or fixed_dec is not None
    earth = fixed_phi_rad is not None or fixed_declination_rad is not None
    event = any(v is not None for v in (event_ra_deg, event_dec_deg, event_time_utc))
    if sum((legacy, earth, event)) > 1:
        raise ValueError('Choose only one sky input mode: event, fixed Earth angles, or legacy aliases')
    if event:
        if any(v is None for v in (event_ra_deg, event_dec_deg, event_time_utc)):
            raise ValueError('event_ra_deg, event_dec_deg and event_time_utc are all required')
        return event_to_gwak(event_ra_deg, event_dec_deg, event_time_utc,
                             sky_window_deg=sky_window_deg)
    if legacy:
        warnings.warn('fixed_ra/fixed_dec mean Earth-relative radians, not celestial RA/Dec. '
                      'Use fixed_phi_rad/fixed_declination_rad or event_* inputs.',
                      DeprecationWarning, stacklevel=2)
        fixed_phi_rad, fixed_declination_rad = fixed_ra, fixed_dec
    phi, dec, width = validate_sky_window(fixed_phi_rad, fixed_declination_rad, sky_window_deg)
    return {'convention': CONVENTION if phi is not None else 'uniform_sky',
            'gwak_sampling': {'fixed_phi_rad': phi, 'fixed_declination_rad': dec,
                              'sky_window_deg': width}}


class SkyPositionMixin:
    """Small reusable sampler interface; torch is needed only when sampling."""
    def configure_sky_position(self, **kwargs):
        self.sky_position = resolve_sky_position(**kwargs)
        for name, value in self.sky_position['gwak_sampling'].items():
            setattr(self, name, value)

    @property
    def fixed_ra(self):
        """Deprecated mutable alias retained for archived evaluation scripts."""
        return self.fixed_phi_rad

    @fixed_ra.setter
    def fixed_ra(self, value):
        self.fixed_phi_rad = value
        self.sky_position = None  # explicit mutation invalidates conversion provenance

    @property
    def fixed_dec(self):
        return self.fixed_declination_rad

    @fixed_dec.setter
    def fixed_dec(self, value):
        self.fixed_declination_rad = value
        self.sky_position = None

    def _sample_sky_location(self, batch_size):
        import torch
        phi, dec, width = validate_sky_window(
            self.fixed_phi_rad, self.fixed_declination_rad, self.sky_window_deg)
        if phi is None:
            return self.ra_prior.sample((batch_size,)), self.dec_prior.sample((batch_size,))
        window = math.radians(width)
        # Keep two separate RNG draws in this order for legacy reproducibility.
        phis = (phi + (torch.rand(batch_size) - 0.5)*window) % (2*torch.pi)
        declinations = dec + (torch.rand(batch_size) - 0.5)*window
        return phis, declinations


def read_icecube_event(csv_path, event_name=None):
    """Read one ICECAT-1 row; check START against EVENTMJD interpreted as UTC.

    Uses catalog best-fit RA/DEC, not an IceCube detector-local zenith/azimuth.
    The catalog's error bounds are preserved as metadata, not used as a GWAK box.
    """
    path = Path(csv_path)
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {'NAME', 'RA', 'DEC', 'START', 'EVENTMJD'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'ICECAT CSV requires {sorted(required)}')
        rows = [row for row in reader if event_name is None or row['NAME'] == event_name]
    if not rows:
        raise ValueError(f'No matching IceCube event: {event_name!r}')
    if event_name is not None and len(rows) != 1:
        raise ValueError(f'Ambiguous event name: {event_name}')
    row = rows[0]
    utc = _utc_time(row['START'], 'isot')
    mjd = _utc_time(row['EVENTMJD'], 'mjd')
    difference = float(abs((utc - mjd).to_value('sec')))
    if difference > 0.001:
        raise ValueError(f'START and EVENTMJD disagree by {difference:.6g} seconds')
    return {'name': row['NAME'], 'ra_deg': _number(row['RA'], 'RA'),
            'dec_deg': _number(row['DEC'], 'DEC'), 'trigger_time': utc.utc.isot,
            'catalog': {'path': str(path.resolve()),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'row': row, 'start_mjd_difference_seconds': difference,
                        'time_scale': 'UTC'}}


def _position_figure(result, *, name='Event'):
    """Build from an existing result, including the independent arithmetic path."""
    import numpy as np
    import matplotlib.pyplot as plt
    sky_window_deg = result['gwak_sampling']['sky_window_deg']
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(121, projection='3d')
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 37), np.linspace(0, np.pi, 19))
    ax.plot_wireframe(np.sin(v)*np.cos(u), np.sin(v)*np.sin(u), np.cos(v),
                      color='#8fa9b8', linewidth=.45, alpha=.4)
    n = result['source_unit_vector']
    ax.quiver(0, 0, 0, *n, color='#d65a31', linewidth=3, arrow_length_ratio=.12)
    ax.scatter(*n, color='#d65a31', s=55)
    for vec, label in [((1,0,0), '+x: Greenwich'), ((0,1,0), '+y: 90 E'),
                       ((0,0,1), '+z: north pole')]:
        ax.quiver(0,0,0,*vec,color='#304c61',alpha=.7,arrow_length_ratio=.08)
        ax.text(*(1.18*np.array(vec)), label, fontsize=8)
    ax.set(xlim=(-1.4,1.4), ylim=(-1.4,1.4), zlim=(-1.4,1.4))
    ax.set_box_aspect((1,1,1)); ax.set_axis_off()
    ax.view_init(elev=23, azim=result['earth_phi_deg']-40)
    ax.set_title('Source direction in Earth-fixed axes', fontsize=12)
    ax2 = fig.add_subplot(122)
    phi, dec, half = result['earth_phi_deg'], result['earth_declination_deg'], sky_window_deg/2
    ax2.scatter([0], [0], s=65, color='#d65a31', label='Best-fit event direction')
    from matplotlib.patches import Rectangle
    ax2.add_patch(Rectangle((-half,-half), sky_window_deg, sky_window_deg,
                           facecolor='#61a5c2', edgecolor='#174b65', alpha=.35,
                           label=f'{sky_window_deg:g} degree sampling box'))
    extent = max(sky_window_deg*.8, .1)
    ax2.set(xlim=(-extent,extent), ylim=(-extent,extent),
            xlabel='Azimuth offset from center (degrees)',
            ylabel='Declination offset from center (degrees)')
    ax2.set_aspect('equal'); ax2.grid(alpha=.2); ax2.legend(loc='upper right', fontsize=8)
    ax2.set_title(f'phi = {phi:.4f} deg; dec = {dec:.4f} deg\n'
                  f'polar theta = {result["earth_polar_deg"]:.4f} deg',fontsize=12)
    fig.suptitle(f'{name} | {result["input"]["trigger_time_utc"]} UTC', fontsize=15)
    fig.supxlabel('RA - GMST (IAU2006) convention; vector points toward source.\n'
                  'Box is the GWAK sampling choice, not the event localization uncertainty.', fontsize=9)
    return fig


def plot_event_position(ra_deg, dec_deg, trigger_time, *, output='event_sky',
                        name='Event', time_format='isot',
                        sky_window_deg=DEFAULT_SKY_WINDOW_DEG, temp_plot=False):
    """Convert and plot; save PNG/PDF/JSON unless temp_plot=True.

    Temporary mode ignores output, opens an unsaved window, and waits until it
    is closed. Both modes return the conversion result with the event name.
    """
    import matplotlib.pyplot as plt
    result = event_to_gwak(ra_deg, dec_deg, trigger_time, time_format=time_format,
                           sky_window_deg=sky_window_deg)
    fig = _position_figure(result, name=name)
    result['name'] = name
    if temp_plot:
        _show_temporary_figure(fig)
        return result
    base = Path(output)
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf'):
        fig.savefig(str(base)+suffix, dpi=180)
    plt.close(fig)
    Path(str(base)+'.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ra-deg', type=float)
    parser.add_argument('--dec-deg', type=float)
    parser.add_argument('--trigger-time')
    parser.add_argument('--time-format', choices=['isot','mjd'], default='isot')
    parser.add_argument('--sky-window-deg', type=float, default=DEFAULT_SKY_WINDOW_DEG)
    parser.add_argument('--csv', type=Path, help='ICECAT-1 CSV; uses first row unless --event is set')
    parser.add_argument('--event', help='Exact NAME field in the IceCube CSV')
    parser.add_argument('--polarization-rad', type=float, default=0.0, help='Reference psi in radians (default 0)')
    parser.add_argument('--temp-plot', action='store_true', help='Show unsaved windows instead of writing files')
    parser.add_argument('--antenna-patterns', action='store_true', help='Also save H1/L1 antenna maps')
    parser.add_argument('--name', default='Event')
    parser.add_argument('--output', default='event_sky', help='Output filename stem')
    args = parser.parse_args()
    catalog = None
    if args.csv:
        if any(x is not None for x in (args.ra_deg,args.dec_deg,args.trigger_time)):
            parser.error('Use either --csv or explicit RA/Dec/time')
        event = read_icecube_event(args.csv, args.event)
        args.ra_deg,args.dec_deg,args.trigger_time = event['ra_deg'],event['dec_deg'],event['trigger_time']
        args.time_format='isot'; args.name=event['name']; catalog=event['catalog']
    elif any(x is None for x in (args.ra_deg,args.dec_deg,args.trigger_time)):
        parser.error('Provide --csv or all of --ra-deg, --dec-deg, --trigger-time')
    result = plot_event_position(args.ra_deg,args.dec_deg,args.trigger_time,output=args.output,
                                name=args.name,time_format=args.time_format,sky_window_deg=args.sky_window_deg,
                                temp_plot=args.temp_plot)
    if catalog:
        result['catalog']=catalog
    result['antenna_patterns'] = antenna_patterns(result, polarization_rad=args.polarization_rad)
    if args.antenna_patterns:
        plot_antenna_patterns(result, polarization_rad=args.polarization_rad,
                              output=args.output+'_antenna', name=args.name, temp_plot=args.temp_plot)
    if not args.temp_plot:
        Path(args.output+'.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
