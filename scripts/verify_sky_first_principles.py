"""Compare independent arithmetic with the existing Astropy-based GWAK helper.

ONLY this verification harness calls Astropy indirectly through event_to_gwak.
The direct implementation receives just the measured DUT1 scalar, not any
reference sidereal time, converted angle, Julian date or output vector.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from gwak.sky_position_antenna_patterns import event_to_gwak
from gwak.sky_position_antenna_patterns import event_to_gwak_from_first_principles


def wrapped_difference(a,b):
    return math.atan2(math.sin(a-b),math.cos(a-b))


def compare(name,ra,dec,utc):
    reference=event_to_gwak(ra,dec,utc)
    corrected=event_to_gwak_from_first_principles(ra,dec,utc,
                       ut1_minus_utc_seconds=reference['ut1_minus_utc_seconds'])
    approximate=event_to_gwak_from_first_principles(ra,dec,utc)
    errors={key:abs(wrapped_difference(corrected[key],reference[key]))
            for key in ['gmst_rad','earth_phi_rad','earth_declination_rad','earth_polar_rad']}
    vector_error=max(abs(a-b) for a,b in zip(corrected['source_unit_vector'],reference['source_unit_vector']))
    assert max(errors.values())<1e-10,(name,errors)
    assert vector_error<1e-10,(name,vector_error)
    return {'name':name,'utc':utc,'ra_deg':ra,'dec_deg':dec,
            'reference_phi_deg':reference['earth_phi_deg'],
            'direct_phi_deg':corrected['earth_phi_deg'],
            'polar_deg':corrected['earth_polar_deg'],
            'ut1_minus_utc_seconds':reference['ut1_minus_utc_seconds'],
            'tai_minus_utc_seconds':corrected['tai_minus_utc_seconds'],
            'phi_error_rad':errors['earth_phi_rad'],
            'phi_error_arcsec':math.degrees(errors['earth_phi_rad'])*3600,
            'max_angle_error_rad':max(errors.values()),'max_vector_component_error':vector_error,
            'utc_approx_phi_error_arcsec':math.degrees(wrapped_difference(
                approximate['earth_phi_rad'],reference['earth_phi_rad']))*3600}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    with args.csv.open(newline='',encoding='utf-8-sig') as f:catalog=list(csv.DictReader(f))
    records=[compare(r['NAME'],float(r['RA']),float(r['DEC']),r['START']) for r in catalog]
    grbs=[compare('GRB 190114C',54.50484,-26.94637,'2019-01-14T20:57:03'),
          compare('GRB 130427A',173.1362,27.7129,'2013-04-27T07:47:57')]
    # Include changes of day/year/leap-second offset and RA wrap.
    boundaries=[compare('boundary '+t,ra,0,t) for t,ra in [
        ('2000-01-01T12:00:00',0),('2012-06-30T23:59:59',359.999),
        ('2012-07-01T00:00:00',0),('2016-12-31T23:59:59',0),('2017-01-01T00:00:00',0)]]
    all_records=records+grbs+boundaries
    result={'result':'PASS','catalog':str(args.csv.resolve()),'catalog_events':len(records),
            'grb_events':len(grbs),'boundary_checks':len(boundaries),'total_comparisons':len(all_records),
            'tolerance_rad':1e-10,
            'maximum_angle_error_rad':max(r['max_angle_error_rad'] for r in all_records),
            'maximum_phi_error_arcsec':max(r['phi_error_arcsec'] for r in all_records),
            'comparison_inputs':'Shared measured UT1-UTC only; leap seconds from independent IERS table. No reference angles/time conversions passed to direct function.',
            'scope':'Same RA-minus-GMST convention; does not validate a full ICRS-to-ITRS transform.',
            'examples':[records[0]]+grbs,'records':all_records}
    (args.output_dir/'first_principles_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.output_dir/'first_principles_verification.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_records[0]));writer.writeheader();writer.writerows(all_records)
    lines=['DIRECT CALCULATION VS EXISTING GWAK FUNCTION','===========================================','',
           f'PASS: {len(records)} IceCube rows + {len(grbs)} GRBs + {len(boundaries)} boundary cases.',
           f'Maximum angle difference: {result["maximum_angle_error_rad"]:.3e} rad.',
           f'Maximum azimuth difference: {result["maximum_phi_error_arcsec"]:.3e} arcsec.',
           'Tolerance: 1e-10 rad. All computations use double precision.','',
           'Event           Reference phi (deg)    Direct phi (deg)       Polar (deg)']
    for r in result['examples']:
        lines.append(f'{r["name"]:15s} {r["reference_phi_deg"]:20.10f} {r["direct_phi_deg"]:20.10f} {r["polar_deg"]:15.6f}')
    lines+=['','WHAT IS INDEPENDENT',
            'Gregorian date -> Julian day, leap-second lookup, UTC -> UT1/TT, Earth',
            'rotation angle, sidereal-time polynomial, source rotation and unit vector',
            'are computed in a separate standard-library-only Python implementation.',
            'It does not import Astropy, ERFA, NumPy, GWAK coordinate helpers or call',
            'their time/coordinate transformations. The verifier calls both paths.','',
            'WHAT IS SHARED',
            'Both calculations implement the same IAU 2006 GMST convention. For exact',
            'comparison the verifier supplies the same measured UT1-UTC scalar from',
            'the reference IERS data. This is an input measurement, not a computed',
            'sky angle. Earth rotation irregularities cannot be inferred from UTC alone.','',
            'IF UT1 IS APPROXIMATED BY UTC',
            'The direct function defaults explicitly to that approximation when DUT1',
            'is omitted. Signed azimuth errors in the example events:']
    for r in result['examples']:lines.append(f'  {r["name"]}: {r["utc_approx_phi_error_arcsec"]:+.6f} arcsec')
    lines+=['','LIMITS',
            'This checks our existing simplified RA-minus-GMST convention, including',
            'the time sign and units. It is not independent observational validation',
            'or a test of full ICRS/J2000-to-ITRS precession/nutation/polar motion.',
            'Literal leap-second timestamps ending in :60 are explicitly rejected;',
            'ordinary timestamps immediately before and after leap changes are tested.',
            'The bundled leap table expires 2027-06-28; later dates need a verified',
            'explicit TAI-UTC correction or a table update.','',
            'FORMULAS / PRIMARY REFERENCES',
            'UT1 = UTC + DUT1; TT = UTC + (TAI-UTC) + 32.184 seconds.',
            'ERA = 2*pi * [0.7790572732640 + 1.00273781191135448*d_UT1] mod 2*pi.',
            'GMST = ERA + [0.014506 + 4612.156534*T + 1.3915817*T^2',
            '       -0.00000044*T^3 -0.000029956*T^4 -0.0000000368*T^5] arcsec.',
            'd_UT1 is days from J2000; T is TT centuries from J2000.',
            'phi = (RA - GMST) mod 2*pi; theta = pi/2 - dec.',
            'n = (cos(dec)*cos(phi), cos(dec)*sin(phi), sin(dec)).',
            'https://raw.githubusercontent.com/liberfa/erfa/master/src/era00.c',
            'https://raw.githubusercontent.com/liberfa/erfa/master/src/gmst06.c',
            'https://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat']
    (args.output_dir/'FIRST_PRINCIPLES_CHECK.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:15]))


if __name__=='__main__':main()
