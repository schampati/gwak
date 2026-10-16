"""Compatibility entry point; both implementations now live in sky_position_antenna_patterns.py."""
import argparse
import json
try:
    from .sky_position_antenna_patterns import event_to_gwak_from_first_principles
except ImportError:
    # Also support running this file directly with python.
    from sky_position_antenna_patterns import event_to_gwak_from_first_principles

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ra-deg',type=float,required=True);p.add_argument('--dec-deg',type=float,required=True)
    p.add_argument('--trigger-time',required=True);p.add_argument('--dut1',type=float)
    p.add_argument('--tai-minus-utc',type=float);p.add_argument('--sky-window-deg',type=float,default=1.)
    a=p.parse_args()
    print(json.dumps(event_to_gwak_from_first_principles(a.ra_deg,a.dec_deg,a.trigger_time,
        ut1_minus_utc_seconds=a.dut1,tai_minus_utc_seconds=a.tai_minus_utc,sky_window_deg=a.sky_window_deg),indent=2))
