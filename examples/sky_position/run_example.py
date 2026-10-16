"""Edit the event below and press Run. Close each plot to continue.

Use run_example(temp_plot=False) for calculations without GUI windows or files.
Run from a checkout or an installed GWAK package.
"""
from pathlib import Path
import sys

# Allow direct execution from an IDE without installing the training stack.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gwak.sky_position_antenna_patterns import (
    event_to_gwak,
    event_to_gwak_from_first_principles,
    antenna_patterns,
)


def run_example(temp_plot=True):
    """Compare both conversions and show H1/L1 responses for IC110514A."""
    ra_deg = 138.47
    dec_deg = -1.94
    trigger_time_utc = '2011-05-14T01:32:22.654109'
    position = event_to_gwak(ra_deg, dec_deg, trigger_time_utc, temp_plot=temp_plot)
    # UT1-UTC is an observed Earth-rotation correction, shared as an input.
    # The independent path computes the calendar, time scales and GMST itself.
    arithmetic = event_to_gwak_from_first_principles(
        ra_deg, dec_deg, trigger_time_utc,
        ut1_minus_utc_seconds=position['ut1_minus_utc_seconds'],
        temp_plot=temp_plot,
    )
    delta = max(abs(a-b) for a,b in zip(position['source_unit_vector'],
                                       arithmetic['source_unit_vector']))
    if delta > 1e-10:
        raise AssertionError(f'Conversions disagree: vector difference {delta}')
    response = antenna_patterns(position, temp_plot=temp_plot)
    print('GWAK sampling parameters:', position['gwak_sampling'])
    print('Maximum source-vector difference:', delta)
    for ifo, values in response['detectors'].items():
        print(ifo, values)
    return {'astropy': position, 'arithmetic': arithmetic, 'antenna': response}


if __name__ == '__main__':
    run_example()
