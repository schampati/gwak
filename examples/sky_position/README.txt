EVENT COORDINATES FOR GWAK
==========================

1. Normal training interface (SignalDataloader)

Under data.init_args in a training configuration, supply:

    event_ra_deg: 138.47
    event_dec_deg: -1.94
    event_time_utc: "2011-05-14T01:32:22.654109"
    sky_window_deg: 1.0

These example values are ICECAT-1 event IC110514A. Omit sky_window_deg to
use the same 1-degree FULL-WIDTH box as the prior studies: +/-0.5 degree
in phi and declination. This is not a circular cone/radius, equal-solid-angle
patch, or automatically the event's localization uncertainty. Boxes that
cross a pole are rejected; azimuth wraps through zero.

The loader converts once, at initialization. It freezes the Earth-relative
center at the event trigger time. Training/background timestamps never rotate
that center. Polarization and waveform parameters may still vary as before.

Alternatively, supply angles already converted to the Earth frame:

    fixed_phi_rad: 4.257267356544483
    fixed_declination_rad: -0.03385938748868999
    sky_window_deg: 1.0

Do not combine these two modes. All three event fields are required together.
No position fields means the existing all-sky sampler. Legacy fixed_ra and
fixed_dec still work as deprecated aliases for Earth-relative RADIANS; they
never meant celestial RA and Dec. Old scripts that mutate these aliases still
work. Old study configurations and trained models are not rewritten.

This interface applies to SignalDataloader. Archived custom multi-target
subclasses override sky sampling; their frozen target lists remain their inputs.
The example sg150_icecube_event.json is a full SG150/Q9 configuration with the
new event fields and the packaged gwak.data.fixed_sine_gaussian prior. It requires the
usual GWAK runtime/data paths. No training has been launched for this event.

2. Standalone function and plot, independent of the training pipeline

The separate file gwak/sky_position_antenna_patterns.py can be copied anywhere and run directly.
It has no GWAK import and does not need PyTorch for conversion or plotting.

From the repository root:

    python3 -m venv .venv-sky
    .venv-sky/bin/pip install -r scripts/requirements-sky.txt
    .venv-sky/bin/python gwak/sky_position_antenna_patterns.py \
      --csv examples/sky_position/icecube_event.csv \
      --event IC110514A --output examples/sky_position/IC110514A

This saves .png, .pdf and .json. Omit --event to choose the first catalog row.
For another event use --ra-deg, --dec-deg and --trigger-time instead of --csv.
For a numeric UTC MJD use --time-format mjd --trigger-time 55695.06415108922.

Python usage (repository on PYTHONPATH):

    from gwak.sky_position_antenna_patterns import event_to_gwak, plot_event_position
    result = event_to_gwak(138.47, -1.94, "2011-05-14T01:32:22.654109")
    sampling_kwargs = result["gwak_sampling"]
    plot_event_position(138.47, -1.94, "2011-05-14T01:32:22.654109",
                        name="IC110514A", output="icecube_example")

Output includes phi, declination, polar theta, a source unit vector, input time,
GMST, UT1-UTC, IERS status and library versions. Phi is from Earth-fixed +x
(Greenwich) toward +y (90 E); theta is down from +z (north). The vector points
TOWARD the source; the wave propagates in the opposite direction. This is not
an IceCube-local zenith/azimuth or a distance to a source.

3. Conversion convention and time handling

For compatibility with prior GWAK experiments:
    phi = (RA - GMST_IAU2006(trigger_time)) modulo 2*pi
    Earth declination = catalog declination
    theta = pi/2 - Earth declination

Astropy Time(..., scale="utc") supplies mean Greenwich sidereal time, with
UT1-UTC from bundled IERS data. No internet is accessed by the converter.
An out-of-range epoch fails and requests updated astropy-iers-data; predictions
are recorded via iers_status. Updating IERS tables can change the last digits.

This is the recorded simplified J2000-RA-minus-GMST convention, NOT a full
ICRS/J2000-to-ITRS transformation. Precession, nutation and polar motion are not
applied to the source. A future change to a full frame transform should be an
explicit scientific-convention change, not a silent variable rename.

The IceCube CSV reader uses RA/DEC in degrees and START as UTC, cross-checks
EVENTMJD within 1 ms, and records the source file checksum and catalog row.
IC110514A gives phi=243.9234518 deg, dec=-1.94 deg, theta=91.94 deg.
The example window is a training choice; this event's asymmetric 90% catalog
error rectangle is larger and is preserved in JSON metadata only.

4. Server installation

First run python scripts/package_sky_update.py from the repository root.
The generated scripts/install_sky_position.sh is a self-contained installer.
This migration tool targets our earlier directional server checkout, not an
arbitrary upstream version. The generated shell payload is excluded from Git.
Upload it to ~/gwak-full-run/install_sky_position.sh using the usual scp route.
On ldas-pcdev2:

    bash ~/gwak-full-run/install_sky_position.sh ~/gwak-full-run

It checks payload hashes, applies only known dataloader edits, validates the
candidate using the existing PyTorch Apptainer image on CPU, then backs up and
installs. If the source differs unexpectedly or the preflight fails, original
source files stay unchanged. It prints the backup directory and installed hashes.
The default image is ~/apptainer-images/pytorch-2.3.1-cuda12.1.sif. The converter
needs Astropy >=6; prior study site-packages already used Astropy 6.0.1.
No worker is started/stopped, no new model is trained, and archived experiments
are untouched. Running Python processes retain their already-imported code.

5. Validation and sources

Local tests cover historical GRB coordinates, an independent GMST polynomial,
a 12-hour time shift, UTC/MJD agreement, angle wrapping, bounds, input-mode
conflicts, old/new fixed-seed sampler equivalence, all-sky regression, actual
constructor wiring, catalog validation, output plots and idempotent patching.
verify_sky_position.py imports the real dataloader and projects impulses through
ml4gw on CPU. It intercepts base initialization to avoid background/GPU access;
it is not a full GPU training smoke test.

https://docs.astropy.org/en/stable/api/astropy.time.Time.html#astropy.time.Time.sidereal_time
https://docs.astropy.org/en/stable/utils/iers.html
https://icecube.wisc.edu/data-releases/2023/04/icecat-1-the-icecube-event-catalog-of-alert-tracks/
https://gcn.nasa.gov/missions/icecube
Local catalog definitions: ../dataverse_files/a_README (relative to repository).

6. H1/L1 antenna response

Both conversion functions and antenna calculations now live in
gwak/sky_position_antenna_patterns.py. sky_position.py remains an import/CLI
compatibility wrapper so training and archived scripts keep working.

    from gwak.sky_position_antenna_patterns import (
        event_to_gwak, event_to_gwak_from_first_principles,
        antenna_patterns, plot_antenna_patterns)
    position = event_to_gwak(138.47, -1.94, "2011-05-14T01:32:22.654109")
    response = antenna_patterns(position, polarization_rad=0.0)
    print(response["detectors"]["H1"])
    print(response["detectors"]["L1"])
    plot_antenna_patterns(position, output="icecube_antenna", name="IC110514A")

A position returned by event_to_gwak_from_first_principles works identically.
The antenna function uses only Python arithmetic with LAL arm vectors.
It returns F_plus, F_cross, and response_amplitude=sqrt(F_plus^2+F_cross^2).
Psi defaults to zero as a reference basis; event coordinates alone do not
determine physical polarization. The amplitude is independent of psi.
Responses are at the exact center, not averaged over the sampling window.
These are long-wavelength, dimensionless geometric factors, not efficiency,
strain, SNR or PSD-weighted sensitivity. Time delays are a separate effect.

Add --antenna-patterns to the standalone command to save six sky maps
(two detectors x F_plus, F_cross, amplitude), with the event marked.
Use --polarization-rad to change the reference polarization angle.

Primary response equations / detector geometry:
https://lscsoft.docs.ligo.org/lalsuite/7.26/lal/_det_response_8c_source.html
https://lscsoft.docs.ligo.org/lalsuite/7.26/lal/_l_a_l_detectors_8h_source.html

7. Mathematical walkthrough

The eight-page gwak/sky_position_antenna_patterns_math.pdf sits beside the
calculation module. It follows the implemented calendar, time-scale, rotation,
coordinate, polarization and detector-tensor formulas, with a worked IceCube
example and primary references. Rebuild using:

    python scripts/build_sky_antenna_math_pdf.py

Antenna maps label each event marker with its directly calculated response
to six decimal places; the numerical labels are not sampled from the map grid.

8. Temporary plot windows (no output files)

All three numerical functions default to temp_plot=False: they only return
values. For a window that stays open until you close it:

    position = event_to_gwak(138.47, -1.94,
                            "2011-05-14T01:32:22.654109", temp_plot=True)
    response = antenna_patterns(position, temp_plot=True)

The independent event_to_gwak_from_first_principles accepts temp_plot=True too.
It plots its own calculated position without calling the Astropy calculation.
The optional plot uses Matplotlib/NumPy; calculation alone is standard-library.

The same option works on plot_event_position and plot_antenna_patterns; in
that mode output is ignored and neither figures nor JSON are saved. Their
default remains the explicit save-to-file behavior.

Command-line equivalent: add --temp-plot (and --antenna-patterns if desired).
Run on your local Mac with an interactive Matplotlib backend. In notebooks,
select a GUI backend if you want a separate desktop window. Headless Agg or
inline backends raise a clear error instead of silently failing or saving files.
The function pauses until the displayed windows close, then returns its result.

Matplotlib display semantics:
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html

9. Run an editable example

    python examples/sky_position/run_example.py

Edit the RA, Dec and UTC time inside run_example. Both conversions run and are
compared with a shared UT1-UTC input; position and antenna windows are unsaved.
Use run_example(temp_plot=False) when calling from a headless environment.
