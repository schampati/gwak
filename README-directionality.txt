DIRECTIONAL GWAK: REVIEW ENTRY POINT
===================================

This contribution lives in Sujay Champati's fork, schampati/gwak, on the
directionality branch. It is a focused addition to the existing snapshot;
merging it into ML4GW/gwak dev or dev-andy is a separate review/integration step.

Review these files first:

  gwak/sky_position_antenna_patterns.py
    Astropy and independent arithmetic coordinate conversions, H1/L1 antenna
    coefficients, optional plots, and the fixed-direction sampler interface.
  gwak/train/dataloader.py
    Connects the interface to waveform projection. CCSN data is loaded only
    when CCSN is selected, allowing SG-only jobs without a CCSN waveform file.
  examples/sky_position/README.txt
    API, angular/time conventions, configuration, validation and deployment.
  examples/sky_position/run_example.py
    Edit RA, Dec and UTC trigger time and press Run; plots are unsaved windows.
  examples/sky_position/train_event.sh
    Launches the GWAK training CLI with the event inputs below. Default mode is
    a short encoder smoke run; train mode uses 25 epochs. See the example README.
  examples/sky_position/sg150_directional_event.json
    SG150/Q9 example; uses gwak/data/fixed_sine_gaussian.py, with no dependency
    on the untracked experiment drivers. Training requires the GWAK runtime,
    data location and GPU. This is an example, not a new trained result.
  gwak/sky_position_antenna_patterns_math.pdf
    Mathematical walkthrough of the implemented formulas, beside the code.
  examples/sky_position/IC110514A*.pdf (and .png)
    Example source position and labeled H1/L1 antenna maps.
  tests/test_sky_*.py, tests/test_antenna_patterns.py, tests/test_temp_plots.py
    CPU regression checks, numerical comparisons and installer transactions.

SCIENTIFIC CONTRACT

event_ra_deg + event_dec_deg + event_time_utc select the Earth-relative center
once at initialization. Canonical preconverted inputs are fixed_phi_rad and
fixed_declination_rad. The default sky_window_deg=1.0 is a FULL-WIDTH phi/dec
box (+/-0.5 degree in each), not a cone. Legacy fixed_ra/fixed_dec are deprecated
aliases for Earth-relative radians. With no fixed inputs, sampling is all-sky.

The preserved convention is phi=(RA-GMST_IAU2006) modulo 2*pi, with declination
unchanged. It is NOT the full ICRS-to-ITRS transform; precession, nutation and
polar motion are not applied to the source. The arithmetic version evaluates
established ERA/GMST formulas; it is not a derivation of Earth's rotation from
mechanics. Matching Astropy requires the same measured UT1-UTC correction.
Antenna coefficients use the long-wavelength detector-tensor contraction.

TESTING

For the lightweight conversion/plot API:
  python -m pip install -r scripts/requirements-sky.txt
  python examples/sky_position/run_example.py

For CPU regression tests, also install pytest, torch, jaxtyping and lalsuite,
and initialize the ml4gw submodule. From the repository root:
  MPLBACKEND=Agg PYTHONPATH=.:ml4gw python -m pytest -q tests

Installer tests use a fake Apptainer command to check transactions; they do not
validate a container or CUDA. scripts/verify_sky_position.py is the separate
real-dataloader/CPU-projection preflight for the existing server environment.

DEPLOYMENT AND ANDY'S INFRASTRUCTURE

Source changes are suitable for review/cherry-picking onto the new training
infrastructure. Nothing here changes Condor job submission, transfers datasets,
or assumes a home directory exists on a Condor execution point. Andy should
ship the package, configuration, dependencies and data using the new infra.

For our existing manually maintained ~/gwak-full-run checkout only:
  python scripts/package_sky_update.py
Then upload the generated scripts/install_sky_position.sh and run:
  bash install_sky_position.sh ~/gwak-full-run

The migration installer recognizes our earlier directional dataloader or an
already-updated version; it is not a general patcher for arbitrary upstream
versions. It runs a CPU preflight and keeps backups before installation.
The generated base64 shell payload and backups are excluded from Git.

Archived sweeps, trained models, results and unrelated local infrastructure
edits are not included in this change. Their original local files are retained.

VALIDATION OF THIS HANDOFF (2026-10-16)

53 CPU tests passed on macOS with Python 3.9.6, torch 2.3.1, Astropy 6.0.1,
NumPy 1.26.4 and LALSuite 7.26.4, using the pinned ml4gw checkout. This is a
lightweight test environment, not the project's declared Python 3.11 runtime.
No new GPU training or Condor execution was performed for this handoff.
The runnable example agreed between the two source vectors within 6.4e-15;
the fixed SG prior retained the original seeded samples for all other fields.
Third-party deprecation warnings and the sandbox's Astropy cache warning did
not affect the test results. The GUI path is mocked in headless tests.

The old examples/sky_position/sky_position*.py entry points forward to the
canonical package code so existing local launch paths continue to work.
