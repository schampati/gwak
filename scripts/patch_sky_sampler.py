"""Surgical, fail-closed update for the previously used SignalDataloader.

Accepts LF or CRLF blocks, preserves unrelated file content, and is idempotent.
The deployment installer backs up files before using this transformation.
"""
import re


def patch_dataloader(source):
    text = source.decode('utf-8')
    marker = 'from gwak.sky_position import SkyPositionMixin, resolve_sky_position'
    if marker in text:
        required = ['class SignalDataloader(SkyPositionMixin, GwakBaseDataloader):',
                    'event_time_utc: Optional[str] = None', 'self.configure_sky_position(',
                    'return self._sample_sky_location(batch_size)']
        if not all(x in text for x in required):
            raise ValueError('Partial sky-position update detected; inspect before replacing')
        return source

    def replace(old, new):
        nonlocal text
        pattern = re.escape(old).replace(r'\
', r'\r?\n')
        text, count = re.subn(pattern, lambda _: new, text)
        if count != 1:
            raise ValueError(f'Expected one matching block, found {count}: {old[:100]!r}')

    replace('from gwak.data.prior import FakeGlitchPrior',
            'from gwak.data.prior import FakeGlitchPrior\n'+marker)
    replace('class SignalDataloader(GwakBaseDataloader):',
            'class SignalDataloader(SkyPositionMixin, GwakBaseDataloader):')
    replace('''        fixed_ra: Optional[float] = None,
        fixed_dec: Optional[float] = None,
        sky_window_deg: float = 1.0,''','''        # Canonical Earth-relative angles are radians; event coordinates are degrees.
        fixed_phi_rad: Optional[float] = None,
        fixed_declination_rad: Optional[float] = None,
        event_ra_deg: Optional[float] = None,
        event_dec_deg: Optional[float] = None,
        event_time_utc: Optional[str] = None,
        sky_window_deg: float = 1.0,  # full box width: +/-0.5 deg by default
        fixed_ra: Optional[float] = None,  # deprecated Earth-phi alias, NOT celestial RA
        fixed_dec: Optional[float] = None,  # deprecated Earth-declination alias''')
    replace('''        super().__init__(*args, **kwargs)
        if (fixed_ra is None) != (fixed_dec is None):
            raise ValueError("fixed_ra and fixed_dec must either both be set or both be None")
        if fixed_ra is not None:
            if not np.isfinite(fixed_ra) or not 0 <= fixed_ra < 2 * np.pi:
                raise ValueError("fixed_ra must be finite and in the range [0, 2*pi)")
            if not np.isfinite(fixed_dec) or not -np.pi / 2 <= fixed_dec <= np.pi / 2:
                raise ValueError("fixed_dec must be finite and in the range [-pi/2, pi/2]")
            if not np.isfinite(sky_window_deg) or sky_window_deg <= 0:
                raise ValueError("sky_window_deg must be finite and greater than zero")
            half_window = np.deg2rad(sky_window_deg) / 2
            if fixed_dec - half_window < -np.pi / 2 or fixed_dec + half_window > np.pi / 2:
                raise ValueError("fixed_dec sky window must remain within [-pi/2, pi/2]")''','''        # Resolve once at initialization, before data/GPU access; never use noise times.
        self.configure_sky_position(
            fixed_phi_rad=fixed_phi_rad, fixed_declination_rad=fixed_declination_rad,
            event_ra_deg=event_ra_deg, event_dec_deg=event_dec_deg,
            event_time_utc=event_time_utc, sky_window_deg=sky_window_deg,
            fixed_ra=fixed_ra, fixed_dec=fixed_dec)
        super().__init__(*args, **kwargs)''')
    replace('''        self.fixed_ra = fixed_ra
        self.fixed_dec = fixed_dec
        self.sky_window_deg = sky_window_deg''','''        # Fixed coordinates/window were resolved by configure_sky_position above.''')
    replace('''    def sample_sky_location(self, batch_size):
        if self.fixed_ra is None:
            return (
                self.ra_prior.sample((batch_size,)),
                self.dec_prior.sample((batch_size,))
            )

        window = np.deg2rad(self.sky_window_deg)
        ras = (self.fixed_ra + (torch.rand(batch_size) - 0.5) * window) % (2 * torch.pi)
        decs = self.fixed_dec + (torch.rand(batch_size) - 0.5) * window
        return ras, decs''','''    def sample_sky_location(self, batch_size):
        """Return Earth-relative (phi, declination) in radians, not celestial RA."""
        return self._sample_sky_location(batch_size)''')
    replace('''                if self.fixed_ra is not None:
                    ccsn_kwargs.update(
                        fixed_ra=self.fixed_ra,
                        fixed_dec=self.fixed_dec,''','''                if self.fixed_phi_rad is not None:
                    ccsn_kwargs.update(
                        fixed_phi_rad=self.fixed_phi_rad,
                        fixed_declination_rad=self.fixed_declination_rad,''')
    replace('''        sky_window_deg=1.0
    ):

        if (fixed_ra is None) != (fixed_dec is None):
            raise ValueError("fixed_ra and fixed_dec must either both be set or both be None")
        if fixed_ra is not None:
            if not np.isfinite(sky_window_deg) or sky_window_deg <= 0:
                raise ValueError("sky_window_deg must be finite and greater than zero")
            half_window = np.deg2rad(sky_window_deg) / 2
            if fixed_dec - half_window < -np.pi / 2 or fixed_dec + half_window > np.pi / 2:
                raise ValueError("fixed_dec sky window must remain within [-pi/2, pi/2]")''','''        sky_window_deg=1.0,
        *,
        fixed_phi_rad=None,
        fixed_declination_rad=None,
    ):
        sky = resolve_sky_position(
            fixed_phi_rad=fixed_phi_rad, fixed_declination_rad=fixed_declination_rad,
            fixed_ra=fixed_ra, fixed_dec=fixed_dec, sky_window_deg=sky_window_deg
        )['gwak_sampling']
        fixed_phi_rad = sky['fixed_phi_rad']
        fixed_declination_rad = sky['fixed_declination_rad']
        sky_window_deg = sky['sky_window_deg']''')
    replace('''        if fixed_ra is None:
            dec = dec_distro.rsample(total_counts)
            ra = Uniform(0, 2 * np.pi).sample((total_counts,))
        else:
            window = np.deg2rad(sky_window_deg)
            dec = fixed_dec + (torch.rand(total_counts) - 0.5) * window
            ra = (fixed_ra + (torch.rand(total_counts) - 0.5) * window) % (2 * torch.pi)''','''        if fixed_phi_rad is None:
            dec = dec_distro.rsample(total_counts)
            phi = Uniform(0, 2 * np.pi).sample((total_counts,))
        else:
            window = np.deg2rad(sky_window_deg)
            dec = fixed_declination_rad + (torch.rand(total_counts) - 0.5) * window
            phi = (fixed_phi_rad + (torch.rand(total_counts) - 0.5) * window) % (2 * torch.pi)''')
    replace('''        ht = gw.compute_observed_strain(
            dec,
            psi,
            ra,''','''        ht = gw.compute_observed_strain(
            dec,
            psi,
            phi,''')
    replace('        return ht, dec, ra, hrss', '        return ht, dec, phi, hrss')
    compile(text, 'dataloader.py', 'exec')
    return text.encode('utf-8')
