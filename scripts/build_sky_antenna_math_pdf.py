"""Build the mathematical walkthrough beside the standalone calculation module."""
import argparse
import importlib.util
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'gwak/sky_position_antenna_patterns_math.pdf')
    parser.add_argument('--preview-dir',type=Path)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('sky',ROOT/'gwak/sky_position_antenna_patterns.py')
    sky=importlib.util.module_from_spec(spec);spec.loader.exec_module(sky)
    ref=sky.event_to_gwak(138.47,-1.94,'2011-05-14T01:32:22.654109')
    direct=sky.event_to_gwak_from_first_principles(138.47,-1.94,'2011-05-14T01:32:22.654109',
                      ut1_minus_utc_seconds=ref['ut1_minus_utc_seconds'])
    antenna=sky.antenna_patterns(direct)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans'})
    pages=[]
    class Page:
        def __init__(self,title,subtitle):
            self.fig=plt.figure(figsize=(8.27,11.69),facecolor='white');self.y=.845
            self.fig.text(.075,.942,'GWAK  /  SKY GEOMETRY & ANTENNA RESPONSE',fontsize=9,color='#47657b')
            self.fig.text(.075,.906,title,fontsize=21,fontweight='bold',color='#17354a')
            self.fig.text(.075,.878,subtitle,fontsize=10,color='#47657b')
            self.fig.text(.075,.035,'sky_position_antenna_patterns.py  |  Mathematical walkthrough',fontsize=8,color='#526674')
            self.fig.text(.925,.035,f'{len(pages)+1} / 8',fontsize=9,ha='right',color='#526674')
            pages.append(self)
        def section(self,title):
            self.y-=.025;self.fig.text(.075,self.y,title,fontsize=13,fontweight='bold',color='#17354a');self.y-=.028
        def text(self,s,size=10.5):
            lines='\n'.join(textwrap.fill(line,width=94) if line else '' for line in s.split('\n'))
            self.fig.text(.075,self.y,lines,fontsize=size,va='top',linespacing=1.45)
            self.y-=.0185*len(lines.splitlines())+.014
        def eq(self,s,size=14):
            self.fig.text(.095,self.y,'$'+s+'$',fontsize=size,va='top');self.y-=.049
        def code(self,s):
            self.fig.text(.085,self.y,s,fontsize=9.2,fontfamily='DejaVu Sans Mono',va='top',linespacing=1.55)
            self.y-=.020*len(s.splitlines())+.017
        def rows(self,headers,rows,widths):
            x=.075
            for label,w in zip(headers,widths):
                self.fig.text(x,self.y,label,fontsize=10,fontweight='bold');x+=w
            self.y-=.03
            for row in rows:
                x=.075
                for item,w in zip(row,widths):
                    self.fig.text(x,self.y,str(item),fontsize=10);x+=w
                self.y-=.028
            self.y-=.012
        def finish(self,pdf):
            assert self.y>.06,('Page overflow',len(pages),self.y)
            self.fig.canvas.draw()
            renderer=self.fig.canvas.get_renderer()
            for item in self.fig.texts:
                box=item.get_window_extent(renderer).transformed(self.fig.transFigure.inverted())
                assert box.x0>=.02 and box.x1<=.98 and box.y0>=.015 and box.y1<=.98, ('Clipped text',item.get_text())
            pdf.savefig(self.fig)
            if args.preview_dir:
                args.preview_dir.mkdir(parents=True,exist_ok=True)
                self.fig.savefig(args.preview_dir/f'page-{pages.index(self)+1}.png',dpi=110)
            plt.close(self.fig)

    with PdfPages(args.output,metadata={'Title':'GWAK: Sky position and antenna patterns from first principles',
                                        'Author':'Sujay Champati / GWAK project',
                                        'Subject':'Calendar arithmetic, Earth rotation, and detector tensor projection'}) as pdf:
        p=Page('1. Inputs and coordinate conventions','What the two conversion functions calculate')
        p.text('The calculation takes an event sky position and UTC trigger time, rotates the direction into the Earth-fixed frame used by GWAK, and projects a gravitational-wave polarization basis onto the Hanford and Livingston detector arms.')
        p.section('Inputs and units')
        p.rows(['Symbol','Meaning','Code input'],[
            ('alpha','Right ascension, degrees','ra_deg'),('delta','Declination, degrees','dec_deg'),
            ('UTC','Full trigger date and clock time','trigger_time_utc'),
            ('DUT1','UT1 minus UTC, seconds','ut1_minus_utc_seconds'),
            ('Delta AT','TAI minus UTC, seconds','tai_minus_utc_seconds'),
            ('psi','Polarization angle, radians','polarization_rad')],[.13,.38,.34])
        p.text('RA is in [0, 360) degrees; declination is in [-90, 90] degrees. All trigonometric calculations use radians. Psi defaults to zero as a reference basis, not an inferred physical polarization.')
        p.section('Earth-fixed axes and angles')
        p.text('+z points north along the rotation axis. +x points toward the equator at Greenwich; +y points toward the equator at 90 degrees east. Phi increases from +x toward +y. Declination is measured up from the equatorial plane; polar angle theta is measured down from +z.')
        p.eq(r'\alpha=\alpha_{\rm deg}\,\pi/180,\quad\delta=\delta_{\rm deg}\,\pi/180')
        p.section('Scope of “from first principles”')
        p.text('The arithmetic implementation uses Gregorian calendar arithmetic, published IAU Earth-rotation/sidereal-time formulas, and detector tensor contractions. It does not call Astropy or ERFA. The IAU coefficients, leap seconds, arm vectors and measured DUT1 are inputs, not quantities derived from fundamental physics here.')
        p.text('Both paths preserve the project convention phi = RA - GMST, with declination unchanged. This is not a full ICRS/J2000-to-ITRS transformation: source precession, nutation and polar motion are not applied.')
        p.finish(pdf)

        p=Page('2. Calendar and time scales','Code: _calendar(), _tai_offset(), and the first calculation block')
        p.text('Let Y, M, D be the Gregorian year, month and day. Let h, m_clock and s be the UTC hour, minute and possibly fractional second. Floor brackets correspond to Python integer division //.')
        p.eq(r'a=\lfloor(14-M)/12\rfloor,\quad y=Y+4800-a,\quad m=M+12a-3',size=12)
        p.eq(r'J=D+\lfloor(153m+2)/5\rfloor+365y+\lfloor y/4\rfloor',size=13)
        p.eq(r'\qquad-\lfloor y/100\rfloor+\lfloor y/400\rfloor-32045',size=13)
        p.eq(r'JD_0=J-0.5,\qquad s_{\rm day}=3600h+60m_{\rm clock}+s')
        p.text('J is the integer Julian day number at noon. Subtracting 0.5 gives midnight at the start of the UTC date. The code keeps the day and its fraction separate to preserve floating-point precision.')
        p.section('Two clocks are needed')
        p.text('UT1 follows the actual rotation of Earth. TT (Terrestrial Time) is a uniform time scale used for the sidereal-time polynomial. DUT1 must come from Earth-orientation measurements; leap-second history supplies Delta AT.')
        p.eq(r'UT1=UTC+DUT1,\qquad TT=UTC+\Delta AT+32.184\ {\rm s}')
        p.eq(r'f_{UT1}=\frac{s_{\rm day}+DUT1}{86400},\qquad f_{TT}=\frac{s_{\rm day}+\Delta AT+32.184}{86400}')
        p.eq(r'JD_{UT1}=JD_0+f_{UT1},\qquad JD_{TT}=JD_0+f_{TT}')
        p.text('If DUT1 is omitted, the independent function explicitly assumes UT1 = UTC. For exact comparison, the verifier supplies the same measured DUT1 used by the Astropy path. TAI-UTC is selected independently from the dated leap-second table.')
        p.text('The bundled table covers dates from 1972-01-01 until its expiry, 2027-06-28; outside that range an explicit verified TAI-UTC is required. Literal leap-second timestamps ending in :60 are rejected. Ordinary times on either side of leap changes are tested.')
        p.finish(pdf)

        p=Page('3. Earth rotation and sidereal time','Code: ut1_days, tt_centuries, era_turns, offset_arcsec, gmst')
        p.section('Measure time from J2000')
        p.eq(r'd=JD_{UT1}-2451545.0,\qquad T=\frac{JD_{TT}-2451545.0}{36525}')
        p.text('d is elapsed UT1 days from the J2000 reference Julian date. T is elapsed TT Julian centuries. A Julian century is 36,525 days.')
        p.section('Earth rotation angle (ERA)')
        p.eq(r'ERA=2\pi\,[0.7790572732640+1.00273781191135448\,d]\ \mathrm{mod}\, 2\pi',size=12)
        p.text('The implementation removes integer turns before summation. Since JD0 is an integer minus 0.5, d modulo one turn is equivalent to 0.5 + f_UT1. Thus the following expression has the same angle, with less precision lost to large numbers:')
        p.eq(r'ERA=2\pi\,[0.7790572732640+0.00273781191135448\,d',size=12)
        p.eq(r'\qquad +0.5+f_{UT1}]\ \mathrm{mod}\, 2\pi',size=12)
        p.section('IAU 2006 Greenwich mean sidereal time')
        p.text('The polynomial P(T) is in arcseconds. It accounts for the offset between the Earth rotation angle and mean equinox-based sidereal time; it does not transform the input catalog source coordinates.')
        p.eq(r'P(T)=0.014506+4612.156534T+1.3915817T^2',size=14)
        p.eq(r'\qquad-0.00000044T^3-0.000029956T^4-0.0000000368T^5',size=12)
        p.eq(r'g=GMST=\left(ERA+\frac{\pi}{180\times3600}\,P(T)\right)\mathrm{mod}\, 2\pi')
        p.text('math.fsum sums the polynomial terms. The Astropy path requests mean Greenwich sidereal time with model IAU2006, using the same convention and the same measured UT1-UTC input.')
        p.finish(pdf)

        p=Page('4. Earth-relative source direction','Code: phi, declination, theta, vector, and gwak_sampling')
        p.text('Earth has turned through Greenwich sidereal angle g. Subtract this rotation from the celestial right ascension to get the Earth-relative azimuth. The minus sign is essential: at fixed RA, increasing sidereal time decreases phi.')
        p.eq(r'\phi=(\alpha-g)\mathrm{mod}\, 2\pi,\quad\delta_E=\delta,\quad\theta=\pi/2-\delta_E')
        p.eq(r'\hat{s}=(\cos\delta_E\cos\phi,\ \cos\delta_E\sin\phi,\ \sin\delta_E)')
        p.text('The unit vector s points FROM Earth TOWARD the source. The wave propagation direction is -s. This geocentric direction is shared by H1 and L1 in the distant-source approximation; detector response differences come from their arm orientations.')
        p.section('Connection to the training parameters')
        p.rows(['Returned quantity','GWAK sampler parameter'],[
            ('phi (radians)','fixed_phi_rad'),('delta_E (radians)','fixed_declination_rad'),
            ('W (degrees, full width)','sky_window_deg')],[.40,.45])
        p.text('The default W = 1 degree is the same full-width box used in our earlier studies: +/-0.5 degree in each coordinate. It is not a cone radius, an equal-solid-angle patch or an event localization uncertainty.')
        p.eq(r'\phi_{\rm sample}=(\phi+\epsilon_\phi)\mathrm{mod}\, 2\pi,\qquad\delta_{\rm sample}=\delta_E+\epsilon_\delta')
        p.eq(r'\epsilon_\phi,\epsilon_\delta\ \sim\ U\!\left(-\frac{W\pi}{360},\frac{W\pi}{360}\right)')
        p.text('The two offsets are independent. Boxes crossing either pole are rejected. Conversion fixes the center at the trigger time once; background timestamps do not rotate it. antenna_patterns() evaluates exactly the supplied center, not an average over this box.')
        p.section('What remains separate')
        p.text('Arrival delays require detector positions and are not calculated by antenna_patterns(). A fixed center fixes the geometric response during a short signal; the waveform polarizations can still vary in time.')
        p.finish(pdf)

        p=Page('5. Polarization and detector tensors','Code: u, v, m, n and DETECTOR_ARMS')
        p.text('Construct two orthonormal vectors perpendicular to the source direction. u points toward increasing polar angle (south); v points toward increasing azimuth (east).')
        p.eq(r'\hat{u}=(\cos\phi\sin\delta_E,\ \sin\phi\sin\delta_E,\ -\cos\delta_E)')
        p.eq(r'\hat{v}=(-\sin\phi,\ \cos\phi,\ 0)')
        p.text('The local ml4gw/LAL polarization convention chooses the following basis at angle psi. The code reduces psi modulo pi because the tensor response repeats after a half-turn.')
        p.eq(r'\hat{m}=-\hat{u}\sin\psi-\hat{v}\cos\psi,\qquad\hat{n}=-\hat{u}\cos\psi+\hat{v}\sin\psi')
        p.eq(r'e^+_{ij}=m_i m_j-n_i n_j,\qquad e^\times_{ij}=m_i n_j+n_i m_j')
        p.section('The two perpendicular detector arms')
        p.text('For each detector, x and y are the unit vectors along its two arms, expressed in the same Earth-fixed Cartesian frame as s. Their differential response is the detector tensor D:')
        p.eq(r'D_{ij}=\frac{1}{2}(x_i x_j-y_i y_j)')
        p.text('The code uses these fixed LAL arm-vector constants (also used by local ml4gw):')
        p.code('H1 x = (-0.22389266154, +0.79983062746, +0.55690487831)\nH1 y = (-0.91397818574, +0.02609403989, -0.40492342125)\n\nL1 x = (-0.95457412153, -0.14158077340, -0.26218911324)\nL1 y = (+0.29774156894, -0.48791033647, -0.82054461286)')
        p.text('The same source basis is projected onto different arm vectors at Hanford and Livingston. This is why each detector gets a different pair of coefficients, even before accounting for arrival delay or detector noise.')
        p.finish(pdf)

        p=Page('6. Antenna coefficients','Code: dot(), f_plus, f_cross, response_amplitude')
        p.section('Contract the two tensors')
        p.eq(r'F_+=\sum_{i,j}D_{ij}e^+_{ij},\qquad F_\times=\sum_{i,j}D_{ij}e^\times_{ij}')
        p.text('The implementation uses four dot products rather than constructing full matrices. Define a = x dot m, b = x dot n, c = y dot m, and d_arm = y dot n.')
        p.eq(r'F_+=\frac{1}{2}(a^2-b^2-c^2+d_{\rm arm}^2),\qquad F_\times=ab-cd_{\rm arm}')
        p.eq(r'R=\sqrt{F_+^2+F_\times^2}')
        p.text('The returned keys are F_plus, F_cross and response_amplitude (= R), separately for H1 and L1. These dimensionless long-wavelength geometric factors are not efficiency, SNR, strain amplitude or a noise-weighted sensitivity.')
        p.section('How the coefficients multiply a waveform')
        p.eq(r'h_I(t)=F_{+,I}\,h_+(t-\tau_I)+F_{\times,I}\,h_\times(t-\tau_I)')
        p.text('Here I denotes H1 or L1; tau_I is the arrival delay relative to a chosen reference time. The current function returns only the F coefficients and R. A neutrino event supplies a hypothetical GW direction/time, not evidence that a GW was observed.')
        p.section('Why R does not depend on the polarization angle')
        p.eq(r'F_+(\psi)=F_+(0)\cos2\psi+F_\times(0)\sin2\psi')
        p.eq(r'F_\times(\psi)=F_\times(0)\cos2\psi-F_+(0)\sin2\psi')
        p.text('These equations are a rotation of the two-component response vector, so its squared length is unchanged. Individual coefficients can change sign or magnitude as psi changes, while R stays fixed. R is a geometric summary, not the response to every possible polarization waveform.')
        p.text('The full-sky plots vary phi and declination to show the pattern. The star and numerical callout give the response at the event itself; those numbers are evaluated directly, not read back from the colored grid.')
        p.finish(pdf)

        p=Page('7. Worked example and verification','IceCube IC110514A  |  reference polarization psi = 0')
        p.code('RA = 138.47 deg, Dec = -1.94 deg\nUTC = 2011-05-14T01:32:22.654109\nDUT1 = '+f'{direct["ut1_minus_utc_seconds"]:.14f}'+' s; TAI-UTC = 34 s')
        p.rows(['Step','Calculated value'],[
            ('Julian date at midnight',f'{direct["julian_date_midnight"]:.1f}'),
            ('UT1 fraction of day',f'{direct["ut1_fraction_of_day"]:.14f}'),
            ('TT centuries T',f'{direct["tt_centuries_from_j2000"]:.14f}'),
            ('Earth rotation angle',f'{direct["earth_rotation_angle_rad"]:.12f} rad'),
            ('GMST',f'{direct["gmst_rad"]:.12f} rad'),
            ('Earth phi',f'{direct["earth_phi_deg"]:.10f} deg'),
            ('Earth declination / polar angle','-1.94 deg / 91.94 deg')],[.42,.43])
        p.rows(['Detector','F_plus','F_cross','R'],[
            (ifo,)+tuple(f'{antenna["detectors"][ifo][k]:+.8f}' for k in ('F_plus','F_cross','response_amplitude'))
            for ifo in ('H1','L1')],[.17,.22,.22,.24])
        p.section('What was checked')
        p.text('Both coordinate functions agree for 348 IceCube catalog rows, two GRBs and five date-boundary cases: maximum angle difference 3.553e-14 rad. The comparison shares measured DUT1, but no reference sidereal time, Julian date or output angle is fed to the arithmetic calculation.')
        p.text('Antenna coefficients agree with local ml4gw and LALSuite across 262 sky/polarization cases to absolute tolerance 2e-7 (reference tensors use float32). Tests cover signs, poles, polarization rotation, compatibility and output plots. These checks validate implementation consistency within the stated coordinate convention.')
        p.finish(pdf)

        p=Page('8. References and usage','How to reproduce the calculation and relate the document to the code')
        p.code('from gwak.sky_position_antenna_patterns import (\n    event_to_gwak, event_to_gwak_from_first_principles,\n    antenna_patterns, plot_antenna_patterns)\n\na = event_to_gwak(138.47, -1.94,\n                  "2011-05-14T01:32:22.654109")\nb = event_to_gwak_from_first_principles(\n    138.47, -1.94, "2011-05-14T01:32:22.654109",\n    ut1_minus_utc_seconds=a["ut1_minus_utc_seconds"])\nresponse = antenna_patterns(b, polarization_rad=0.0)\nplot_antenna_patterns(b, name="IC110514A",\n                      output="IC110514A_antenna")')
        p.text('The example deliberately shares only the measured DUT1 scalar. The independent conversion receives no converted angles or time values from the Astropy path. antenna_patterns accepts either a or b unchanged.')
        p.section('Primary formula references')
        p.text('IAU 2006 GMST and Earth rotation angle, ERFA implementation:\nhttps://raw.githubusercontent.com/liberfa/erfa/master/src/gmst06.c\nhttps://raw.githubusercontent.com/liberfa/erfa/master/src/era00.c',size=9)
        p.text('IERS leap-second table:\nhttps://hpiers.obspm.fr/iers/bul/bulc/Leap_Second.dat',size=9)
        p.text('LAL detector response and arm vectors:\nhttps://lscsoft.docs.ligo.org/lalsuite/7.26/lal/_det_response_8c_source.html\nhttps://lscsoft.docs.ligo.org/lalsuite/7.26/lal/_l_a_l_detectors_8h_source.html',size=9)
        p.finish(pdf)
    print(args.output)

if __name__=='__main__':
    main()
