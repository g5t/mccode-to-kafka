import unittest
from pathlib import Path

# Importantly, this string must not start or end with a blank line!
ONE_D_MONITOR = r"""# Format: McCode with text headers
# URL: http://www.mccode.org
# Creator: McStas
# Instrument: 
# Ncount: 200000
# Trace: no
# Gravitation: no
# Seed: 2845994662 3123214533
# Directory: /home/user/data/1
# Param: ps1speed=196
# Param: ps2speed=196
# Date: Wed Oct 25 15:01:11 2023 (1698238871)
# type: array_1d(10)
# Source: BIFROST_first ()
# component: monitor_0
# position: -0.030881 0 6.36934
# title: Time-of-flight monitor
# Ncount: 200000
# filename: monitor_0.dat
# statistics: X0=4120.01; dX=242.952;
# signal: Min=0; Max=1.73106e+09; Mean=3.34694e+06;
# values: 3.34694e+09 1.41799e+08 1380
# xvar: t
# yvar: (I,I_err)
# xlabel: Time-of-flight [\gms]
# ylabel: Intensity
# xlimits: 2030.659785 5234.044652
# variables: t I I_err N
2030.659785 0.0 0.0 0.0
2386.591437 0.0 0.0 0.0
2742.523089 0.0 0.0 0.0
3098.45474 0.0 0.0 0.0
3454.386392 0.0 0.0 0.0
3810.318044 2004197430.0 151945266.68 945.0
4166.249696 3633480854.0 214858025.1 1399.0
4522.181348 1343555711.6999998 137386776.55 466.0
4878.113 0.0 0.0 0.0
5234.044652 0.0 0.0 0.0"""

TWO_D_MONITOR = """# Format: McCode with text headers
# URL: http://www.mccode.org
# Creator: McStas
# Instrument: 
# Ncount: 200000
# Trace: no
# Gravitation: no
# Seed: 2845994662 3123214533
# Directory: /home/user/data/1
# Param: ps1speed=196
# Param: ps2speed=196
# Date: Wed Oct 25 15:01:11 2023 (1698238871)
# type: array_2d(20, 20)
# Source: BIFROST_first ()
# component: psd0
# position: 0.188537 0 161.349
# title: PSD monitor
# Ncount: 200000
# filename: psd0.dat
# statistics: X0=-0.0763542; dX=0.785462; Y0=0.0511929; dY=0.788474;
# signal: Min=0; Max=6.1598e+07; Mean=3.01152e+06;
# values: 1.20461e+09 7.38804e+07 752
# xvar: X
# yvar: Y
# xlabel: X position [cm]
# ylabel: Y position [cm]
# zvar: I
# zlabel: Signal per bin
# xylimits: -5 5 -5 5
# variables: I I_err N
# Data [psd0/psd0.dat] I:
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 25175740.125 16532898.336 67056486.36 65717585.650000006 62134534.09 29219357.014000002 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 59632700.239999995 101164957.44 88598148.53999999 83576096.16999999 80176395.15 46543625.33 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 55015618.019999996 104564949.25 109751856.86000001 63432826.97 125276099.24000001 74819562.88 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 98564526.55 87894630.21 76285725.59 100820996.97999999 72892216.14 48887064.058000006 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 45528092.900000006 117627456.81 90587360.64 52326676.11 80962746.05000001 80433084.74000001 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 35614078.019999996 49075439.51 54805390.269999996 78537181.7 47812301.19 41423450.64 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
# Errors [psd0/psd0.dat] I_err:
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 12435698.431000002 8931937.284 24545367.07 23735472.509999998 23075952.717 15602389.833999999 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 24281594.11 30469667.39 26854526.880000003 29765114.85 27881759.509999998 18250887.663 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 23369949.661 32731137.85 33818517.18 20660626.698 33834937.19 26696718.937 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 31884173.96 27318795.159999996 26309514.07 31709430.43 25745791.28 18478347.929 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 18590701.606 34462458.87 26829395.87 18825333.67 27487471.77 31699933.119999997 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 20419088.84 21809042.84 24177544.439999998 28745587.7 21384234.189999998 19364943.506 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
# Events [psd0/psd0.dat] N:
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 23.0 32.0 37.0 36.0 37.0 16.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 32.0 62.0 69.0 58.0 62.0 37.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 26.0 73.0 59.0 43.0 64.0 34.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 36.0 61.0 51.0 60.0 67.0 37.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 36.0 55.0 64.0 57.0 60.0 33.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 20.0 28.0 36.0 42.0 30.0 20.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"""


class DatFileTestCase(unittest.TestCase):
    def setUp(self):
        import mock
        exists_patcher = mock.patch('pathlib.Path.exists')
        mock_exists = exists_patcher.start()
        mock_exists.return_value = True
        is_file_patcher = mock.patch('pathlib.Path.is_file')
        mock_is_file = is_file_patcher.start()
        mock_is_file.return_value = True

    def tearDown(self):
        import mock
        mock.patch.stopall()

    @staticmethod
    def _load_dat_string(string, filename):
        import mock
        from mccode_to_kafka.datfile import read_mccode_dat
        mock_open = mock.mock_open(read_data=string)

        def mocked_open(s, *args, **kwargs):
            return mock_open(s, *args, **kwargs)

        with mock.patch.object(Path, 'open', mocked_open):
            dat = read_mccode_dat(filename)
        return dat

    def test_load_1d(self):
        from mccode_to_kafka.datfile import DatFile1D

        dat = self._load_dat_string(ONE_D_MONITOR, 'monitor_0.dat')

        self.assertTrue(isinstance(dat, DatFile1D))
        self.assertEqual(int(dat['Ncount']), 200000)
        self.assertEqual(dat['type'], 'array_1d(10)')
        self.assertEqual(dat['filename'], 'monitor_0.dat')
        self.assertEqual(dat['statistics'], 'X0=4120.01; dX=242.952;')
        self.assertEqual(dat['signal'], 'Min=0; Max=1.73106e+09; Mean=3.34694e+06;')
        self.assertEqual(dat['values'], '3.34694e+09 1.41799e+08 1380')
        self.assertEqual(dat['xvar'], 't')
        self.assertEqual(dat['yvar'], '(I,I_err)')
        self.assertEqual(dat['xlabel'], r'Time-of-flight [\gms]')
        self.assertEqual(dat['ylabel'], 'Intensity')
        self.assertEqual(dat['xlimits'], '2030.659785 5234.044652')
        self.assertEqual(dat['variables'], 't I I_err N')
        self.assertEqual(dat.data.shape, (4, 10))

    def test_dim_metadata_1d(self):
        dat = self._load_dat_string(ONE_D_MONITOR, 'filename.dat')

        lower_limit = 2030.659785
        upper_limit = 5234.044652
        bin_width = (upper_limit - lower_limit) / 9
        bin_boundaries = [lower_limit - bin_width/2 + x * bin_width for x in range(10)] + [upper_limit + bin_width / 2]

        dim_metadata = {'length': 10, 'label': 'Time-of-flight', 'unit': 'microseconds',
                        'bin_boundaries': bin_boundaries}
        x_dim_metadata = dat.dim_metadata()[0]
        for key in ('length', 'label', 'unit'):
            self.assertEqual(x_dim_metadata[key], dim_metadata[key])
        for a, b in zip(x_dim_metadata['bin_boundaries'], dim_metadata['bin_boundaries']):
            self.assertAlmostEqual(a, b)

    # def test_hs01_dict_1d(self):
    #     from mccode_to_kafka.utils import now_in_ns_since_epoch
    #     from numpy import isnan
    #     dat = self._load_dat_string(ONE_D_MONITOR, 'filename.dat')
    #     now = now_in_ns_since_epoch()
    #
    #     for normalise in (True, False):
    #         hs01 = dat.to_hs01_dict(info='free text', time=now, normalise=normalise)
    #         self.assertEqual(hs01['source'], str(Path(__file__).parent.joinpath('filename.dat')))
    #         self.assertEqual(hs01['info'], 'free text')
    #         self.assertEqual(hs01['timestamp'], now)
    #         for x, y in (('data', 'I'), ('errors', 'I_err')):
    #             for d, i, n in zip(hs01[x], dat[y], dat['N']):
    #                 if normalise and n == 0:
    #                     self.assertTrue(isnan(d))
    #                 else:
    #                     self.assertEqual(d, i/n if normalise else i)

    def test_load_2d(self):
        from mccode_to_kafka.datfile import DatFile2D
        dat = self._load_dat_string(TWO_D_MONITOR, 'psd0.dat')

        self.assertTrue(isinstance(dat, DatFile2D))
        self.assertEqual(int(dat['Ncount']), 200000)
        self.assertEqual(dat['type'], 'array_2d(20, 20)')
        self.assertEqual(dat['filename'], 'psd0.dat')
        self.assertEqual(dat['statistics'], 'X0=-0.0763542; dX=0.785462; Y0=0.0511929; dY=0.788474;')
        self.assertEqual(dat['signal'], 'Min=0; Max=6.1598e+07; Mean=3.01152e+06;')
        self.assertEqual(dat['values'], '1.20461e+09 7.38804e+07 752')
        self.assertEqual(dat['xvar'], 'X')
        self.assertEqual(dat['yvar'], 'Y')
        self.assertEqual(dat['zvar'], 'I')
        self.assertEqual(dat['xlabel'], 'X position [cm]')
        self.assertEqual(dat['ylabel'], 'Y position [cm]')
        self.assertEqual(dat['zlabel'], 'Signal per bin')
        self.assertEqual(dat['xylimits'], '-5 5 -5 5')
        self.assertEqual(dat['variables'], 'I I_err N')
        self.assertEqual(dat.data.shape, (3, 20, 20))

    def test_dim_metadata_2d(self):
        dat = self._load_dat_string(TWO_D_MONITOR, 'psd0.dat')
        lower_limit = -5
        upper_limit = 5
        bin_width = (upper_limit - lower_limit) / 19
        bin_boundaries = [lower_limit - bin_width/2 + x * bin_width for x in range(20)] + [upper_limit + bin_width / 2]
        x_dim_metadata = {'length': 20, 'label': 'X position', 'unit': 'cm', 'bin_boundaries': bin_boundaries}
        y_dim_metadata = {'length': 20, 'label': 'Y position', 'unit': 'cm', 'bin_boundaries': bin_boundaries}
        dat_y_metadata, dat_x_metadata = dat.dim_metadata()  # metadata dimension order matches data
        for key in ('length', 'label', 'unit'):
            self.assertEqual(dat_x_metadata[key], x_dim_metadata[key])
            self.assertEqual(dat_y_metadata[key], y_dim_metadata[key])
        for a, b in zip(dat_x_metadata['bin_boundaries'], x_dim_metadata['bin_boundaries']):
            self.assertAlmostEqual(a, b)
        for a, b in zip(dat_y_metadata['bin_boundaries'], y_dim_metadata['bin_boundaries']):
            self.assertAlmostEqual(a, b)

    # def test_hs01_dict_2d(self):
    #     from numpy import isnan
    #     dat = self._load_dat_string(TWO_D_MONITOR, 'data.dat')
    #     for normalise in (True, False):
    #         hs01 = dat.to_hs01_dict(source='not a filename', normalise=normalise)
    #         self.assertEqual(hs01['source'], 'not a filename')
    #         self.assertTrue('info' not in hs01)
    #         self.assertTrue(hs01['current_shape'], (20, 20))
    #         for x, y in (('data', 'I'), ('errors', 'I_err')):
    #             for vd, vi, vn in zip(hs01[x], dat[y], dat['N']):
    #                 for d, i, n in zip(vd, vi, vn):
    #                     if normalise and n == 0:
    #                         self.assertTrue(isnan(d))
    #                     else:
    #                         self.assertEqual(d, i/n if normalise else i)


if __name__ == '__main__':
    unittest.main()
