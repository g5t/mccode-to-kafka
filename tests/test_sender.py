import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil


# Sample 1D histogram data
SAMPLE_1D_DAT = """# Format: McCode with text headers
# URL: http://www.mccode.org
# Creator: McStas
# Instrument: test_instrument
# Ncount: 1000
# Trace: no
# Gravitation: no
# Seed: 123456 789012
# Directory: /tmp/test
# Param: speed=100
# Date: Mon Dec 22 10:00:00 2025 (1734865200)
# type: array_1d(5)
# Source: test_source ()
# component: monitor_1
# position: 0 0 1
# title: Test Monitor
# Ncount: 1000
# filename: hist1.dat
# statistics: X0=50; dX=10;
# signal: Min=0; Max=100; Mean=50;
# values: 250 15 50
# xvar: x
# yvar: (I,I_err)
# xlabel: Position [cm]
# ylabel: Intensity
# xlimits: 0 100
# variables: x I I_err N
0 10.0 1.0 5.0
25 20.0 2.0 10.0
50 30.0 3.0 15.0
75 20.0 2.0 10.0
100 10.0 1.0 5.0"""

SAMPLE_2D_DAT = """# Format: McCode with text headers
# URL: http://www.mccode.org
# Creator: McStas
# Instrument: test_instrument
# Ncount: 1000
# Trace: no
# Gravitation: no
# Seed: 123456 789012
# Directory: /tmp/test
# Param: speed=100
# Date: Mon Dec 22 10:00:00 2025 (1734865200)
# type: array_2d(3, 3)
# Source: test_source ()
# component: psd_monitor
# position: 0 0 1
# title: PSD Monitor
# Ncount: 1000
# filename: hist2.dat
# statistics: X0=0; dX=1; Y0=0; dY=1;
# signal: Min=0; Max=100; Mean=50;
# values: 450 30 90
# xvar: X
# yvar: Y
# xlabel: X position [cm]
# ylabel: Y position [cm]
# zvar: I
# zlabel: Signal per bin
# xylimits: -1 1 -1 1
# variables: I I_err N
# Data [psd_monitor/psd_monitor.dat] I:
10.0 20.0 10.0
20.0 30.0 20.0
10.0 20.0 10.0
# Errors [psd_monitor/psd_monitor.dat] I_err:
1.0 2.0 1.0
2.0 3.0 2.0
1.0 2.0 1.0
# Events [psd_monitor/psd_monitor.dat] N:
5 10 5
10 15 10
5 10 5"""


@pytest.fixture
def temp_dir():
    """Create a temporary directory with test histogram files."""
    temp_path = Path(tempfile.mkdtemp())

    # Create test .dat files
    (temp_path / "hist1.dat").write_text(SAMPLE_1D_DAT)
    (temp_path / "hist2.dat").write_text(SAMPLE_2D_DAT)
    (temp_path / "hist3.dat").write_text(SAMPLE_1D_DAT)

    yield temp_path

    # Cleanup
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_sink():
    """Create a mock histogram sink."""
    sink = Mock()
    sink.send_histogram = Mock()
    return sink


@pytest.fixture
def mock_histogram():
    """Create a mock histogram object."""
    from mccode_to_kafka.datfile import DatFileCommon
    return Mock(spec=DatFileCommon)


class TestSendHistogramsSingleSource:
    """Tests for send_histograms_single_source function."""

    @patch('mccode_to_kafka.sender.create_histogram_sink')
    @patch('mccode_to_kafka.sender.read_mccode_dat')
    def test_sends_multiple_histograms_same_sink(self, mock_read, mock_create_sink, temp_dir, mock_sink, mock_histogram):
        """Test that multiple histograms are sent using the same sink."""
        from mccode_to_kafka.sender import send_histograms_single_source, HistogramInfo

        mock_create_sink.return_value = mock_sink
        mock_read.return_value = mock_histogram

        names = ['hist1', 'hist2']
        config = {'source': 'test-source', 'data_brokers': ['localhost:9092']}
        security = {}

        histograms = [HistogramInfo(temp_dir, name) for name in names]

        send_histograms_single_source(histograms, config, security)

        # Verify sink was created once
        mock_create_sink.assert_called_once_with(config, security)

        # Verify both histograms were read
        assert mock_read.call_count == 2
        mock_read.assert_any_call(str(temp_dir / 'hist1.dat'))
        mock_read.assert_any_call(str(temp_dir / 'hist2.dat'))

        # Verify both histograms were sent with different topic names
        assert mock_sink.send_histogram.call_count == 2
        calls = mock_sink.send_histogram.call_args_list
        assert calls[0][0][0] == 'hist1'
        assert calls[1][0][0] == 'hist2'

    @patch('mccode_to_kafka.sender.create_histogram_sink')
    @patch('mccode_to_kafka.sender.read_mccode_dat')
    def test_includes_information_field(self, mock_read, mock_create_sink, temp_dir, mock_sink, mock_histogram):
        """Test that information field is properly set."""
        from mccode_to_kafka.sender import send_histograms_single_source, HistogramInfo

        mock_create_sink.return_value = mock_sink
        mock_read.return_value = mock_histogram

        names = ['hist1']
        config = {'source': 'test-source', 'data_brokers': ['localhost:9092']}
        security = {}

        histograms = [HistogramInfo(temp_dir, name) for name in names]

        send_histograms_single_source(histograms, config, security)

        # Check that information parameter is passed
        call_args = mock_sink.send_histogram.call_args
        assert 'information' in call_args.kwargs
        assert 'hist1' in call_args.kwargs['information']
        assert str(temp_dir) in call_args.kwargs['information']


class TestSendHistogramsSingleTopic:
    """Tests for send_histograms_single_topic function."""

    @patch('mccode_to_kafka.sender.create_histogram_sink')
    @patch('mccode_to_kafka.sender.read_mccode_dat')
    def test_creates_sink_per_histogram(self, mock_read, mock_create_sink, temp_dir, mock_histogram):
        """Test that a new sink is created for each histogram."""
        from mccode_to_kafka.sender import send_histograms_single_topic, HistogramInfo

        # Capture config state at each call
        captured_configs = []
        mock_sink = Mock()

        def capture_config(config, security):
            # Make a copy of config to capture its state at this moment
            captured_configs.append(config.copy())
            return mock_sink

        mock_create_sink.side_effect = capture_config
        mock_read.return_value = mock_histogram

        names = ['hist1', 'hist2']
        topic = 'test-topic'
        config = {'data_brokers': ['localhost:9092']}
        security = {}

        histograms = [HistogramInfo(temp_dir, name) for name in names]

        send_histograms_single_topic(histograms, topic, config, security)

        # Verify sink was created twice (once per histogram)
        assert mock_create_sink.call_count == 2

        # Verify each sink was created with a different source name
        assert captured_configs[0]['source'] == 'hist1'
        assert captured_configs[1]['source'] == 'hist2'

        # Verify both histograms were sent to the same topic
        assert mock_sink.send_histogram.call_count == 2
        send_calls = mock_sink.send_histogram.call_args_list
        assert send_calls[0][0][0] == topic
        assert send_calls[1][0][0] == topic


class TestSendHistograms:
    """Tests for the main send_histograms function."""

    @patch('mccode_to_kafka.sender.send_histograms_single_source')
    def test_defaults_broker(self, mock_single_source, temp_dir):
        """Test that broker defaults to localhost:9092."""
        from mccode_to_kafka.sender import send_histograms

        send_histograms(temp_dir, names=['hist1'])

        # Verify the config has the default broker
        mock_single_source.assert_called_once()
        call_kwargs = mock_single_source.call_args.kwargs
        config = call_kwargs['config']
        assert 'data_brokers' in config
        assert config['data_brokers'] == ['localhost:9092']

    @patch('mccode_to_kafka.sender.send_histograms_single_source')
    def test_defaults_source_name(self, mock_single_source, temp_dir):
        """Test that source defaults to 'mccode-to-kafka' when no topic specified."""
        from mccode_to_kafka.sender import send_histograms

        send_histograms(temp_dir, names=['hist1'])

        # Verify the config has the default source
        mock_single_source.assert_called_once()
        call_kwargs = mock_single_source.call_args.kwargs
        config = call_kwargs['config']
        assert 'source' in config
        assert config['source'] == 'mccode-to-kafka'

    @patch('mccode_to_kafka.sender.send_histograms_single_topic')
    def test_uses_single_topic_when_topic_specified(self, mock_single_topic, temp_dir):
        """Test that single_topic mode is used when topic is specified."""
        from mccode_to_kafka.sender import send_histograms

        send_histograms(temp_dir, names=['hist1'], topic='my-topic')

        # Verify single_topic function was called with correct topic
        mock_single_topic.assert_called_once()
        assert mock_single_topic.call_args[0][1] == 'my-topic'

    @patch('mccode_to_kafka.sender.send_histograms_single_source')
    def test_uses_single_source_when_source_specified(self, mock_single_source, temp_dir):
        """Test that single_source mode is used when source is specified."""
        from mccode_to_kafka.sender import send_histograms

        send_histograms(temp_dir, names=['hist1'], source='my-source')

        # Verify single_source function was called
        mock_single_source.assert_called_once()
        call_kwargs = mock_single_source.call_args.kwargs
        config = call_kwargs['config']
        assert config['source'] == 'my-source'

    def test_raises_error_if_root_does_not_exist(self):
        """Test that RuntimeError is raised if root path doesn't exist."""
        from mccode_to_kafka.sender import send_histograms

        non_existent_path = Path('/non/existent/path')

        with pytest.raises(RuntimeError, match='does not exist'):
            send_histograms(non_existent_path)

    @patch('mccode_to_kafka.sender.send_histograms_single_source')
    def test_handles_file_as_root(self, mock_single_source, temp_dir):
        """Test that a file path can be used as root."""
        from mccode_to_kafka.sender import send_histograms

        file_path = temp_dir / 'hist1.dat'

        send_histograms(file_path)

        # Verify function was called with parent directory and file stem as name
        mock_single_source.assert_called_once()
        called_histograms = mock_single_source.call_args[0][0]
        called_roots = [h.root for h in called_histograms]
        called_names = [h.name for h in called_histograms]

        assert called_roots == [temp_dir]
        assert called_names == ['hist1']

    @patch('mccode_to_kafka.sender.send_histograms_single_source')
    def test_discovers_all_dat_files_in_directory(self, mock_single_source, temp_dir):
        """Test that all .dat files are discovered when no names specified."""
        from mccode_to_kafka.sender import send_histograms

        send_histograms(temp_dir)

        # Verify function was called with all histogram names
        mock_single_source.assert_called_once()
        called_names = [h.name for h in mock_single_source.call_args[0][0]]

        assert 'hist1' in called_names
        assert 'hist2' in called_names
        assert 'hist3' in called_names
        assert len(called_names) == 3

    @patch('mccode_to_kafka.sender.send_histograms_single_source')
    def test_filters_nonexistent_names(self, mock_single_source, temp_dir):
        """Test that nonexistent histogram names are filtered out."""
        from mccode_to_kafka.sender import send_histograms

        names = ['hist1', 'nonexistent', 'hist2']
        send_histograms(temp_dir, names=names)

        # Verify only existing files are passed
        mock_single_source.assert_called_once()
        called_histograms = mock_single_source.call_args[0][0]
        called_names = [h.name for h in called_histograms]

        assert 'hist1' in called_names
        assert 'hist2' in called_names
        assert 'nonexistent' not in called_names
        assert len(called_names) == 2

    @patch('mccode_to_kafka.sender.send_histograms_single_source')
    def test_custom_config_and_security(self, mock_single_source, temp_dir):
        """Test that custom config and security dicts are passed through."""
        from mccode_to_kafka.sender import send_histograms

        custom_config = {
            'data_brokers': ['kafka1:9092', 'kafka2:9092'],
            'custom_option': 'value'
        }
        custom_security = {
            'security.protocol': 'SASL_SSL',
            'sasl.mechanism': 'PLAIN'
        }

        send_histograms(temp_dir, names=['hist1'], config=custom_config, security=custom_security)

        # Verify custom config and security are passed
        mock_single_source.assert_called_once()
        call_kwargs = mock_single_source.call_args.kwargs
        called_config = call_kwargs['config']
        called_security = call_kwargs['security']

        assert called_config == custom_config
        assert called_security == custom_security

    @patch('mccode_to_kafka.sender.send_histograms_single_source')
    def test_remove_deletes_histogram_files(self, mock_single_source, temp_dir):
        """Test that remove=True deletes histogram files after sending."""
        from mccode_to_kafka.sender import send_histograms

        # Verify files exist before
        assert (temp_dir / 'hist1.dat').exists()
        assert (temp_dir / 'hist2.dat').exists()

        send_histograms(temp_dir, names=['hist1', 'hist2'], remove=True)

        # Verify files were deleted
        assert not (temp_dir / 'hist1.dat').exists()
        assert not (temp_dir / 'hist2.dat').exists()
        # hist3 should still exist (not in names list)
        assert (temp_dir / 'hist3.dat').exists()

    @patch('mccode_to_kafka.sender.send_histograms_single_source')
    def test_remove_false_preserves_files(self, mock_single_source, temp_dir):
        """Test that remove=False (default) preserves histogram files."""
        from mccode_to_kafka.sender import send_histograms

        send_histograms(temp_dir, names=['hist1', 'hist2'], remove=False)

        # Verify files still exist
        assert (temp_dir / 'hist1.dat').exists()
        assert (temp_dir / 'hist2.dat').exists()


class TestCommandLineSend:
    """Tests for command_line_send function."""

    @patch('mccode_to_kafka.sender.send_histograms')
    @patch('argparse.ArgumentParser.parse_args')
    def test_parses_basic_arguments(self, mock_parse_args, mock_send_histograms, temp_dir):
        """Test that command line arguments are properly parsed."""
        from mccode_to_kafka.sender import command_line_send

        # Mock the parsed arguments
        mock_args = Mock()
        mock_args.root = str(temp_dir)
        mock_args.name = ['hist1', 'hist2']
        mock_args.broker = 'kafka:9092'
        mock_args.topic = None
        mock_args.source = 'cli-source'
        mock_parse_args.return_value = mock_args

        command_line_send()

        # Verify send_histograms was called with correct arguments
        mock_send_histograms.assert_called_once_with(
            root=temp_dir,
            names=['hist1', 'hist2'],
            broker='kafka:9092',
            topic=None,
            source='cli-source'
        )

    @patch('mccode_to_kafka.sender.send_histograms')
    @patch('argparse.ArgumentParser.parse_args')
    def test_handles_topic_argument(self, mock_parse_args, mock_send_histograms, temp_dir):
        """Test that topic argument is properly handled."""
        from mccode_to_kafka.sender import command_line_send

        mock_args = Mock()
        mock_args.root = str(temp_dir)
        mock_args.name = None
        mock_args.broker = None
        mock_args.topic = 'my-topic'
        mock_args.source = None
        mock_parse_args.return_value = mock_args

        command_line_send()

        mock_send_histograms.assert_called_once()
        call_kwargs = mock_send_histograms.call_args.kwargs
        assert call_kwargs['topic'] == 'my-topic'
        assert call_kwargs['source'] is None


class TestIntegrationWithRealFiles:
    """Integration tests using actual file I/O (no mocking of datfile reading)."""

    @patch('mccode_to_kafka.sender.create_histogram_sink')
    def test_reads_and_sends_real_histogram(self, mock_create_sink, temp_dir):
        """Test end-to-end with real file reading."""
        from mccode_to_kafka.sender import send_histograms

        mock_sink = Mock()
        mock_create_sink.return_value = mock_sink

        send_histograms(temp_dir, names=['hist1'], source='integration-test')

        # Verify sink was created and histogram was sent
        mock_create_sink.assert_called_once()
        mock_sink.send_histogram.assert_called_once()

        # Verify the histogram object is a real DatFileCommon object
        sent_histogram = mock_sink.send_histogram.call_args[0][1]
        from mccode_to_kafka.datfile import DatFileCommon
        assert isinstance(sent_histogram, DatFileCommon)

