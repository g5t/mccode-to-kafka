from pathlib import Path
from dataclasses import dataclass
from .histogram import create_histogram_sink
from .datfile import read_mccode_dat

@dataclass
class HistogramInfo:
    root: Path
    name: str

    @property
    def path(self) -> Path:
        return self.root / f'{self.name}.dat'

    @property
    def filename(self) -> str:
        return str(self.path)

    @property
    def data(self):
        return read_mccode_dat(self.filename)

    @property
    def exists(self):
        return self.path.exists()

    @property
    def message(self):
        return f'{self.name} from {self.root}'

    def delete(self):
        self.path.unlink()


def send_histograms_single_source(histograms: list[HistogramInfo], config: dict, security: dict):
    """
    Single source allows for reusing the same sink for multiple histograms. But all histograms will have
    the same source name, and must be sent to different topics.
    """
    sink = create_histogram_sink(config, security)
    for histogram in histograms:
        sink.send_histogram(histogram.name, histogram.data, information=histogram.message)


def send_histograms_single_topic(histograms: list[HistogramInfo], topic: str, config: dict, security: dict):
    """
    Single topic requires one sink per histogram, but allows all histograms to be sent to the same topic.
    The source name will be set per histogram and is taken from the histogram name.
    This is useful for sending multiple histograms to a single topic for later processing.
    """
    for histogram in histograms:
        config['source'] = histogram.name
        sink = create_histogram_sink(config, security)
        sink.send_histogram(topic, histogram.data, information=histogram.message)


def send_histograms(
        root: Path, names: list[str] | None = None,
        topic: str | None = None, source: str | None = None, broker: str | None = None,
        config: dict | None = None, security: dict | None = None,
        remove: bool = False
):
    """
    Send histogram data files to Kafka.

    Args:
        root: The root directory containing histogram files, or a single .dat file path.
        names: Optional list of histogram names to send. If None, discovers all .dat files in root.
        topic: Kafka topic name. If provided, all histograms are sent to this single topic.
        source: Source name for the sink. Mutually exclusive with topic.
        broker: Kafka broker address. Defaults to 'localhost:9092'.
        config: Optional sink configuration dict. If None, uses default with broker.
        security: Optional security configuration dict for Kafka connection.
        remove: If True, delete histogram files after sending.

    Raises:
        RuntimeError: If root path does not exist.
    """
    if broker is None:
        broker = 'localhost:9092'

    if config is None:
        config = {'data_brokers': [broker]}

    if topic is None and source is None:
        source = 'mccode-to-kafka'
    if source is not None:
        config['source'] = source

    if security is None:
        security = {}

    if not root.exists():
        raise RuntimeError(f'{root} does not exist')

    if root.is_file() and names is None:
        names = [root.stem]
        root = root.parent
    elif root.is_dir() and names is None:
        names = [Path(x).stem for x in root.glob('*.dat')]

    # Construct objects to hold information about the histogram files, plus
    # If the user specified names, ensure they're present before trying to read them
    histograms = [h for h in [HistogramInfo(root, name) for name in names] if h.exists]

    if topic is None:
        send_histograms_single_source(histograms, config=config, security=security)
    else:
        send_histograms_single_topic(histograms, topic, config=config, security=security)

    if remove:
        for histogram in histograms:
            histogram.delete()



def command_line_send():
    import argparse
    parser = argparse.ArgumentParser(description='Send histograms to Kafka')
    parser.add_argument('root', type=str, help='The root directory or file to send')
    parser.add_argument('-n', '--name', type=str, nargs='+', help='The names of the histograms to send', default=None)
    parser.add_argument('--broker', type=str, help='The broker to send to', default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--topic', type=str, help='The topic name to use', default=None)
    group.add_argument('--source', type=str, help='The source name to use', default=None)
    args = parser.parse_args()
    send_histograms(
        root=Path(args.root), names=args.name, broker=args.broker, topic=args.topic, source=args.source)
