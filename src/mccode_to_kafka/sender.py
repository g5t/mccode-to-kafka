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


def config_security(
        broker: str | None, topic: str | None, source: str | None,
        config: dict[str,str|list[str]] | None, security: dict | None
) -> tuple[dict[str,str|list[str]], dict]:
    if config is None:
        config = {'data_brokers': [broker or 'localhost:9092']}
    if topic is None and source is None:
        source = 'mccode-to-kafka'
    if source:
        config['source'] = source
    return config, security or {}


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
    config, security = config_security(broker, topic, source, config, security)

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


def send_json_serialized_histogram(
        json_data: str,
        topic: str | None = None, source : str | None = None, broker: str | None = None,
        config: dict | None = None, security: dict | None = None,
):
    from .datfile import DatFileCommon
    if topic is None:
        topic = 'json_dat'
    config, security = config_security(broker, topic, source, config, security)
    sink = create_histogram_sink(config, security)
    sink.send_histogram(topic, DatFileCommon.from_json(json_data), information="via 'mccode-to-kafka json'")


def command_line_send():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Send histograms to Kafka')
    subparsers = parser.add_subparsers(dest='mode')

    # Subcommand for file-based sending
    file_parser = subparsers.add_parser('file', help='Send from file(s)')
    file_parser.add_argument('root', type=str, help='The root directory or file')
    file_parser.add_argument('-n', '--name', type=str, nargs='+', default=None)
    group = file_parser.add_mutually_exclusive_group()
    group.add_argument('--topic', type=str, default=None)
    group.add_argument('--source', type=str, default=None)

    # Subcommand for JSON-based sending
    json_parser = subparsers.add_parser('json', help='Send from JSON string')
    json_parser.add_argument('data', type=str, help='JSON string of DatFile')
    json_parser.add_argument('--topic', type=str, default=None)
    json_parser.add_argument('--source', type=str, default=None)

    # Common arguments for both
    for p in [file_parser, json_parser]:
        p.add_argument('--broker', type=str, default=None)

    # Default to 'file' mode if no subcommand provided
    if len(sys.argv) > 1 and sys.argv[1] not in ('file', 'json', '-h', '--help'):
        sys.argv.insert(1, 'file')

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    if args.mode == 'json':
        send_json_serialized_histogram(
            json_data=args.data, topic=args.topic, source=args.source,
            broker=args.broker
        )
    else:
        send_histograms(
            root=Path(args.root), names=args.name, broker=args.broker,
            topic=args.topic, source=args.source
        )