from pathlib import Path
from .histogram import create_histogram_sink
from .datfile import read_mccode_dat


def send_histograms_single_source(root: Path, names: list[str], config: dict, security: dict):
    """
    Single source allows for reusing the same sink for multiple histograms. But all histograms will have
    the same source name, and must be sent to different topics.
    """
    sink = create_histogram_sink(config, security)
    for name in names:
        dat = read_mccode_dat(str(root.joinpath(f'{name}.dat')))
        sink.send_histogram(name, dat, information=f'{name} from {root}')


def send_histograms_single_topic(root: Path, topic: str, names: list[str], config: dict, security: dict):
    """
    Single topic requires one sink per histogram, but allows all histograms to be sent to the same topic.
    The source name will be set per histogram and is taken from the histogram name.
    This is useful for sending multiple histograms to a single topic for later processing.
    """
    for name in names:
        config['source'] = name
        sink = create_histogram_sink(config, security)
        sink.send_histogram(topic, read_mccode_dat(str(root.joinpath(f'{name}.dat'))), information=f'{name} from {root}')


def send_histograms(
        root: Path, names: list[str] | None = None,
        topic: str | None = None, source: str | None = None, broker: str | None = None,
        config: dict | None = None, security: dict | None = None
):
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

    # If the user specified names, ensure they're present before trying to read them
    names = [name for name in names if root.joinpath(f'{name}.dat').exists()]

    if topic is None:
        send_histograms_single_source(root, names, config=config, security=security)
    else:
        send_histograms_single_topic(root, topic, names, config=config, security=security)



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
