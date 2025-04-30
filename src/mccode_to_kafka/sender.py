from pathlib import Path


def send_histograms(root: Path, names: list[str] = None, source: str = None, broker: str = None):
    from .histogram import create_histogram_sink
    from .datfile import read_mccode_dat
    if source is None:
        source = 'mccode-to-kafka'

    if broker is None:
        broker = 'localhost:9092'

    if not root.exists():
        raise RuntimeError(f'{root} does not exist')

    if root.is_file() and names is None:
        names = [root.stem]
        root = root.parent
    elif root.is_dir() and names is None:
        names = [Path(x).stem for x in root.glob('*.dat')]

    config = dict(data_brokers=[broker], source=source)
    security_config = dict()
    sink = create_histogram_sink(config, security_config)

    # If the user specified names, ensure they're present before trying to read them
    names = [name for name in names if root.joinpath(f'{name}.dat').exists()]

    for name in names:
        dat = read_mccode_dat(str(root.joinpath(f'{name}.dat')))
        sink.send_histogram(name, dat, information=f'{name} from {root}')


def command_line_send():
    import argparse
    parser = argparse.ArgumentParser(description='Send histograms to Kafka')
    parser.add_argument('root', type=str, help='The root directory or file to send')
    parser.add_argument('-n', '--name', type=str, nargs='+', help='The names of the histograms to send', default=None)
    parser.add_argument('--source', type=str, help='The source name to use', default=None)
    parser.add_argument('--broker', type=str, help='The broker to send to', default=None)
    args = parser.parse_args()
    send_histograms(Path(args.root), args.name, args.source, args.broker)
