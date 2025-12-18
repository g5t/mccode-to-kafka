from __future__ import annotations
from pathlib import Path
import unittest
import os

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

WRITER = None
WRITER_COUNT = 2
BROKER = 'localhost:9093'
CONTROL_KAFKA = True
CONTROL_WRITER = True
SLEEP_TIME = 1.5
CLIENT = None
TOPICS = ['TEST_epicsConnectionStatus', 'TEST_sampleEnv', 'TEST_writer_jobs', 'TEST_writer_commands',
          'TEST_writer_commands_alternative', 'TEST_forwarderConfig', 'TEST_forwarderStatusLR', 'TEST_forwarderDataLR'
          ]

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
# type: array_1d(3)
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
# xlimits: -0.5 1.5
# variables: t I I_err N
-0.5 10.0 1.0 100.0
0.5 20.0 2.0 200.0
1.5 30.0 3.0 300.0"""


def load_dat_string(string, filename):
    import mock
    from pathlib import Path
    from mccode_to_kafka.datfile import read_mccode_dat
    exists_patcher = mock.patch('pathlib.Path.exists')
    mock_exists = exists_patcher.start()
    mock_exists.return_value = True
    is_file_patcher = mock.patch('pathlib.Path.is_file')
    mock_is_file = is_file_patcher.start()
    mock_is_file.return_value = True

    mock_open = mock.mock_open(read_data=string)

    def mocked_open(s, *args, **kwargs):
        return mock_open(s, *args, **kwargs)

    with mock.patch.object(Path, 'open', mocked_open):
        dat = read_mccode_dat(filename)

    mock.patch.stopall()
    return dat


def wait_on(command, timeout: float):
    from time import sleep
    from datetime import datetime, timedelta
    from file_writer_control.CommandStatus import CommandState
    until = datetime.now() + timedelta(seconds=timeout)
    while not command.is_done():
        if datetime.now() > until:
            raise RuntimeError(f'Time out while waiting for command to finish')
        if command.get_state() == CommandState.ERROR:
            raise RuntimeError(f'Command failed: {command.get_message()}')
        sleep(0.5)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            instance = cls._instances[cls]
            if getattr(cls, '__allow_reinitialization', False):
                instance.__init__(*args, **kwargs)
        return instance


class DockerClient(metaclass=Singleton):
    from pathlib import Path

    @staticmethod
    def client_name():
        from shutil import which
        clients = ('podman', 'docker', 'nerdctl')
        for command in clients:
            if which(command) is not None:
                return command
        return None

    def __init__(self, file_path: Path | str | None = None, up: dict | None = None, down: dict | None = None):
        from python_on_whales import DockerClient
        if file_path is None:
            file_path = Path(__file__).parent.joinpath('docker-compose.yml')
        if isinstance(file_path, str):
            file_path = Path(file_path)
        self.file = file_path.as_posix()
        if (name := self.client_name()) is not None:
            self.client = DockerClient(client_call=[name], compose_files=[self.file])
        else:
            self.client = None
        self.up_dict = up or {}
        self.down_dict = down or {}
        self.is_up = False

    def __del__(self):
        if self.is_up:
            self.down()

    def up(self):
        if not self.is_up:
            self.client.compose.up(**self.up_dict)
            self.is_up = True

    def down(self):
        if self.is_up:
            self.client.compose.down(**self.down_dict)
            self.is_up = False


class FileWriter:
    def __init__(self, executable, broker: str, name: str):
        from subprocess import Popen, PIPE
        from os import access, X_OK
        from pathlib import Path
        path = Path(executable).resolve()
        if not path.exists() or not path.is_file() or not access(path, X_OK):
            raise RuntimeError(f'No such executable {path}')
        opts = dict(status_master_interval='1500ms',
                    kafka_error_timeout='2s',
                    verbosity='trace',
                    job_pool_uri=f'{broker}/TEST_writer_jobs',
                    command_status_uri=f'{broker}/TEST_writer_commands',
                    service_name=name)
        self.name = name
        command = [path] + [f"--{k.replace('_', '-')}={v}" for k, v in opts.items()]
        self.proc = Popen(command, stdout=PIPE, text=True)
        print(f'File Writer {self.name} launched')

    def stop(self):
        from pathlib import Path
        logfile = Path().cwd().joinpath('logs', f'{self.name}.txt')
        if not logfile.parent.exists():
            logfile.parent.mkdir(parents=True)

        self.proc.terminate()
        self.proc.wait()
        output, errors = self.proc.communicate(input=None, timeout=10)
        with logfile.open('w') as file:
            file.write(output)


class FileWriters(metaclass=Singleton):
    def __init__(self, executable, broker: str, count: int):
        if CONTROL_WRITER and executable:
            print("Start up the file writers")
            self.writers = [FileWriter(executable, broker, f'file_writer_{i}') for i in range(count)]
        else:
            print("No file writers started")
            self.writers = []

    def stop(self):
        for w in self.writers:
            w.stop()


def register_topics(broker, topics):
    from confluent_kafka.admin import AdminClient, NewTopic, KafkaError

    client = AdminClient({"bootstrap.servers": broker})
    topics = [NewTopic(t, num_partitions=1, replication_factor=1) for t in topics]
    futures = client.create_topics(topics)

    for topic, future in futures.items():
        try:
            future.result()
            print(f"Topic {topic} created")
        except KafkaError as ke:
            if ke.code() == KafkaError.TOPIC_ALREADY_EXISTS:
                pass
            else:
                print(f'Unexpected Kafka Error {ke}')
        except Exception as e:
            print(f"Failed to create topic {topic}: {e}")


def send_one_d_monitor(file_contents: str, topic: str):
    from mccode_to_kafka.histogram import create_histogram_sink
    sink = create_histogram_sink(dict(data_brokers=[BROKER], source='mccode-to-kafka'), dict())
    dat = load_dat_string(file_contents, f'{topic}.dat')
    sink.send_histogram(topic, dat,
                        information=f'topic {topic} from mccode-to-kafka')  # timestamped as now_in_ns_since_epoch


@unittest.skipIf(IN_GITHUB_ACTIONS, "Orchestrated test probably won't run on GitHub Actions")
class HDF5OutputTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from time import sleep

        client = DockerClient(up={'detach': True}, down={'volumes': True, 'timeout': 20})
        if CONTROL_KAFKA:
            client.up()

        register_topics(BROKER, TOPICS)

        if WRITER and WRITER_COUNT > 0:
            FileWriters(WRITER, BROKER, WRITER_COUNT)
            sleep(1)

    @classmethod
    def tearDownClass(cls):
        client = DockerClient()
        if CONTROL_KAFKA:
            client.down()
        if WRITER:
            single = FileWriters(WRITER, BROKER, WRITER_COUNT)
            single.stop()

    def setUp(self):
        from pathlib import Path
        from file_writer_control import WorkerJobPool
        name = self.id().split('.')[-1]
        # self.file_path = Path('__file__').parent.joinpath('output-files').joinpath(f'{name}.nxs').resolve()
        self.file_path = Path(f'{name}.nxs')
        if not self.file_path.parent.exists():
            self.file_path.parent.mkdir(parents=True)

        # if self.file_path.exists():
        #    self.skipTest(f'{self.file_path} already exists -- remove file to run test')

        self.workers = WorkerJobPool(
            job_topic_url=f"{BROKER}/TEST_writer_jobs",
            command_topic_url=f'{BROKER}/TEST_writer_commands',
            max_message_size=1048576 * 500,
        )

    def tearDown(self):
        pass

    def stop_all_writers(self):
        from time import sleep
        from file_writer_control import JobHandler, JobState
        sleep(SLEEP_TIME)
        jobs = self.workers.list_known_jobs()
        for job in jobs:
            handler = JobHandler(self.workers, job.job_id)
            if handler.get_state() == JobState.WRITING:
                handler.stop_now()
                while not handler.is_done():
                    print(f"Waiting for {job.job_id=} to finish.")
                    sleep(1)
                print(f"Job {job.job_id=} stopped")

    def wait_for_writers(self, *, idle: int | None = None, busy: int | None = None, timeout: float = 10 * SLEEP_TIME):
        from datetime import datetime, timedelta
        from time import sleep
        from file_writer_control.WorkerStatus import WorkerState

        def test():
            if idle is not None:
                return sum([w.state == WorkerState.IDLE for w in self.workers.list_known_workers()]) < idle
            elif busy is not None:
                return sum([w.state != WorkerState.IDLE for w in self.workers.list_known_workers()]) > busy
            raise RuntimeError('Either idle or busy count must be defined')

        give_up_after = datetime.now() + timedelta(seconds=timeout)

        while test() and datetime.now() < give_up_after:
            sleep(SLEEP_TIME)
        if datetime.now() > give_up_after:
            raise RuntimeError(f'Time-out waiting for {idle=} {busy=} workers')

    def wait_for_job(self, job, timeout: float = 10 * SLEEP_TIME):
        from file_writer_control import JobHandler
        handler = JobHandler(worker_finder=self.workers)
        start_job = handler.start_job(job)
        wait_on(start_job, timeout)

    def test_structure_1d(self):
        from file_writer_control import WriteJob
        from mccode_to_kafka.writer import da00_dataarray_config, da00_variable_config
        from datetime import datetime, timezone
        from time import sleep
        from json import dumps
        import h5py
        self.wait_for_writers(idle=1)
        write_from = datetime.now(timezone.utc)

        signal = da00_variable_config(name='signal', unit='counts', axes=['t'], shape=[3], label="the signal",
                                      data_type="float64")
        errors = da00_variable_config(name='signal_errors', unit='counts', axes=['t'], shape=[3], label="the errors",
                                      data_type="float64")
        x = da00_variable_config(name='t', unit='microseconds', label='Time-of-flight', axes=['t'], shape=[4],
                                 data={'first': -1.0, 'last': 2.0, 'size': 4})
        config = da00_dataarray_config(topic='monitor_0', variables=[signal, errors], constants=[x],)


        entry = dict(type='group', name='entry',
                     children=[dict(type="group", name="hist_data", children=[config])],
                     attributes=[dict(name='NX_class', values='NXentry')])
        nx_structure = dict(children=[entry], )

        for _ in range(5):
            send_one_d_monitor(ONE_D_MONITOR, 'monitor_0')

        # sleep to ensure the histogram is sent?
        sleep(1)

        write_until = datetime.now(timezone.utc)

        job = WriteJob(nexus_structure=dumps(nx_structure),
                       file_name=self.file_path.as_posix(),
                       broker=BROKER,
                       start_time=write_from,
                       stop_time=write_until
                       )

        if not self.file_path.exists():
            self.wait_for_job(job)
            self.wait_for_writers(busy=0)

        # in lieu of actually waiting for the writer to close-out the file... 
        sleep(10)

        #        # Check output for NXdata of (NXdata, NXlog, NXlog) -- can't be loaded in scippnexus
        #        with h5py.File(self.file_path, 'r') as file:
        #          self.assertEqual(len(list(file)), 1)
        #          self.assertTrue('entry' in list(file))
        #          self.assertEqual(len(list(file['entry'])), 1)
        #          self.assertTrue('hist_data' in list(file['entry']))
        #          #
        #          hist_data = file['entry/hist_data']
        #          self.assertEqual(len(list(hist_data)), 3)
        #          for name, nx_type in (('histogram', 'NXdata'), ('info', 'NXlog'), ('message_full', 'NXlog')):
        #            print(f'{name=} in {list(hist_data)=}')
        #            self.assertTrue(name in list(hist_data))
        #            print(f'{list(hist_data[name].attrs)=}')
        #            self.assertTrue('NX_class' in hist_data[name].attrs)
        #            self.assertEqual(hist_data[name].attrs['NX_class'], nx_type)

        # TODO continue from here:
        # Check output for NXdata (can be loaded in scippnexus)
        with h5py.File(self.file_path, 'r') as file:
            self.assertEqual(len(list(file)), 1)
            self.assertTrue('entry' in list(file))
            self.assertEqual(len(list(file['entry'])), 1)
            self.assertTrue('hist_data' in list(file['entry']))
            #
            hist = file['entry/hist_data']
            self.assertTrue('NX_class' in hist.attrs)
            self.assertEqual(hist.attrs['NX_class'], 'NXlog')
            # self.assertTrue('signal' in hist.attrs)
            # self.assertEqual(hist.attrs['signal'], 'histograms')
            expected = ('signal', 'signal_errors', 't', 'time', 'cue_index', 'cue_timestamp_zero')
            self.assertEqual(len(list(hist)), len(expected))
            for name in expected:
                self.assertTrue(name in list(hist))


if __name__ == '__main__':
    import sys
    from os import access, X_OK
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument('--broker', default='')
    parser.add_argument('--build-path', default='')
    parser.add_argument('--external-writer', action='store_true')
    parser.add_argument('unittest_arguments', nargs='*')

    parsed_args = parser.parse_args()
    if parsed_args.broker and len(parsed_args.broker):
        CONTROL_KAFKA = False
        BROKER = parsed_args.broker

    if parsed_args.external_writer:
        CONTROL_WRITER = False

    if parsed_args.build_path and len(parsed_args.build_path):
        writer = Path(parsed_args.build_path)
        if writer.exists() and writer.is_file() and access(writer, X_OK):
            WRITER = writer
        elif writer.is_dir() and (writer := writer.joinpath('bin', 'kafka-to-nexus')).exists() and access(writer, X_OK):
            WRITER = writer
        elif writer is not Path() or writer is not Path().joinpath('bin', 'kafka-to-nexus'):
            print(f'Specified writer binary {writer} does not exist')

    unittest.main(argv=[sys.argv[0]] + parsed_args.unittest_arguments)
