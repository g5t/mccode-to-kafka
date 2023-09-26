from dataclasses import dataclass, field
from pathlib import Path
from numpy import ndarray

@dataclass
class DatFileCommon:
    source: Path
    metadata: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    variables: list[str] = field(default_factory=list)
    data: ndarray = field(default_factory=ndarray)

    @classmethod
    def from_filename(cls, filename: str):
        from numpy import array
        source = Path(filename).resolve()
        if not source.exists():
            raise RuntimeError('Source filename does not exist')
        if not source.is_file():
            raise RuntimeError(f'{filename} does not name a valid file')
        with source.open('r') as file:
            lines = file.readlines()

        header = [x.strip(' #\n') for x in filter(lambda x: x[0] == '#', lines)]
        meta = {k.strip(): v.strip() for k, v in [x.split(':', 1) for x in filter(lambda x: not x.startswith('Param'), header)]}
        parm = {k.strip(): v.strip() for k, v in [x.split(':', 1)[1].split('=', 1) for x in filter(lambda x: x.startswith('Param'), header)]}
        var = meta.get('variables', '').split(' ')
        data = array([[float(x) for x in line.strip().split()] for line in filter(lambda x: x[0] != '#', lines)])
        return cls(source, meta, parm, var, data)


@dataclass
class DatFile1D(DatFileCommon):
    def __post_init__(self):
        nx = int(self.metadata['type'].split('(', 1)[1].strip(')'))
        nv = len(self.variables)
        if self.data.shape[0] != nx or self.data.shape[1] != nv:
            raise RuntimeError(f'Unexpected data shape {self.data.shape} for metadata specifying {nx=} and {nv=}')
        # we always want the variables along the first dimension:
        self.data = self.data.transpose((1, 0))


@dataclass
class DatFile2D(DatFileCommon):
    def __post_init__(self):
        nx, ny = [int(x) for x in self.metadata['type'].split('(', 1)[1].strip(')').split(',')]
        nv = len(self.variables)
        # FIXME Sort out whether this is right or not
        if self.data.shape[0] != ny * nv or self.data.shape[1] != nx:
            raise RuntimeError(f'Expected {ny*nv =} by {nx =} but have {self.data.shape}')
        self.data = self.data.reshape((nv, ny, nx))


