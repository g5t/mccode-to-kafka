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
        source = Path(filename).resolve()
        if not source.exists():
            raise RuntimeError('Source filename does not exist')
        if not source.is_file():
            raise RuntimeError(f'{filename} does not name a valid file')
        with source.open('r') as file:
            lines = file.readlines()
        return cls.from_lines(source, lines)

    @classmethod
    def from_lines(cls, source: Path, lines: list[str]):
        from numpy import array
        header = [x.strip(' #\n') for x in filter(lambda x: x[0] == '#', lines)]
        meta = {k.strip(): v.strip() for k, v in
                [x.split(':', 1) for x in filter(lambda x: not x.startswith('Param'), header)]}
        parm = {k.strip(): v.strip() for k, v in
                [x.split(':', 1)[1].split('=', 1) for x in filter(lambda x: x.startswith('Param'), header)]}
        var = meta.get('variables', '').split(' ')
        data = array([[float(x) for x in line.strip().split()] for line in filter(lambda x: x[0] != '#', lines)])
        return cls(source, meta, parm, var, data)

    def __getitem__(self, item):
        if item in self.variables:
            index = [i for i, x in enumerate(self.variables) if x == item]
            if len(index) != 1:
                raise RuntimeError(f'Expected one index for {item} but found {index}')
            return self.data[index[0], ...]
        elif item in self.parameters:
            return self.parameters[item]
        elif item in self.metadata:
            return self.metadata[item]
        else:
            raise KeyError(f'Unknown key {item}')

    def dim_metadata(self) -> list[dict]:
        pass

    def to_hs_dict(self, source: str = None, info: str = None, time: int = None, normalise: bool = False):
        """Produce a dictionary suitable for serialising to HS00 or HS01 via ess-streaming-data-types"""
        from .utils import now_in_ns_since_epoch
        from numpy import geterr, seterr
        hs = dict(source=source or str(self.source), timestamp=time or now_in_ns_since_epoch())
        if info:
            hs['info'] = info

        # We want to ignore division by zero errors, since N == 0 is a valid case indicating no counts
        invalid = geterr()['invalid']
        seterr(invalid='ignore')
        hs['data'] = self['I'] / self['N'] if normalise else self['I']
        hs['errors'] = self['I_err'] / self['N'] if normalise else self['I_err']
        seterr(invalid=invalid)

        hs['current_shape'] = list(hs['data'].shape)
        hs['dim_metadata'] = self.dim_metadata()
        return hs

    def to_hs01_dict(self, source: str = None, info: str = None, time: int = None, normalise: bool = False):
        # any integer values are allowed to be signed for HS01
        return self.to_hs_dict(source=source, info=info, time=time, normalise=normalise)

    def to_hs00_dict(self, source: str = None, info: str = None, time: int = None, normalise: bool = False):
        # integer values must be unsigned for HS00 -- but that should be the case already, so ignore it?
        return self.to_hs_dict(source=source, info=info, time=time, normalise=normalise)


def dim_metadata(length, label_unit, lower_limit, upper_limit) -> dict:
    from numpy import linspace
    parts = label_unit.split(' ')
    label = ' '.join(parts[:-1])
    unit = parts[-1].strip('[] ')
    if '\\gms' == unit:
        unit = 'microseconds'
    bin_width = (upper_limit - lower_limit) / (length - 1)
    boundaries = linspace(lower_limit - bin_width / 2, upper_limit + bin_width / 2, length + 1)
    return dict(length=length, label=label, unit=unit, bin_boundaries=boundaries)


@dataclass
class DatFile1D(DatFileCommon):
    def __post_init__(self):
        nx = int(self.metadata['type'].split('(', 1)[1].strip(')'))
        nv = len(self.variables)
        if self.data.shape[0] != nx or self.data.shape[1] != nv:
            raise RuntimeError(f'Unexpected data shape {self.data.shape} for metadata specifying {nx=} and {nv=}')
        # we always want the variables along the first dimension:
        self.data = self.data.transpose((1, 0))

    def dim_metadata(self) -> list[dict]:
        lower_limit, upper_limit = [float(x) for x in self['xlimits'].split()]
        return [dim_metadata(self.data.shape[1], self['xlabel'], lower_limit, upper_limit), ]


@dataclass
class DatFile2D(DatFileCommon):
    def __post_init__(self):
        nx, ny = [int(x) for x in self.metadata['type'].split('(', 1)[1].strip(')').split(',')]
        nv = len(self.variables)
        # FIXME Sort out whether this is right or not
        if self.data.shape[0] != ny * nv or self.data.shape[1] != nx:
            raise RuntimeError(f'Expected {ny*nv =} by {nx =} but have {self.data.shape}')
        self.data = self.data.reshape((nv, ny, nx))

    def dim_metadata(self) -> list[dict]:
        lower_x, upper_x, lower_y, upper_y = [float(x) for x in self['xylimits'].split()]
        return [dim_metadata(self.data.shape[2], self['xlabel'], lower_x, upper_x),
                dim_metadata(self.data.shape[1], self['ylabel'], lower_y, upper_y)]


def read_mccode_dat(filename: str):
    common = DatFileCommon.from_filename(filename)
    ndim = len(common.metadata['type'].split('(', 1)[1].strip(')').split(','))
    if ndim < 1 or ndim > 2:
        raise RuntimeError(f'Unexpected number of dimensions: {ndim}')
    dat_type = DatFile1D if ndim == 1 else DatFile2D
    return dat_type(common.source, common.metadata, common.parameters, common.variables, common.data)
