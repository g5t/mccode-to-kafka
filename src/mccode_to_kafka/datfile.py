from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from numpy import ndarray
from datetime import datetime


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
class DatFileCommon:
    source: Path
    metadata: dict
    parameters: dict
    variables: list[str]
    data: ndarray
    id: int | None = None

    def __post_init__(self):
        if self.id is None:
            from uuid import uuid4
            self.id = hash(uuid4())

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

    def meta_dim_metadata(self, functor) -> list[dict]:
        pass

    def dim_metadata(self) -> list[dict]:
        return self.meta_dim_metadata(dim_metadata)

    def _to_partial_dict(self, source: str = None, time: int = None, normalise: bool = False):
        from .utils import now_in_ns_since_epoch
        from numpy import geterr, seterr
        hs = dict(source=source or str(self.source), timestamp=time or now_in_ns_since_epoch())
        # We want to ignore division by zero errors, since N == 0 is a valid case indicating no counts
        invalid = geterr()['invalid']
        seterr(invalid='ignore')
        hs['data'] = self['I'] / self['N'] if normalise else self['I']
        hs['errors'] = self['I_err'] / self['N'] if normalise else self['I_err']
        seterr(invalid=invalid)
        return hs

    def dim_variable_dicts(self) -> list[dict]:
        pass

    def to_da00_variables(self, normalise: bool = False):
        from streaming_data_types.dataarray_da00 import Variable
        pd = self._to_partial_dict(normalise=normalise)
        constants = self.to_da00_constants()
        axes = [ax.name for ax in constants]
        variables = [Variable(name='signal', data=pd['data'], axes=axes, unit='counts'),
                     Variable(name='signal_errors', data=pd['errors'], axes=axes, unit='counts')]
        return variables + constants

    def to_da00_constants(self):
        from streaming_data_types.dataarray_da00 import Variable
        constants = [
            Variable(name=x['name'], data=x['bin_boundaries'], axes=[x['name']], unit=x['unit'], label=x['label'])
            for x in self.dim_variable_dicts()]
        return constants

    def to_da00_dict(self, source: str, timestamp: datetime | None = None, normalise: bool = False, info: str | None = None):
        import time
        from streaming_data_types.dataarray_da00 import Variable
        data = self.to_da00_variables(normalise=normalise)
        data.append(Variable(name="producer", data="mccode-to-kafka"))
        if info is not None:
            data.append(Variable(name="info", data=info, source="mccode-to-kafka"))
        # the da00 serializer takes a timestamp _in_ nanoseconds
        timestamp_ns = int(timestamp.timestamp() * 1e9) if timestamp is not None else time.time_ns()
        return dict(source_name=source, timestamp_ns=timestamp_ns, data=data)


@dataclass
class DatFile1D(DatFileCommon):
    def __post_init__(self):
        nx = int(self.metadata['type'].split('(', 1)[1].strip(')'))
        nv = len(self.variables)
        if self.data.shape[0] != nx or self.data.shape[1] != nv:
            raise RuntimeError(f'Unexpected data shape {self.data.shape} for metadata specifying {nx=} and {nv=}')
        # we always want the variables along the first dimension:
        self.data = self.data.transpose((1, 0))

    def meta_dim_metadata(self, functor) -> list[dict]:
        lower_limit, upper_limit = [float(x) for x in self['xlimits'].split()]
        return [functor(self.data.shape[1], self['xlabel'], lower_limit, upper_limit), ]

    def dim_variable_dicts(self) -> list[dict]:
        md = self.meta_dim_metadata(dim_metadata)
        md[0]['name'] = self['xvar']
        return md



@dataclass
class DatFile2D(DatFileCommon):
    def __post_init__(self):
        nx, ny = [int(x) for x in self.metadata['type'].split('(', 1)[1].strip(')').split(',')]
        nv = len(self.variables)
        # FIXME Sort out whether this is right or not
        if self.data.shape[0] != ny * nv or self.data.shape[1] != nx:
            raise RuntimeError(f'Expected {ny*nv =} by {nx =} but have {self.data.shape}')
        self.data = self.data.reshape((nv, ny, nx))

    def meta_dim_metadata(self, functor) -> list[dict]:
        lower_x, upper_x, lower_y, upper_y = [float(x) for x in self['xylimits'].split()]
        # The order here should match the order of the dimensions in self.data!
        return [functor(self.data.shape[1], self['ylabel'], lower_y, upper_y),
                functor(self.data.shape[2], self['xlabel'], lower_x, upper_x)]

    def dim_variable_dicts(self) -> list[dict]:
        md = self.meta_dim_metadata(dim_metadata)
        md[0]['name'] = self['yvar']
        md[1]['name'] = self['xvar']
        return md


def read_mccode_dat(filename: str):
    common = DatFileCommon.from_filename(filename)
    ndim = len(common.metadata['type'].split('(', 1)[1].strip(')').split(','))
    if ndim < 1 or ndim > 2:
        raise RuntimeError(f'Unexpected number of dimensions: {ndim}')
    dat_type = DatFile1D if ndim == 1 else DatFile2D
    return dat_type(common.source, common.metadata, common.parameters, common.variables, common.data)
