"""The kafka-to-nexus file writer must be configured to include streamed histogram data in the output file."""
from __future__ import annotations


def da00_dataarray_config(topic: str,
                          source: str | None = None,
                          variables: list[dict] | None = None,
                          constants: list[dict] | None = None,
                          attrs: list[dict] | None = None,
                          signal: str | None = None,
                          axes: list[str] | None = None
                          ) -> dict:
    """
    Construct a JSON object dict representing the configuration of the `da00` file writer module

    :param topic: The Kafka stream name for the writer to read from
    :param source: The producer name, `'kafka-to-nexus'` if not provided
    :param variables: The configurations of time-dependent datasets, as JSON object dicts as produced by
                    `da00_dataset_nexus_structure`
    :param constants: The configurations of time-independent datasets, as JSON object dicts as produced by
                    `da00_dataset_nexus_structure`
    :param attrs: A list of {name: , value:, ...} attribute dictionaries for inclusion in the group attributes
    :param signal: A special attribute that is the name of the 'signal' dataset
    :param axes: A special attribute that is the list of axes names for the 'signal' dataset
    :return: A JSON object dict suitable for inclusion in a NeXus Structure
    """
    config = {'topic': topic, 'source': source or 'mccode-to-kafka'}
    if variables is not None:
        config['variables'] = variables
    if constants is not None:
        config['constants'] = constants
    if attrs is not None:
        config['attributes'] = attrs
    if (signal is not None or axes is not None) and 'attributes' not in config:
        config['attributes'] = []
    if signal is not None:
        config['attributes'].append({'name': 'signal', 'value': signal})
    if axes is not None:
        config['attributes'].append({'name': 'axes', 'value': axes})
    return {'module': 'da00', 'config': config}


def da00_variable_config(name: str,
                         unit: str | None = None,
                         label: str | None = None,
                         source: str | None = None,
                         data_type: str | None = None,
                         axes: list[str] | None = None,
                         shape: list[int] | None = None,
                         data: list | dict | None = None,
                         ):
    """
    Return a JSON object dict representing a dataset suitable for use in a NeXus Structure specification

    :param name: the name of the dataset
    :param unit: the unit of its values, optional
    :param label: a descriptive label for the dataset, optional
    :param source: the identifier for the source of this dataset, optional
    :param data_type: the element type of the data, the provided string should be
                      in (int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64, c_string)
    :param axes: the _names_ of the dimensions of the dataset, e.g., ['x', 'time'], optional
    :param shape: the _sizes_ of the dimensions of the dataset, e.g., [10, 15], optional
    :param data: the C-ordered flattened array of the _fixed_ values of the dataset, _or_ a JSON object dict
                 representing the 'linspace' of the values {'first': #, 'last': #, 'size': #}, optional
    :return: the JSON object dictionary suitable for serialization
    """
    config = {'name': name}
    if unit is not None:
        config['unit'] = unit
    if label is not None:
        config['label'] = label
    if source is not None:
        config['source'] = source
    if data_type is not None:
        config['data_type'] = data_type
    if axes is not None and all(isinstance(x, str) for x in axes):
        config['axes'] = axes
    if shape is not None:
        config['shape'] = shape
    if data is not None:
        if isinstance(data, list):
            config['data'] = data
        elif isinstance(data, dict) and all(isinstance(data.get(x, None), (int, float)) for x in ('first', 'last', 'size')):
            config['data'] = data
    if not (('data_type' in config and 'shape' in config) or 'data' in config):
        raise ValueError(f'da00 requires at least (`data_type`, `shape`) [or `data`] to make a Variable HDF5 dataset')
    return config
