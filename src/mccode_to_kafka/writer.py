"""The kafka-to-nexus file writer must be configured to include streamed histogram data in the output file.

As noted at https://github.com/ess-dmsc/kafka-to-nexus/blob/main/documentation/writer_module_hs00_event_histogram.md
the NeXus Structure JSON must include
    - topic             # the Kafka topic to subscribe to
    - source            # the Kafka source name to use for monitors
    - module            # the streaming-data-type module used for serialisation -- this must be 'hs00' at the moment
    - data_type         # the signal data type, one of ('uint32', 'uint64', 'float', or 'double')
    - error_type        # the error data type, one of ('uint32', 'uint64', 'float', or 'double')
    - edge_type         # the bin-edge values data type, one of ('uint32', 'uint64', 'float', or 'double')
    - shape             # A list of JSON objects with the following keys:
        - size            # the number of bins in this dimension, must be an integer value
        - label           # the label for this dimension
        - unit            # the unit for this dimension
        - edges           # the bin-edge values for this dimension, should be one longer than `size`
        - dataset_name    # the name of the dataset to write to in the NeXus file (TODO verify what this does)
"""

# It might be nice to re-use `DatFileCommon` and its subclasses for this, but they rely heavily on construction from
# McStas/McXtrace written .dat files; and writing even a fake file in a string is potentially dangerous for a user.


def edge(bins: int, lower: float, upper: float, label: str, unit: str, name: str):
    from numpy import linspace
    return {
        "size": bins,
        "label": label,
        "unit": unit,
        "edges": linspace(lower, upper, bins + 1).tolist(),
        "dataset_name": name
    }


def nexus_structure(topic: str, shape: list[dict], source: str = None, module: str = None):
    if source is None:
        # By default, we should enforce a common 'source' name for all histograms in this module
        source = 'mccode-to-kafka'
    if module is None:
        # for now at least, only hs00 is possible, so we can hard code it
        module = 'hs00'

    # McStas/McXtrace dat files *always* print data as floating point values, even if they are actually integers.
    # And the DatFileCommon reader uses `float` to convert the data block, which produces IEEE 754 64-bit floats
    data_type = 'double'
    error_type = 'double'
    edge_type = 'double'

    return {
        "topic": topic,
        "source": source,
        "module": module,
        "data_type": data_type,
        "error_type": error_type,
        "edge_type": edge_type,
        "shape": shape
    }
