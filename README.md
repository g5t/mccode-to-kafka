# McCode to Kafka monitor data converter
Convert McCode style monitor output files to FlatBuffer histograms for Kafka.

## Background
### McStas
[McStas](https://mcstas.org) is one of the Monte Carlo ray tracing suites of McCode.
It is used to simulate neutron scattering instruments and experiments.
McStas can output monitor data in a text format that can be reformulated for transmission to Kafka.

The monitor data format contains a header, comprised of simulation parameters and metadata, 
and then either a number-of-dependent-variables (N) plus 1 by number of independent-variable-observations (M)
2D array in the case of one independent-variable;
or N times M by L arrays in the case of two independent variables, where the second independent-variable
has L observations.
The dependent variables always include `I`, `I_err`, and `N`, from which the simulated cross section 
can be calculated as `I/N` and its uncertainty as `I_err/N` for each observation.

### FlatBuffers
The European Spallation Source (ESS) data acquisition system uses [FlatBuffers](https://google.github.io/flatbuffers/)
for data serialization.
The exact schema used by ESS are part of the [streaming-data-types](https://github.com/ess-dmsc/streaming-data-types)
library, exposed to python via [ess-streaming-data-types](https://github.com/ess-dmsc/python-streaming-data-types).

One of the types defined in the streaming-data-types library, `da00`, can be used to represent
related datasets, including histograms and their axes.
This module can encode McStas histograms as `da00` messages and send them to a Kafka stream.
The histograms can then be retrieved by a [kafka-to-nexus](https://github.com/ess-dmsc/kafka-to-nexus) file writer
for inclusion in a NeXus data file.

## Motivation
The motivation for this project is to be able to use the McStas simulation output directly in the ESS data acquisition
system, without having to keep track of monitor files in addition to the produced NeXus file.

This will enable storing beam-monitor data in the same file as the simulated event data,
even in cases where the beam-monitor data is not available from the running simulation because it was
produced in an earlier saved-run.
E.g., in the case of a MCPL saved primary-spectrometer simulation, the beam-monitor data is not available during
later simulations of the sample and secondary-spectrometer, but is required for the final data reduction.

## Usage
The `mccode-to-kafka` script is used to convert McStas monitor files to FlatBuffer histograms and send them to Kafka.
It is intended to be used as a command-line tool, but can also be used as a python module.

```shell
$ mccode-to-kafka --help
usage: mccode-to-kafka [-h] [-n NAME [NAME ...]] [--source SOURCE] [--broker BROKER] root

Send histograms to Kafka

positional arguments:
  root                  The root directory or file to send

options:
  -h, --help            show this help message and exit
  -n NAME [NAME ...], --name NAME [NAME ...]
                        The names of the histograms to send
  --source SOURCE       The source name to use
  --broker BROKER       The broker to send to
```

Expected usage is to specify the root directory of the McStas simulation output, and the names of the histograms to send.
The names of the histograms are the names of the monitor files, without the `.dat` extension.
```shell
$ mccode-to-kafka -n monitor0 monitor1 --broker localhost:9092 /path/to/simulation/output
```
