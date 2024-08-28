from __future__ import annotations

from datetime import datetime
from .datfile import DatFileCommon


class HistogramSink:
    def __init__(self, producer, source_name, normalise: bool = False):
        """
        Constructor.

        :param producer: The underlying Kafka producer to publish to.
        :param source_name: The source identifier for produced da00 messages
        """
        if producer is None:
            raise Exception("Histogram sink must have a producer")  # pragma: no mutate
        self.producer = producer
        self.source = source_name
        self.normalise = normalise

    def serialise_function(self, histogram: DatFileCommon, timestamp: datetime | None, information: str | None):
        """
        Serialise a histogram as a hs01 FlatBuffers message.

        :param histogram: The histogram to serialise.
        :param timestamp: The timestamp to assign to the histogram.
        :param information: Information to write to the 'info' field.
        :return: The raw buffer of the FlatBuffers message.
        """
        from streaming_data_types.dataarray_da00 import serialise_da00
        da_dict = histogram.to_da00_dict(source=self.source, timestamp=timestamp, info=information,
                                         normalise=self.normalise)
        return serialise_da00(**da_dict)

    def send_histogram(self, topic, histogram: DatFileCommon,
                       timestamp: datetime | None = None, information: str | None = None):
        """
        Send a histogram.

        :param topic: The topic to post to.
        :param histogram: The histogram to send.
        :param timestamp: The timestamp to set (ns since epoch).
        :param information: The message to write to the 'info' field.
        """
        self.producer.publish_message(topic, self.serialise_function(histogram, timestamp, information))


class Mccode2KafkaException(Exception):
    pass


class Producer:
    """
    Publishes messages to Kafka.

    This contains the least amount of logic because it is hard to effectively
    mock the Kafka side without making the tests trivial or pointless.
    """

    def __init__(self, brokers, security_config):
        """
        Constructor.

        :param brokers: The brokers to connect to.
        :param security_config: The security configuration for Kafka.
        """
        from confluent_kafka import KafkaException, Producer as KafkaProducer
        try:
            options = {"bootstrap.servers": ",".join(brokers), "message.max.bytes": 100_000_000}
            self.producer = KafkaProducer({**options, **security_config})
        except KafkaException as error:
            raise Mccode2KafkaException(error)

    def publish_message(self, topic, message):
        """
        Publish messages into Kafka.

        :param topic: The topic to publish to.
        :param message: The message to publish.
        """
        from confluent_kafka import KafkaException
        try:
            self.producer.produce(topic, message)
            self.producer.flush()
        except KafkaException as error:
            raise Mccode2KafkaException(error)


def create_histogram_sink(config, security_config):
    producer = Producer(config["data_brokers"], security_config)
    sink = HistogramSink(producer, config['source'])
    return sink
