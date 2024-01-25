import unittest


class WriterTestCase(unittest.TestCase):
    def test_nexus_structure_1d(self):
        """This exists mostly to illustrate the expected use case of the writer module."""
        from mccode_to_kafka.writer import da00_variable_config, da00_dataarray_config
        import json
        x = {'name': 'x', 'unit': 'm', 'label': 'x_axis', 'data': {'first': 0, 'last': 10, 'size': 11}}
        source = 'source'
        topic = 'topic'
        expected = {
            "module": 'da00',
            "config": {
                "topic": topic,
                "source": source,
                "constants": [x]
            }
        }
        constants = [da00_variable_config(**x)]
        structure = da00_dataarray_config(topic=topic, source=source, constants=constants)
        self.assertEqual(json.dumps(structure), json.dumps(expected))

    def test_nexus_structure_2d(self):
        """This exists mostly to illustrate the expected use case of the writer module for 2D histograms"""
        from mccode_to_kafka.writer import da00_variable_config, da00_dataarray_config
        import json
        x = dict(name='x', unit='m', label='x_axis', data=dict(first=14, last=30, size=31))
        y = dict(name='y', unit='angstrom', label='y_axis', data=dict(first=1, last=3, size=12))
        topic = 'topic'
        expected = {
            "module": 'da00',
            "config": {
                "topic": topic,
                "source": 'mccode-to-kafka',
                "constants": [x, y]
            }
        }
        constants = [da00_variable_config(**c) for c in (x, y)]
        structure = da00_dataarray_config(topic=topic, source='mccode-to-kafka', constants=constants)
        self.assertEqual(json.dumps(structure), json.dumps(expected))


if __name__ == '__main__':
    unittest.main()
