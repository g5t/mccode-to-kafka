import unittest


class WriterTestCase(unittest.TestCase):
    def test_nexus_structure_1d(self):
        """This exists mostly to illustrate the expected use case of the writer module."""
        from mccode_to_kafka.writer import nexus_structure, edge
        from numpy import allclose
        import json
        x = edge(10, 0, 10, 'x', 'm', 'x_axis')
        source = 'source'
        topic = 'topic'
        expected = {
            "module": 'hs00',
            "config": {
                "topic": topic,
                "source": source,
                "data_type": 'double',
                "error_type": 'double',
                "edge_type": 'double',
                "shape": [x]
            }
        }
        structure = nexus_structure(source=source, topic=topic, shape=[x])
        self.assertEqual(json.dumps(structure), json.dumps(expected))
        self.assertTrue(allclose(structure['config']['shape'][0]['edges'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

    def test_nexus_structure_2d(self):
        """This exists mostly to illustrate the expected use case of the writer module for 2D histograms"""
        from mccode_to_kafka.writer import nexus_structure, edge
        import json
        shape = [edge(30, 14, 30, 'x', 'm', 'x_axis'),
                 edge(12, 1, 3, 'y', 'angstrom', 'y_axis')]
        topic = 'topic'
        expected = {
            "module": 'hs00',
            "config": {
                "topic": topic,
                "source": 'mccode-to-kafka',
                "data_type": 'double',
                "error_type": 'double',
                "edge_type": 'double',
                "shape": shape
            }
        }
        self.assertEqual(json.dumps(nexus_structure(topic, shape)), json.dumps(expected))


if __name__ == '__main__':
    unittest.main()
