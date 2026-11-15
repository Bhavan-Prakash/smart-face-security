import unittest

import numpy as np

from face_engine import FaceEngine, UNKNOWN


class DatabaseIdentityIndexTests(unittest.TestCase):
    def setUp(self):
        self.engine = object.__new__(FaceEngine)

    def test_database_person_id_maps_through_match(self):
        vector = np.zeros(512, dtype=np.float32)
        vector[3] = 1.0
        self.engine.load_identity_embeddings([
            {"person_id": "81", "name": "Person One", "embedding": vector},
        ])
        person_id, name, similarity = self.engine.match(vector)
        self.assertEqual(person_id, "81")
        self.assertEqual(name, "Person One")
        self.assertAlmostEqual(similarity, 1.0)

    def test_multiple_embeddings_keep_the_same_database_identity(self):
        first = np.zeros(512, dtype=np.float32)
        first[0] = 1.0
        second = np.zeros(512, dtype=np.float32)
        second[1] = 1.0
        self.engine.load_identity_embeddings([
            {"person_id": "92", "name": "Person Two", "embedding": first},
            {"person_id": "92", "name": "Person Two", "embedding": second},
        ])
        person_id, name, _ = self.engine.match(second)
        self.assertEqual((person_id, name), ("92", "Person Two"))

    def test_empty_identity_index_returns_unknown(self):
        self.engine.load_identity_embeddings([])
        person_id, name, similarity = self.engine.match(np.ones(512, dtype=np.float32))
        self.assertIsNone(person_id)
        self.assertEqual(name, UNKNOWN)
        self.assertEqual(similarity, 0.0)

    def test_invalid_vector_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.load_identity_embeddings([
                {"person_id": "1", "name": "Test", "embedding": np.ones(128, dtype=np.float32)},
            ])


if __name__ == "__main__":
    unittest.main()
