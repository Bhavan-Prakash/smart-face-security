import unittest
from unittest.mock import patch

import cv2
import numpy as np

import enrollment_service


class EnrollmentServiceValidationTests(unittest.TestCase):
    def setUp(self):
        enrollment_service.jobs.clear()

    def test_corrupt_image_fails_the_job_instead_of_creating_embeddings(self):
        enrollment_service._generate_job("101", [("image/jpeg", b"not-an-image")])
        result = enrollment_service.jobs["101"]
        self.assertEqual(result["status"], "failed")
        self.assertIn("not a supported image", result["error"])
        self.assertNotIn("embeddings", result)

    def test_validated_image_job_produces_one_normalized_512_vector(self):
        ok, encoded = cv2.imencode(".png", np.zeros((112, 112, 3), dtype=np.uint8))
        self.assertTrue(ok)
        face = np.zeros(15, dtype=np.float32)
        face[2:4] = (80, 80)
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        with patch.object(enrollment_service.engine, "detect", return_value=np.asarray([face])), \
             patch.object(enrollment_service.engine, "embed", return_value=vector):
            enrollment_service._generate_job("102", [("image/png", encoded.tobytes())])
        result = enrollment_service.jobs["102"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["dimension"], 512)
        self.assertEqual(len(result["embeddings"]), 1)
        self.assertAlmostEqual(float(np.linalg.norm(result["embeddings"][0])), 1.0)

    def test_multiple_faces_reject_the_entire_enrollment(self):
        ok, encoded = cv2.imencode(".png", np.zeros((112, 112, 3), dtype=np.uint8))
        self.assertTrue(ok)
        face = np.zeros(15, dtype=np.float32)
        face[2:4] = (80, 80)
        with patch.object(enrollment_service.engine, "detect", return_value=np.asarray([face, face])):
            enrollment_service._generate_job("103", [("image/png", encoded.tobytes())])
        result = enrollment_service.jobs["103"]
        self.assertEqual(result["status"], "failed")
        self.assertIn("exactly one face", result["error"])
        self.assertNotIn("embeddings", result)


if __name__ == "__main__":
    unittest.main()
