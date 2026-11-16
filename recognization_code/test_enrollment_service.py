import unittest
from unittest.mock import patch

import cv2
import numpy as np

import enrollment_service


class EnrollmentServiceValidationTests(unittest.TestCase):
    def setUp(self):
        enrollment_service.jobs.clear()

    def test_corrupt_image_fails_when_it_is_the_only_image(self):
        enrollment_service._generate_job("101", [("image/jpeg", b"not-an-image")])
        result = enrollment_service.jobs["101"]
        self.assertEqual(result["status"], "failed")
        self.assertIn("No valid face", result["error"])
        self.assertNotIn("embeddings", result)
        self.assertEqual(result["images_skipped"], 1)

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

    def test_multiple_faces_are_skipped_without_discarding_a_later_valid_image(self):
        ok, encoded = cv2.imencode(".png", np.zeros((112, 112, 3), dtype=np.uint8))
        self.assertTrue(ok)
        face = np.zeros(15, dtype=np.float32)
        face[2:4] = (80, 80)
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        with patch.object(enrollment_service.engine, "detect", side_effect=[np.asarray([face, face]), np.asarray([face])]), \
             patch.object(enrollment_service.engine, "embed", return_value=vector):
            enrollment_service._generate_job("103", [("image/png", encoded.tobytes()), ("image/png", encoded.tobytes())])
        result = enrollment_service.jobs["103"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["images_skipped"], 1)
        self.assertEqual(result["embeddings_generated"], 1)

    def test_no_face_is_skipped_and_later_face_is_enrolled(self):
        ok, encoded = cv2.imencode(".png", np.zeros((112, 112, 3), dtype=np.uint8))
        self.assertTrue(ok)
        face = np.zeros(15, dtype=np.float32)
        face[2:4] = (80, 80)
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        with patch.object(enrollment_service.engine, "detect", side_effect=[np.asarray([]), np.asarray([face])]), \
             patch.object(enrollment_service.engine, "embed", return_value=vector):
            enrollment_service._generate_job("104", [("image/png", encoded.tobytes()), ("image/png", encoded.tobytes())])
        result = enrollment_service.jobs["104"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["images_skipped"], 1)
        self.assertEqual(len(result["embeddings"]), 1)

    def test_detection_exception_is_isolated_to_its_image(self):
        ok, encoded = cv2.imencode(".png", np.zeros((112, 112, 3), dtype=np.uint8))
        self.assertTrue(ok)
        face = np.zeros(15, dtype=np.float32)
        face[2:4] = (80, 80)
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        with patch.object(enrollment_service.engine, "detect", side_effect=[RuntimeError("detector transient failure"), np.asarray([face])]), \
             patch.object(enrollment_service.engine, "embed", return_value=vector):
            enrollment_service._generate_job("105", [("image/png", encoded.tobytes()), ("image/png", encoded.tobytes())])
        result = enrollment_service.jobs["105"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["images_skipped"], 1)

    def test_corrupt_image_is_skipped_when_a_later_image_is_valid(self):
        ok, encoded = cv2.imencode(".png", np.zeros((112, 112, 3), dtype=np.uint8))
        self.assertTrue(ok)
        face = np.zeros(15, dtype=np.float32)
        face[2:4] = (80, 80)
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        with patch.object(enrollment_service.engine, "detect", return_value=np.asarray([face])), \
             patch.object(enrollment_service.engine, "embed", return_value=vector):
            enrollment_service._generate_job("106", [("image/png", b"broken"), ("image/png", encoded.tobytes())])
        result = enrollment_service.jobs["106"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["images_skipped"], 1)
        self.assertEqual(result["embeddings_generated"], 1)

    def test_all_no_face_images_fail_without_embeddings(self):
        ok, encoded = cv2.imencode(".png", np.zeros((112, 112, 3), dtype=np.uint8))
        self.assertTrue(ok)
        with patch.object(enrollment_service.engine, "detect", return_value=np.asarray([])):
            enrollment_service._generate_job("107", [("image/png", encoded.tobytes()), ("image/png", encoded.tobytes())])
        result = enrollment_service.jobs["107"]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["images_skipped"], 2)
        self.assertNotIn("embeddings", result)


if __name__ == "__main__":
    unittest.main()
