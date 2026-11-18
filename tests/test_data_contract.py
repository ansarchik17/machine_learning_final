"""Check leakage-sensitive data contracts using only real UCI observations."""

import unittest

import numpy as np

from src.data import NOMINAL, engineer_features, feature_profiles, fixed_split, load_data, preprocessor


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = engineer_features(load_data())
        cls.profiles = feature_profiles(cls.frame)
        cls.splits, cls.manifest = fixed_split(cls.frame)

    def test_split_coverage_and_feature_availability(self):
        memberships = list(self.splits.values())
        self.assertEqual(sum(map(len, memberships)), len(self.frame))
        self.assertEqual(len(set(np.concatenate(memberships))), len(self.frame))
        for columns in self.profiles.values():
            self.assertNotIn("Target", columns)
            self.assertNotIn("Debtor", columns)
            self.assertNotIn("Tuition fees up to date", columns)
        self.assertFalse(any("2nd sem" in c for c in self.profiles["plus_semester_1"]))
        self.assertFalse(any("sem (" in c for c in self.profiles["non_semester"]))

    def test_preprocessing_learns_only_training_statistics(self):
        columns = self.profiles["plus_semester_1"]
        train = self.frame.loc[self.splits["train"], columns]
        validation = self.frame.loc[self.splits["validation"], columns]
        transform = preprocessor(columns).fit(train)
        numeric = transform.named_transformers_["numeric"]
        selected = next(cols for name, _, cols in transform.transformers_ if name == "numeric")
        np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, train[selected].median().to_numpy())
        np.testing.assert_allclose(numeric.named_steps["scaler"].mean_, train[selected].mean().to_numpy(), atol=1e-12)
        self.assertEqual(numeric.named_steps["scaler"].n_samples_seen_, len(train))
        means = numeric.named_steps["scaler"].mean_.copy()
        transformed = transform.transform(validation)
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_array_equal(means, numeric.named_steps["scaler"].mean_)
        encoder = transform.named_transformers_["nominal"]
        for column, categories in zip([c for c in columns if c in NOMINAL], encoder.categories_, strict=True):
            self.assertEqual(set(categories), set(train[column]))

    def test_engineering_is_finite_and_respects_zero_enrollment(self):
        for semester in ("1st", "2nd"):
            prefix = f"Curricular units {semester} sem"
            zero = self.frame[f"{prefix} (enrolled)"] == 0
            rate = self.frame[f"{prefix} (pass_rate)"]
            self.assertTrue(np.isfinite(rate).all())
            self.assertTrue((rate[zero] == 0).all())
            self.assertTrue((self.frame.loc[zero, f"{prefix} (zero_enrollment)"] == 1).all())
            np.testing.assert_allclose(rate[~zero], self.frame.loc[~zero, f"{prefix} (approved)"] / self.frame.loc[~zero, f"{prefix} (enrolled)"])


if __name__ == "__main__":
    unittest.main()
