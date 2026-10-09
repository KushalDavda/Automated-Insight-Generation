"""
Unit Tests for Equilym Healthcare Analytics Engine
Tests correctness of Parts A, B, C, D, and E against assessment criteria.
"""

import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics_engine import HealthcareAnalyticsEngine, NUMERICAL_INDICATORS


class TestHealthcareAnalyticsEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "healthcare_district_data.csv")
        cls.engine = HealthcareAnalyticsEngine(data_path=csv_path)

    def test_part_a_data_validation(self):
        """Test Part A: Data loading, shape, and zero missing values."""
        report = self.engine.get_validation_report()
        self.assertEqual(report["total_rows"], 12)
        self.assertEqual(report["total_columns"], 6)
        # Verify 0 missing values across all columns
        for col, missing_cnt in report["missing_values_per_column"].items():
            self.assertEqual(missing_cnt, 0, f"Column {col} has unexpected missing values")

    def test_part_b_trend_detection_ahmedabad(self):
        """Test Part B: Ahmedabad ANC dropped by 18.82%, flagged as significant."""
        trends = self.engine.detect_trends(threshold_pct=10.0)
        ahmedabad_anc = trends[
            (trends["district"] == "Ahmedabad") & (trends["indicator"] == "anc_coverage")
        ]
        self.assertFalse(ahmedabad_anc.empty, "Ahmedabad ANC trend should exist")
        row = ahmedabad_anc.iloc[0]
        self.assertAlmostEqual(row["pct_change"], -18.82, places=1)
        self.assertTrue(row["is_significant"], "Ahmedabad ANC should be flagged as significant")

    def test_part_c_outlier_detection_mehsana(self):
        """Test Part C: Mehsana ANC 42 is flagged as an outlier."""
        outliers_iqr = self.engine.detect_outliers(method="IQR", iqr_multiplier=1.5)
        mehsana_outlier = outliers_iqr[
            (outliers_iqr["district"] == "Mehsana") & (outliers_iqr["indicator"] == "anc_coverage")
        ]
        self.assertFalse(mehsana_outlier.empty, "Mehsana ANC = 42 must be detected as an outlier")
        self.assertEqual(mehsana_outlier.iloc[0]["value"], 42)

    def test_part_d_correlation_flagging(self):
        """Test Part D: Pearson correlation detects strong relationships (|r| >= 0.70)."""
        corr_matrix = self.engine.compute_correlation_matrix()
        self.assertEqual(corr_matrix.shape, (4, 4))
        
        flagged, warning = self.engine.detect_high_correlations(threshold_r=0.70)
        self.assertFalse(flagged.empty, "Strong correlations must be detected")
        self.assertIn("STATISTICAL LIMITATION NOTE", warning)

    def test_part_e_insights_schema(self):
        """Test Part E: All required fields present and severity is valid."""
        insights = self.engine.generate_all_insights()
        required_fields = [
            "insight_id", "type", "indicator", "entity",
            "period", "value", "prev_value", "change_pct", "severity", "explanation"
        ]
        for field in required_fields:
            self.assertIn(field, insights.columns, f"Missing required field: {field}")

        valid_types = {"trend", "outlier", "correlation", "threshold_breach"}
        for t in insights["type"].unique():
            self.assertIn(t, valid_types)

        valid_severities = {"Low", "Medium", "High"}
        for s in insights["severity"].unique():
            self.assertIn(s, valid_severities)

        # Check that INS-0001 format is maintained
        self.assertTrue(insights["insight_id"].str.startswith("INS-").all())


if __name__ == "__main__":
    unittest.main()
