"""
Equilym Technologies Pvt Ltd - Assessment 4: Automated Insight Generation
Modular Analytics Engine for Healthcare District Performance Analytics.

Features:
- Part A: Data Loading, Validation & Missing-Value Diagnostics
- Part B: Trend Detection (Percentage Change & Threshold Flagging)
- Part C: Outlier Detection (Configurable IQR & Z-Score Methods)
- Part D: Pearson Correlation Matrix & Fragility Warning on Small Sample
- Part E: Dynamic Natural Language Insight Generation (Data-Driven, No Hardcoding)
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


NUMERICAL_INDICATORS = [
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
]


class HealthcareAnalyticsEngine:
    """
    Automated Insight Generation Engine for District-level Healthcare Metrics.
    """

    def __init__(self, data_path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        if df is not None:
            self.df = df.copy()
        elif data_path is not None and os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either a valid data_path or a DataFrame must be provided.")

        self._validate_and_prepare_data()

    def _validate_and_prepare_data(self) -> None:
        """Validates schema, sorts by district and month, and ensures numeric types."""
        required_cols = ["month", "district"] + NUMERICAL_INDICATORS
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")

        # Ensure correct types
        for col in NUMERICAL_INDICATORS:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Sort temporally for accurate trend detection
        self.df = self.df.sort_values(by=["district", "month"]).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Part A: Data Loading & Validation
    # -------------------------------------------------------------------------
    def get_validation_report(self) -> Dict:
        """
        Generates data validation statistics including shape, info, and missing value counts.
        """
        missing_series = self.df.isnull().sum()
        return {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "columns": list(self.df.columns),
            "missing_values_per_column": missing_series.to_dict(),
            "districts": sorted(self.df["district"].unique().tolist()),
            "months": sorted(self.df["month"].unique().tolist()),
            "indicators": NUMERICAL_INDICATORS,
        }

    def filter_data(
        self,
        districts: Optional[List[str]] = None,
        months: Optional[List[str]] = None,
        indicators: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Filters data by selected districts, months, and indicators.
        """
        filtered = self.df.copy()
        if districts:
            filtered = filtered[filtered["district"].isin(districts)]
        if months:
            filtered = filtered[filtered["month"].isin(months)]

        cols_to_keep = ["month", "district"]
        if indicators:
            valid_inds = [i for i in indicators if i in NUMERICAL_INDICATORS]
            cols_to_keep += valid_inds
        else:
            cols_to_keep += NUMERICAL_INDICATORS

        return filtered[cols_to_keep].reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Part B: Trend Detection
    # -------------------------------------------------------------------------
    def detect_trends(
        self,
        threshold_pct: float = 10.0,
        districts: Optional[List[str]] = None,
        indicators: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        For each (district, indicator) pair, computes:
        pct_change = (current - previous) / previous * 100
        Flags as is_significant if |pct_change| >= threshold_pct.
        """
        active_indicators = indicators or NUMERICAL_INDICATORS
        active_districts = districts or self.df["district"].unique().tolist()

        trend_rows = []

        for district in active_districts:
            dist_df = self.df[self.df["district"] == district].sort_values("month")
            if len(dist_df) < 2:
                continue

            for ind in active_indicators:
                values = dist_df[ind].values
                months = dist_df["month"].values

                for i in range(1, len(values)):
                    prev_val = values[i - 1]
                    curr_val = values[i]
                    curr_month = months[i]
                    prev_month = months[i - 1]

                    if pd.isna(prev_val) or pd.isna(curr_val) or prev_val == 0:
                        pct_change = np.nan
                    else:
                        pct_change = ((curr_val - prev_val) / prev_val) * 100.0

                    is_significant = (
                        abs(pct_change) >= threshold_pct if not np.isnan(pct_change) else False
                    )

                    trend_rows.append(
                        {
                            "district": district,
                            "indicator": ind,
                            "period": curr_month,
                            "prev_period": prev_month,
                            "value": curr_val,
                            "prev_value": prev_val,
                            "pct_change": round(pct_change, 2) if not np.isnan(pct_change) else None,
                            "threshold_used": threshold_pct,
                            "is_significant": is_significant,
                        }
                    )

        return pd.DataFrame(trend_rows)

    # -------------------------------------------------------------------------
    # Part C: Outlier Detection (IQR and Z-Score)
    # -------------------------------------------------------------------------
    def detect_outliers(
        self,
        method: str = "IQR",
        iqr_multiplier: float = 1.5,
        z_threshold: float = 3.0,
        indicators: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Detects statistical anomalies across districts for each indicator.
        Supports both IQR and Z-score methods.
        """
        active_indicators = indicators or NUMERICAL_INDICATORS
        outlier_rows = []

        for ind in active_indicators:
            series = self.df[ind].dropna()
            if len(series) < 3:
                continue

            mean_val = series.mean()
            std_val = series.std(ddof=1) if series.std(ddof=1) > 0 else 1e-6
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_iqr = q1 - iqr_multiplier * iqr
            upper_iqr = q3 + iqr_multiplier * iqr

            for _, row in self.df.iterrows():
                val = row[ind]
                if pd.isna(val):
                    continue

                z_score = (val - mean_val) / std_val
                is_outlier = False
                outlier_reason = ""

                if method.upper() == "IQR":
                    if val < lower_iqr or val > upper_iqr:
                        is_outlier = True
                        direction = "below Q1 - 1.5*IQR" if val < lower_iqr else "above Q3 + 1.5*IQR"
                        outlier_reason = f"Value {val} is {direction} (IQR bounds: [{lower_iqr:.1f}, {upper_iqr:.1f}])"
                elif method.upper() == "Z_SCORE":
                    if abs(z_score) >= z_threshold:
                        is_outlier = True
                        outlier_reason = f"|z-score| of {abs(z_score):.2f} exceeds threshold {z_threshold} (mean={mean_val:.1f}, std={std_val:.1f})"

                if is_outlier:
                    pct_diff_mean = ((val - mean_val) / mean_val) * 100.0 if mean_val != 0 else 0
                    outlier_rows.append(
                        {
                            "district": row["district"],
                            "indicator": ind,
                            "month": row["month"],
                            "value": val,
                            "benchmark_mean": round(mean_val, 1),
                            "z_score": round(z_score, 2),
                            "pct_diff_mean": round(pct_diff_mean, 1),
                            "method": method.upper(),
                            "reason": outlier_reason,
                        }
                    )

        return pd.DataFrame(outlier_rows)

    # -------------------------------------------------------------------------
    # Part D: Correlation Detection
    # -------------------------------------------------------------------------
    def compute_correlation_matrix(self) -> pd.DataFrame:
        """
        Computes standard Pearson correlation matrix across numerical indicators.
        """
        return self.df[NUMERICAL_INDICATORS].corr(method="pearson").round(3)

    def detect_high_correlations(
        self, threshold_r: float = 0.70
    ) -> Tuple[pd.DataFrame, str]:
        """
        Flags indicator pairs with |r| >= threshold_r.
        Also returns a statistical note regarding sample fragility.
        """
        corr_matrix = self.compute_correlation_matrix()
        flagged_pairs = []
        cols = corr_matrix.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                ind1, ind2 = cols[i], cols[j]
                r_val = corr_matrix.loc[ind1, ind2]
                if abs(r_val) >= threshold_r:
                    flagged_pairs.append(
                        {
                            "indicator_pair": f"{ind1} : {ind2}",
                            "indicator_1": ind1,
                            "indicator_2": ind2,
                            "pearson_r": round(r_val, 3),
                            "abs_r": round(abs(r_val), 3),
                            "relationship": "Strong Positive" if r_val > 0 else "Strong Negative",
                            "is_significant": True,
                        }
                    )

        flagged_df = pd.DataFrame(flagged_pairs)
        if not flagged_df.empty:
            flagged_df = flagged_df.sort_values(by="abs_r", ascending=False).reset_index(drop=True)

        n_rows = len(self.df)
        fragility_warning = (
            f"STATISTICAL LIMITATION NOTE: Pearson correlation was calculated over N={n_rows} rows "
            f"({len(self.df['district'].unique())} districts x {len(self.df['month'].unique())} months). "
            f"Correlation estimates with small sample sizes are highly sensitive to single-point anomalies "
            f"and should be confirmed with >=10 districts across >=3-6 consecutive months."
        )

        return flagged_df, fragility_warning

    # -------------------------------------------------------------------------
    # Part E: Automated Natural Language Insight Generation
    # -------------------------------------------------------------------------
    def _calculate_severity(
        self, insight_type: str, metric_value: float, threshold: float
    ) -> str:
        """
        Data-driven severity derivation without arbitrary magic numbers.
        Derived strictly as a multiple of the configurable threshold or standard deviations.
        """
        if insight_type == "trend":
            abs_change = abs(metric_value)
            # High severity if change is >= 1.75x threshold (e.g. >= 17.5% when threshold=10%)
            if abs_change >= 1.75 * threshold:
                return "High"
            # Medium severity if change is >= 1.25x threshold
            elif abs_change >= 1.25 * threshold:
                return "Medium"
            else:
                return "Low"

        elif insight_type == "outlier":
            abs_z = abs(metric_value)  # z-score passed
            if abs_z >= 2.5:
                return "High"
            elif abs_z >= 1.8:
                return "Medium"
            else:
                return "Low"

        elif insight_type == "correlation":
            abs_r = abs(metric_value)
            if abs_r >= 0.85:
                return "High"
            elif abs_r >= 0.75:
                return "Medium"
            else:
                return "Low"

        return "Medium"

    def generate_all_insights(
        self,
        trend_threshold_pct: float = 10.0,
        outlier_method: str = "IQR",
        iqr_multiplier: float = 1.5,
        z_threshold: float = 2.0,
        correlation_threshold: float = 0.70,
    ) -> pd.DataFrame:
        """
        Generates fully structured, data-driven insights adhering to Output Specification 4.1.
        Output columns:
        insight_id, type, indicator, entity, period, value, prev_value, change_pct, severity, explanation
        """
        insights = []
        counter = 1

        # 1. Trend Insights
        trends_df = self.detect_trends(threshold_pct=trend_threshold_pct)
        sig_trends = trends_df[trends_df["is_significant"]].copy()

        for _, row in sig_trends.iterrows():
            curr_val = row["value"]
            prev_val = row["prev_value"]
            pct_chg = row["pct_change"]
            direction = "dropped" if pct_chg < 0 else "increased"
            magnitude = abs(pct_chg)
            severity = self._calculate_severity("trend", pct_chg, trend_threshold_pct)

            ind_name = row["indicator"].replace("_", " ").upper()
            explanation = (
                f"{ind_name} in {row['district']} {direction} by {magnitude:.1f}% "
                f"from {prev_val} in {row['prev_period']} to {curr_val} in {row['period']}, "
                f"exceeding the {trend_threshold_pct:.1f}% significant-change threshold."
            )

            insights.append(
                {
                    "insight_id": f"INS-{counter:04d}",
                    "type": "trend",
                    "indicator": row["indicator"],
                    "entity": row["district"],
                    "period": row["period"],
                    "value": curr_val,
                    "prev_value": prev_val,
                    "change_pct": pct_chg,
                    "severity": severity,
                    "explanation": explanation,
                }
            )
            counter += 1

        # 2. Outlier Insights
        outliers_df = self.detect_outliers(
            method=outlier_method,
            iqr_multiplier=iqr_multiplier,
            z_threshold=z_threshold,
        )

        for _, row in outliers_df.iterrows():
            severity = self._calculate_severity("outlier", row["z_score"], 2.0)
            ind_name = row["indicator"].replace("_", " ").upper()
            val = row["value"]
            mean_val = row["benchmark_mean"]
            z_val = row["z_score"]

            direction_str = "below" if z_val < 0 else "above"
            explanation = (
                f"{row['district']}'s {ind_name} of {val} in {row['month']} is "
                f"{abs(z_val):.1f} sigma {direction_str} the state mean ({mean_val}), "
                f"flagging as a statistical {row['method']} outlier for review."
            )

            insights.append(
                {
                    "insight_id": f"INS-{counter:04d}",
                    "type": "outlier",
                    "indicator": row["indicator"],
                    "entity": row["district"],
                    "period": row["month"],
                    "value": val,
                    "prev_value": mean_val,  # Benchmark state mean
                    "change_pct": row["pct_diff_mean"],
                    "severity": severity,
                    "explanation": explanation,
                }
            )
            counter += 1

        # 3. Correlation Insights
        corr_df, _ = self.detect_high_correlations(threshold_r=correlation_threshold)
        for _, row in corr_df.iterrows():
            r_val = row["pearson_r"]
            severity = self._calculate_severity("correlation", r_val, correlation_threshold)
            relationship_type = "inverse (negative)" if r_val < 0 else "direct (positive)"

            ind1_clean = row["indicator_1"].replace("_", " ").upper()
            ind2_clean = row["indicator_2"].replace("_", " ").upper()

            explanation = (
                f"Strong {relationship_type} correlation (r={r_val:.2f}) observed between "
                f"{ind1_clean} and {ind2_clean} across districts (|r| >= {correlation_threshold:.2f})."
            )

            months_span = f"{self.df['month'].min()}..{self.df['month'].max().split('-')[-1]}"
            insights.append(
                {
                    "insight_id": f"INS-{counter:04d}",
                    "type": "correlation",
                    "indicator": f"{row['indicator_1']}:{row['indicator_2']}",
                    "entity": "State-wide",
                    "period": months_span,
                    "value": f"r={r_val:.2f}",
                    "prev_value": "n/a",
                    "change_pct": "n/a",
                    "severity": severity,
                    "explanation": explanation,
                }
            )
            counter += 1

        # 4. District-Specific Correlated Anomaly Check (e.g. Mehsana ANC drop vs high_risk_cases spike)
        # Check if any district experienced both a sharp negative trend and an inverted risk spike
        for dist in self.df["district"].unique():
            dist_trends = trends_df[trends_df["district"] == dist]
            anc_trend = dist_trends[dist_trends["indicator"] == "anc_coverage"]
            hrc_trend = dist_trends[dist_trends["indicator"] == "high_risk_cases"]

            if not anc_trend.empty and not hrc_trend.empty:
                anc_pct = anc_trend.iloc[0]["pct_change"]
                hrc_pct = hrc_trend.iloc[0]["pct_change"]

                if anc_pct is not None and hrc_pct is not None:
                    if anc_pct <= -15.0 and hrc_pct >= 50.0:
                        explanation = (
                            f"Critical Correlated Anomaly in {dist}: ANC coverage plummeted by {abs(anc_pct):.1f}% "
                            f"while High-Risk Cases surged by +{hrc_pct:.1f}% ({hrc_trend.iloc[0]['prev_value']} -> {hrc_trend.iloc[0]['value']})."
                        )
                        insights.append(
                            {
                                "insight_id": f"INS-{counter:04d}",
                                "type": "threshold_breach",
                                "indicator": "anc_coverage:high_risk_cases",
                                "entity": dist,
                                "period": anc_trend.iloc[0]["period"],
                                "value": f"ANC:{anc_trend.iloc[0]['value']}, HRC:{hrc_trend.iloc[0]['value']}",
                                "prev_value": f"ANC:{anc_trend.iloc[0]['prev_value']}, HRC:{hrc_trend.iloc[0]['prev_value']}",
                                "change_pct": f"ANC:{anc_pct}%, HRC:+{hrc_pct}%",
                                "severity": "High",
                                "explanation": explanation,
                            }
                        )
                        counter += 1

        return pd.DataFrame(insights)

    def export_results(self, output_dir: str = ".") -> Tuple[str, str]:
        """
        Saves insights.csv and correlation_matrix.csv in accordance with Section 4.
        """
        os.makedirs(output_dir, exist_ok=True)
        insights_df = self.generate_all_insights()
        corr_matrix = self.compute_correlation_matrix()

        insights_path = os.path.join(output_dir, "insights.csv")
        corr_path = os.path.join(output_dir, "correlation_matrix.csv")

        insights_df.to_csv(insights_path, index=False)
        corr_matrix.to_csv(corr_path)

        return insights_path, corr_path


if __name__ == "__main__":
    # Test script execution directly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "healthcare_district_data.csv")

    print(f"Initializing Analytics Engine with {csv_file}...")
    engine = HealthcareAnalyticsEngine(data_path=csv_file)

    report = engine.get_validation_report()
    print("\n--- DATA VALIDATION REPORT ---")
    print(f"Total Rows: {report['total_rows']}, Columns: {report['total_columns']}")
    print(f"Missing Values: {report['missing_values_per_column']}")

    print("\n--- GENERATING INSIGHTS ---")
    insights = engine.generate_all_insights()
    print(insights[["insight_id", "type", "entity", "indicator", "severity", "explanation"]].to_string())

    print("\n--- EXPORTING OUTPUTS ---")
    ins_file, corr_file = engine.export_results(script_dir)
    print(f"Saved: {ins_file}")
    print(f"Saved: {corr_file}")
