# Test Data: Floating Roller Peel (Synthetic)

"
    "This folder contains fully synthetic, make-believe adhesive datasets for local SME testing.
"
    "No real product or test-lab records are included.

"
    "## Files
"
    "- `floating_roller_peel_main.csv` (420 rows): main campaign for ANOVA/ANCOVA workflow testing.
"
    "- `floating_roller_peel_retest.csv` (120 rows): retest campaign with slight process drift.
"
    "- `floating_roller_peel_pilot_small.csv` (54 rows): small-N pilot with a few missing values for health-check edge cases.

"
    "## Suggested response variable
"
    "- `PeelStrength_N_per_25mm`

"
    "## Useful columns for modeling
"
    "- Primary factor candidates: `Adhesive`, `Substrate`, `SurfacePrep`, `Primer`, `CureProfile`
"
    "- Group variables: `AdhesiveLot`, `Shift`, `Operator`
"
    "- Continuous covariates: `DwellTime_hr`, `Bondline_um`, `CoatingWeight_gsm`, `Temp_C`, `Humidity_pct`, `PeelRate_mm_min`
"
    "- Datetime fields: `RunTimestamp`, `LabDay`
"
    "- High-uniqueness text fields (expected to be excluded by SME): `TestRecordID`, `TraceabilityTag`
"
    