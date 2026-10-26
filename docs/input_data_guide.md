# Claim input and upload guide

The dashboard supports either one guided claim or a CSV containing multiple claims. The **Batch upload** page generates a ready-to-edit CSV template with valid default values.

## Required score fields

Every claim requires these 28 model fields:

| Area | Fields |
|---|---|
| Incident timing | `Month`, `WeekOfMonth`, `DayOfWeek` |
| Claim timing | `DayOfWeekClaimed`, `MonthClaimed`, `WeekOfMonthClaimed` |
| Vehicle | `Make`, `VehicleCategory`, `VehiclePrice`, `AgeOfVehicle`, `NumberOfCars`, `Year` |
| Policy | `PolicyType`, `BasePolicy`, `Deductible`, `Days_Policy_Accident`, `Days_Policy_Claim`, `AgeOfPolicyHolder`, `AddressChange_Claim` |
| Incident evidence | `AccidentArea`, `Fault`, `PoliceReportFiled`, `WitnessPresent` |
| Claim history/handling | `PastNumberOfClaims`, `NumberOfSuppliments`, `AgentType`, `RepNumber`, `DriverRating` |

The spelling and category values should follow the downloadable dashboard template. Previously unseen categories are accepted by the encoder, but they should be reviewed because they may represent data-entry or schema drift.

## Optional fields

| Field | Purpose | Used in fraud score? |
|---|---|---|
| `PolicyNumber` | Local claim/reference tracing | No |
| `Sex` | Fairness monitoring | No |
| `Age` | Fairness monitoring and age-band reporting | No |
| `MaritalStatus` | Fairness monitoring | No |
| `FraudFound_P` | Optional known outcome for a labelled batch fairness audit | No |

## Single-claim evidence

The guided form accepts up to five PNG, JPEG, PDF, TXT, CSV, WAV, MP3, M4A, or OGG files, each up to 10 MB. Files are stored as BLOBs in the local SQLite database.

PNG/JPEG vehicle images are analysed by the separate six-class damage model. A high-confidence contradiction between an image and explicit claimant/witness note facts may escalate the workflow route. It does not change the calibrated structured-data fraud probability. Audio is stored locally; paste a human-verified transcript into the transcript field for the current consistency check.

## Decision meanings

- `approve`: score is below the validation-selected low-risk threshold.
- `defer`: score is between the low- and high-risk thresholds; obtain more evidence or use human review.
- `investigate`: score meets or exceeds the validation-selected investigation threshold.

These are workflow recommendations, not final fraud or coverage decisions.
