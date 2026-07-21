# Dataset audit — 20 July 2026

## Hugging Face snapshots

### `datasets/ai-generated-ecommerce-images`

- Source: `JoyCN/ai-generated-ecommerce-images`
- Hub commit: `0ede5c4c360e58a1010bd63169abee81e374684d`
- Repository files: 6,034
- Repository bytes: 6,666,500,474
- Images: 6,031 (4,977 JPEG and 1,054 PNG)
- Classes: 12
- `annotations_ai.jsonl`: 6,031 valid rows and 6,031 unique image references
- Integrity: zero missing files, zero extra dataset files, zero size mismatches, zero image decode errors
- Annotation linkage: zero missing images, zero duplicate references, zero unreferenced images
- Dataset card license: CC0-1.0
- Important limitation: synthetic AI-generated evidence images, not genuine insurance-fraud labels

### `datasets/comprehensive-car-damage`

- Source: `SaiVaibhavS/comprehensive-car-damage`
- Hub commit: `c6334d8e2eebeeeeffe2ca0fd6384ff0cbd0849a`
- Repository files: 2,303
- Repository bytes: 668,831,778
- Images: 2,300 (1,990 JPEG and 310 PNG)
- Classes: `F_Breakage` 500, `F_Crushed` 400, `F_Normal` 500, `R_Breakage` 300, `R_Crushed` 300, `R_Normal` 300
- Integrity: zero missing files, zero extra dataset files, zero size mismatches, zero image decode errors
- Dataset card license: MIT
- Important limitation: damage classes are not fraud labels. The source/provenance and downstream commercial-use rights should be reviewed independently before publication or deployment.

The `.cache/huggingface` content inside each snapshot is resumable-download metadata and is not counted as dataset content.

## Existing local files

### `datasets/INSURBIAS`

- Source: `feihuangfh/INSURBIAS`
- Selectively downloaded files: `README.md` and `Dataset/input_dataset.csv`
- Claims: 1,388
- Fields: 12
- Embedded images: 1,388 JPEGs, all decoded successfully
- Narratives: 1,388, with no missing values
- Missing tabular values: vehicle/coverage fields 84 each; amount/claim 11 each
- Important limitations: narratives are AI-generated, `Severity` is constant (`minor`), and the data does not provide verified fraud outcomes. It is suitable for multimodal claim-processing and fairness experiments, not direct fraud-model training.
- License note: benchmark components are stated as CC BY 4.0; original accident-image source terms still require independent review.

### `fraud_oracle.csv`

- Rows: 15,420
- Columns: 33
- Duplicate rows: 0
- Missing/`?` cells: 0
- Target: `FraudFound_P`
- Class distribution: 923 fraud and 14,497 non-fraud (5.99% fraud)
- Structural result: complete rectangular CSV

### `insurance_claims.csv`

- Rows: 1,000
- Columns: 39
- Duplicate rows: 0
- Target: `fraud_reported`
- Class distribution: 247 fraud and 753 non-fraud
- Structural result: complete rectangular CSV
- Data-quality issue: 881 cells use `?` as a missing-value marker:
  - `collision_type`: 178
  - `police_report_available`: 343
  - `property_damage`: 360

These markers must be converted to missing values inside a training-only preprocessing pipeline; they should not be treated as ordinary categories without an explicit modelling decision.

### `insurance-fraud-master.zip`

- ZIP entries: 21
- Uncompressed bytes: 635,944
- Archive integrity: passed; no corrupt member detected
- Contents: Memgraph synthetic insurance-fraud generator/demo project

## Breakdown implications

The downloaded files support the breakdown's separate benchmark design:

- Benchmark A: tabular fraud prediction using `fraud_oracle.csv` and `insurance_claims.csv`
- Benchmark B: damage/evidence image classification using the two Hugging Face snapshots
- Benchmark D: synthetic linked-claim graph experiments using `insurance-fraud-master.zip`

They do not form a genuine row-level multimodal insurance dataset. Images must not be randomly attached to tabular claims and represented as real linked evidence.
