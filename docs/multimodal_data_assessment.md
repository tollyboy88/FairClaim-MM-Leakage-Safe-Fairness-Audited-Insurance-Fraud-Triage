# Multimodal data assessment

## Core constraint

No downloaded public dataset provides the same claim ID across verified fraud outcome, structured policy data, claimant/witness narrative, accident images, and audio. Therefore, a single end-to-end supervised fraud model cannot be trained honestly from the available files.

The prototype uses component-level evidence models and conservative workflow fusion:

1. calibrated structured fraud probability;
2. image damage classification;
3. explicit fact extraction from claimant/witness notes or a verified transcript;
4. image-to-note consistency checking;
5. escalation-only workflow fusion.

## Added dataset

### INSURBIAS

- Local path: `datasets/INSURBIAS`
- Downloaded: the core 102 MB `Dataset/input_dataset.csv` and README only
- Contains: 1,388 claims, 1,388 embedded accident JPEGs, 1,388 synthetic narratives, policy/coverage metadata
- Useful for: multimodal claim-processing and counterfactual fairness experiments
- Not suitable for: direct fraud training, because it lacks verified fraud outcomes and all severity values are `minor`

The repository's prior LLM-prediction files were not downloaded because they are not needed for training the local evidence components.

## Considered but not added

### INS-MMBench

This is a 2.90 GB insurance visual-question-answering benchmark with 12,052 images and claim-processing scenarios. It is useful for evaluating visual-language reasoning but has no verified fraud target, so it is not required for the current component models.

### AI Visual Evidence

The dataset card describes 200 authentic and 1,200 synthetic legal-evidence images. Live repository access currently returns an authentication error. Even if access is restored, it would train evidence-authenticity detection rather than claim fraud.

### ASVspoof

ASVspoof provides bonafide and spoofed/deepfake speech for audio-authenticity research. It does not label claimant truthfulness or insurance fraud. It should be considered only for a separate audio-spoof warning module.

### Whisper speech recognition

Whisper can create a local transcript from an uploaded audio note, but transcription must be human-verified before decision support because transcription errors can create false contradictions. The current CPU/Python environment does not have a compatible local speech stack installed, so the dashboard stores audio and accepts a verified transcript.

## Data that would materially unlock a real multimodal fraud model

An insurer or research partner would need to provide de-identified, consented records with stable linkage:

```text
claim_id
fraud_outcome_after_investigation
outcome_date
structured_claim_fields
claimant_note
witness_note
audio_file_or_verified_transcript
image_or_document_ids
damage_assessor_findings
investigator_findings
final_claim_outcome
protected_attributes_for_audit_only
```

Fraud outcomes must come from completed investigation/adjudication, not model suspicion. Train/validation/test splitting should group by claimant, policy, vehicle, repairer, and network where possible to prevent entity leakage.
