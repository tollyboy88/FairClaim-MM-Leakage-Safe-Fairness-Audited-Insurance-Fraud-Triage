You are right—the stronger direction is to transform your existing project into a **multimodal, fairness-aware insurance fraud detection system** covering vehicle claims, mobile-device claims, claim notes, images, histories, and linked evidence. 

# **What I found in your repository** 

I inspected the complete GitHub repository. It currently contains: 

- `fraud_oracle.csv` 

- One large notebook: `Untitled.ipynb` 

- 15,420 motor-insurance claims 

- 33 original variables 

- 923 fraudulent claims and 14,497 non-fraudulent claims 

- Fraud prevalence of approximately **5.99%** 

- Random Forest, XGBoost and SVM code, although these sections are commented out 

- Gaussian Naive Bayes as the model producing the saved results 

- Manual subgroup evaluations for sex, age and marital status 

The reported Gaussian Naive Bayes result is approximately: 

- Accuracy: 82.3% 

- Recall: 91.1% 

- Precision: 77.5% 

- ROC-AUC: 0.918 

However, these results are currently unreliable because of data leakage. 

# **Critical problems to correct first** 

## **1. SMOTE is applied before splitting** 

The notebook oversamples the complete dataset and then creates the train/test split. This allows synthetic information derived from test observations to enter training. 

Correct order: 

```
Original data
    ↓
Train–validation–test split
    ↓
Fit preprocessing on training data
    ↓
Apply SMOTE only to training data
    ↓
Train model
    ↓
Evaluate on untouched test data
```

SMOTE and preprocessing should be contained inside an `imblearn.Pipeline` . 

## **2. The test data imputer is refitted** 

The notebook uses: 

```
X_test = imputer.fit_transform(X_test)
```

It should be: 

```
X_test = imputer.transform(X_test)
```

The test set must never fit preprocessing parameters. 

## **3. Ordinal numbers are assigned to nominal categories** 

`LabelEncoder` gives artificial order to variables such as vehicle make, policy type and accident area. A model may incorrectly interpret one category as numerically greater than another. 

Use: 

- `OneHotEncoder` for Logistic Regression, SVM and neural networks 

- Native categorical handling with CatBoost 

- Carefully encoded categories for LightGBM/XGBoost 

## **4. Fairness metrics are incomplete** 

The notebook calculates subgroup accuracy, recall, precision and positive prediction rate, but it does not calculate the actual gaps between groups. 

Add: 

- Demographic-parity difference and ratio 

- Equal-opportunity difference 

- False-positive-rate difference 

- Equalised-odds difference 

- Predictive-parity difference 

- Group calibration 

- Subgroup PR-AUC and ROC-AUC 

- Bootstrap confidence intervals 

- Intersectional results, such as sex × age 

## **5. Some confusion-matrix code uses the wrong variable** 

For example, subgroup matrices are created as `cm_male` , `cm_female` , and others, but several cells still extract values using the overall `cm` . This can report incorrect subgroup TN, FP, FN and TP values. 

## **6. Policy number should not be treated as an ordinary continuous feature** 

`PolicyNumber` has 15,420 unique values—one per record. It risks teaching the model meaningless identifier patterns. Drop it unless it is used to group repeated claims or construct a graph. 

## **7. Accuracy should not drive model selection** 

Because only 5.99% of the original claims are fraudulent, the primary measures should be: 

- PR-AUC 

- Fraud recall 

- Precision 

- F1 or F2 

- Recall at a fixed investigation capacity 

- Expected financial cost 

- Calibration/Brier score 

- False-positive burden 

- Fairness gaps 

# **Recommended upgraded project** 

A strong title would be: 

### **FairClaim-MM: A Fairness-Aware Multimodal Framework for Insurance Fraud Detection Using Structured Claims, Narrative Evidence, Images and Claim Histories** 

The system should have four separate evidence branches. 

```
flowchart TD
```

```
    A["Structured claim data"] --> E["Multimodal fusion"]
    B["Claim notes"] --> E
```

```
    C["Damage images"] --> E
```

```
    D["Claim history and networks"] --> E
    E --> F["Fraud risk and uncertainty"]
    F --> G["Approve, investigate or abstain"]
```

```
    F --> H["Fairness and explanation audit"]
```

# **Models to compare** 

|**Evidence**|**Traditional**<br>**baseline**|**Strong advanced model**|**Deep-learning model**|
|---|---|---|---|
|Tabular<br>claims|Logistic<br>Regression<br>TF-IDF +|CatBoost, XGBoost,<br>LightGBM|MLP, TabNet or FT-Transformer|
|Claim notes|Logistic<br>Regression|TF-IDF + Linear SVM|DistilBERT or MiniLM|
|Damage<br>images|HOG + SVM|Pretrained image<br>embeddings + XGBoost|EfficientNet/ConvNeXt; YOLO<br>for damage localisation|
|Claim<br>sequences|Aggregated<br>historical features|<sup>XGBoost</sup>|GRU, LSTM or Temporal<br>Transformer|
|Claimant<br>networks|Graph statistics +<br>XGBoost|<br>Community/anomaly<br>detection|GraphSAGE or Graph Attention<br>Network|
|Combined<br>evidence|Weighted score|Stacking ensemble|Multimodal fusion network|



# **Where LSTM belongs** 

Do not apply LSTM directly to the current 33-column table. The rows are independent claims, not ordered sequences. 

LSTM becomes appropriate when each policyholder has events such as: 

```
Policy created
→ address changed
→ device added
→ incident reported
→ claim submitted
→ documents uploaded
→ supplementary evidence added
→ payment requested
```

For each claimant, construct: 

```
claimant_id
event_time
event_type
claim_amount
days_since_previous_event
channel
device_or_vehicle
location
outcome
```

Then compare: 

- Engineered historical features + XGBoost 

- GRU 

- LSTM 

- Temporal Convolutional Network 

- Transformer encoder 

This gives the LSTM a genuine temporal prediction problem. 

# **Image analysis: what it can and cannot prove** 

A damage-image model should not directly say “fraud” merely because it sees damage. It should extract evidence such as: 

- Damage present or absent 

- Damage category 

- Location of damage 

- Estimated severity 

- Image duplication 

- Metadata inconsistencies 

- Agreement between image and claim description 

The fraud model can then use consistency variables: 

```
reported_damage = "rear bumper crushed"
image_damage = "front lamp broken"
evidence_consistency = low
```

That mismatch is more meaningful for fraud detection than the raw image label. 

# **Claim-note analysis** 

Add a `claim_note` field containing descriptions such as incident narratives, assessor summaries and reported damage. 

Recommended comparison: 

1. TF-IDF + Logistic Regression 

2. TF-IDF + Linear SVM 

3. MiniLM sentence embeddings + XGBoost 

4. DistilBERT fine-tuning 

5. Structured-data + text-embedding fusion 

Useful NLP-derived features include: 

- Claimed incident type 

- Damage severity 

- Dates and locations mentioned 

- Contradictions 

- Uncertainty language 

- Missing details 

- Text-to-image consistency 

- Text-to-structured-data consistency 

Avoid using a language model’s unsupported judgement that a claimant “sounds fraudulent.” Analyse verifiable inconsistencies instead. 

# **Easy datasets I recommend** 

These options do not require PhysioNet-style training, institutional approval or payment. 

|**Dataset**|**Use**|**Size/access**|**Important**<br>**limitation**|
|---|---|---|---|
|Fraud Oracle on|Your primary motor-|3.45 MB, direct|Tbl l|
|Figshare|fraud benchmark|download, CC BY 4.0|auar ony|
|Insurance Claims CSV<br>on GitHub|External tabular<br>comparison|About 1,000 rows;<br>download directly from<br>GitHub|Small and<br>probably synthetic|
|Comprehensive Car||2,300 images, 669 MB,|Damae labels|
|Damage on Hugging<br>Face|Vehicle-image branch|downloadable through<br>Hugging Face|g ,<br>not fraud labels|
|AI-generated<br>ecommerce damage|Mobile/device image<br>and text prototype|6,031 images with<br>annotations|Synthetic and not<br>directly fraud-|



|**Dataset**|**Use**|**Size/access**|**Important**<br>**limitation**|
|---|---|---|---|
|images|||labelled|
|Memgraph Insurance<br>Fraud|Claim-network and<br>organised-fraud<br>experiments|Clone repository and<br>generate data locally|Synthetic graph<br>data|



The car-damage dataset has six front/rear normal and damage classes and is approximately 669 MB, making it much easier to manage locally than larger image collections. Hugging <u>Face describes 2,300 images across six classes.</u> 

I would **not** recommend the original CarDD download route for you. Although it has rich annotations, its official source requires completing and sending a licensing form. The Hugging Face mirror is easier, but its underlying images still have non-commercial licensing restrictions. The dataset documentation confirms these restrictions. 

# **Important dataset rule** 

Do not randomly attach unrelated damage images to `fraud_oracle.csv` rows and call the result real multimodal insurance data. 

Instead, organise the research into: 

- **Benchmark A:** genuine tabular fraud prediction 

- **Benchmark B:** vehicle/mobile damage recognition 

- **Benchmark C:** synthetic claim-note and evidence-consistency experiments 

- **Benchmark D:** synthetic linked-claim network fraud 

- **Final prototype:** controlled multimodal fusion, clearly labelled partly synthetic 

This is scientifically defensible. 

# **Concrete first experiment** 

Start with a corrected tabular benchmark before adding images. 

Run these models: 

- Dummy classifier 

- Logistic Regression 

- Gaussian Naive Bayes 

- Random Forest 

- Extra Trees 

- SVM 

- XGBoost 

- CatBoost 

- Balanced Random Forest 

- MLP 

- TabNet or FT-Transformer 

### Compare four imbalance approaches: 

- No resampling with class weights 

- Random undersampling 

- SMOTENC applied only inside training folds 

- Cost-sensitive learning 

Use: 

- 60% training 

- 20% validation 

- 20% untouched test 

- Five-fold stratified cross-validation inside training 

- Fixed random seeds 

- Probability calibration 

- Threshold selected on validation data 

- Final test evaluation only once 

The best immediate upgrade is therefore **not LSTM first** . It is: 

Correct the leakage, build a reproducible traditional-versus-deep tabular benchmark, add reliable fairness evaluation, and then introduce image, text, temporal and graph evidence as separate validated modules. 

That will turn the current notebook into a credible research-grade fraud detection project rather than simply adding more models. 

