
# SentinelAI — Login-Free, Fixed-Dataset Flask IDS

This build is locked to the supplied **DrDoS_DNS_R2l_imbala.csv** dataset.

### Dataset used
- 29,999 records
- 15 numeric network-flow features
- Target: `label`
- `BENIGN` = 0
- `DrDoS_DNS` = 1
- `R2L` = 1

### No manual data
The website intentionally has:
- no login/register page
- no manual feature-entry form
- no CSV upload control
- no random/generated traffic

The Prediction and Live Detection pages read rows directly from the bundled CSV.

### ML ensemble
- Random Forest
- Extra Trees
- Logistic Regression
- Soft voting: average of the three attack probabilities
- Threshold: 0.5

### Run
```bash
pip install -r requirements.txt
python app.py
```

Open:
`http://127.0.0.1:5000`

### Retrain
The supplied dataset is already included:
```bash
python train_model.py
```
This overwrites the model with a newly trained model using only that CSV.


## Fixed data mode

The application is permanently configured to use:
`data/DrDoS_DNS_R2l_imbala.csv`

Do not upload another dataset and do not enter feature values manually.
Prediction, Live Detection, Dataset, Dashboard, and History use this fixed project dataset.


## Fixed model

The `models/ensemble_intrusion_model.pkl` file has been replaced with the exact PKL supplied by the user.
PKL SHA256: `1e91ed5bc9f6742300a33191c35ec60f7305e77b74e9cdb9cfed0794b4ac4441`
