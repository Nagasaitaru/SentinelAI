
"""
Retrain SentinelAI using ONLY data/DrDoS_DNS_R2l_imbala.csv.
No manually entered feature columns are required.

The label mapping follows the paper's binary framing:
BENIGN = 0, malicious labels = 1.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE=Path(__file__).resolve().parent
DATASET=BASE/"data"/"DrDoS_DNS_R2l_imbala.csv"
if not DATASET.exists(): raise FileNotFoundError(DATASET)

df=pd.read_csv(DATASET)
features=[c for c in df.columns if c!="label"]
X=df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf],np.nan)
X=X.fillna(X.mean())
y=(df["label"].astype(str).str.upper()!="BENIGN").astype(int).to_numpy()

Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
rf=RandomForestClassifier(n_estimators=150,max_depth=12,min_samples_leaf=10,class_weight="balanced",random_state=42,n_jobs=-1)
et=ExtraTreesClassifier(n_estimators=200,max_depth=12,min_samples_leaf=5,class_weight="balanced",random_state=42,n_jobs=-1)
sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
lr=LogisticRegression(max_iter=1500)
rf.fit(Xtr,ytr); et.fit(Xtr,ytr); lr.fit(Xtr_s,ytr)
rfp=rf.predict_proba(Xte)[:,1]; etp=et.predict_proba(Xte)[:,1]; lrp=lr.predict_proba(Xte_s)[:,1]
ep=(rfp+etp+lrp)/3; pred=(ep>=.5).astype(int)

def m(y,p,prob): return {
 "accuracy":round(float(accuracy_score(y,p)),6),
 "precision":round(float(precision_score(y,p,zero_division=0)),6),
 "recall":round(float(recall_score(y,p,zero_division=0)),6),
 "f1":round(float(f1_score(y,p,zero_division=0)),6),
 "roc_auc":round(float(roc_auc_score(y,prob)),6)
}
metrics={"dataset":{"filename":DATASET.name,"rows":len(df),"features":len(features),
 "labels":{str(k):int(v) for k,v in df.label.value_counts().to_dict().items()},
 "binary_mapping":{"BENIGN":0,"DrDoS_DNS":1,"R2L":1}},
 "ensemble":{**m(yte,pred,ep),"confusion_matrix":confusion_matrix(yte,pred).tolist()},
 "Random Forest":m(yte,rf.predict(Xte),rfp),
 "Extra Trees":m(yte,et.predict(Xte),etp),
 "Logistic Regression":m(yte,lr.predict(Xte_s),lrp)}

(BASE/"models").mkdir(exist_ok=True); (BASE/"artifacts").mkdir(exist_ok=True)
joblib.dump({"random_forest":rf,"extra_trees":et,"logistic_regression":lr,"scaler":sc,
 "voting_type":"soft","threshold":.5,"feature_names":features,
 "dataset_filename":DATASET.name,"label_mapping":{"BENIGN":0,"DrDoS_DNS":1,"R2L":1}},
 BASE/"models"/"ensemble_intrusion_model.pkl")
(BASE/"artifacts"/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
print(json.dumps(metrics,indent=2))
