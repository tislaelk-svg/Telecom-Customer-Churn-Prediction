import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

SEED = 42

# 1. Load and merge datasets
client = pd.read_csv('Client.csv')
record = pd.read_csv('Record.csv')
df = pd.merge(client, record, on='Customer_ID', how='inner')

# 2. Feature engineering
def sdiv(a, b):
    return a / (b.abs() + 1)

def add_features(d):
    d = d.copy()
    d['rev_per_mou'] = sdiv(d['rev_Mean'], d['mou_Mean'])
    d['total_drop_block'] = d['drop_vce_Mean'] + d['blck_vce_Mean']
    d['mou_per_call'] = sdiv(d['mou_Mean'], d['totcalls'])
    d['rev_per_call'] = sdiv(d['rev_Mean'], d['totcalls'])
    d['eqp_months_ratio'] = sdiv(d['eqpdays'], d['months'])
    d['ovr_rev_ratio'] = sdiv(d['ovrrev_Mean'], d['rev_Mean'])
    d['totmrc_to_rev'] = sdiv(d['totmrc_Mean'], d['rev_Mean'])
    d['change_mou_rev'] = d['change_mou'] * d['change_rev']
    d['drop_block_rate'] = sdiv(d['total_drop_block'], d['plcd_vce_Mean'])
    d['custcare_per_call'] = sdiv(d['custcare_Mean'], d['totcalls'])
    d['rev_trend_x_age'] = d['change_rev'] * d['eqpdays']
    return d.replace([np.inf, -np.inf], np.nan)

df_fe = add_features(df)

num_cols = [
    'rev_Mean', 'mou_Mean', 'totmrc_Mean', 'da_Mean', 'ovrmou_Mean',
    'ovrrev_Mean', 'roam_Mean', 'change_mou', 'change_rev', 'drop_vce_Mean',
    'blck_vce_Mean', 'unan_vce_Mean', 'plcd_vce_Mean', 'recv_vce_Mean',
    'comp_vce_Mean', 'custcare_Mean', 'eqpdays', 'months', 'uniqsubs',
    'totcalls', 'totmou', 'totrev', 'adjrev', 'adjmou', 'avgrev', 'avgmou',
    'rev_per_mou', 'total_drop_block', 'mou_per_call', 'rev_per_call',
    'eqp_months_ratio', 'ovr_rev_ratio', 'totmrc_to_rev', 'change_mou_rev',
    'drop_block_rate', 'custcare_per_call', 'rev_trend_x_age'
]
cat_cols = [
    'new_cell', 'dualband', 'refurb_new', 'hnd_webcap', 'ownrent',
    'marital', 'creditcd', 'area', 'asl_flag', 'crclscod'
]
num_cols = [c for c in num_cols if c in df_fe.columns]
cat_cols = [c for c in cat_cols if c in df_fe.columns]

X = df_fe[num_cols + cat_cols].copy()
X[cat_cols] = X[cat_cols].astype('object').fillna('MISSING')
y = df_fe['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ],
    verbose_feature_names_out=False,
)
cat_mask = [False] * len(num_cols) + [True] * len(cat_cols)

model_hgb = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(
        categorical_features=cat_mask, max_iter=1000, learning_rate=0.02, random_state=SEED
    )),
])

model_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=15, random_state=SEED, n_jobs=-1
    )),
])

ensemble_clf = VotingClassifier(
    estimators=[('hgb', model_hgb), ('rf', model_rf)], voting='soft'
)

print("Training ensemble model...")
ensemble_clf.fit(X_train, y_train)

# Evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = cross_val_predict(ensemble_clf, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
thresholds = np.linspace(0.2, 0.8, 121)
best_t = max(thresholds, key=lambda t: accuracy_score(y_train, oof >= t))

ensemble_probs = ensemble_clf.predict_proba(X_test)[:, 1]
ensemble_preds = (ensemble_probs >= best_t).astype(int)

print(f'Test ROC-AUC: {roc_auc_score(y_test, ensemble_probs):.4f}')
print(f'Test Accuracy: {accuracy_score(y_test, ensemble_preds):.4f}')
