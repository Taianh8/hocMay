"""Validation-only hyperparameter selection."""
from time import perf_counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def candidates(spec: dict, seed: int):
    """Yield estimators and their small, explicit parameter grids."""
    if spec['model'] == 'knn':
        for params in ParameterGrid({'n_neighbors': spec['neighbors']}):
            yield params, KNeighborsClassifier(**params, n_jobs=1)
    elif spec['model'] == 'tree':
        for params in ParameterGrid({'max_depth': spec['max_depth'], 'min_samples_leaf': spec['min_samples_leaf']}):
            yield params, DecisionTreeClassifier(**params, random_state=seed)
    else:
        raise ValueError(f"Unknown model: {spec['model']}")

def search(X_train, y_train, X_val, y_val, spec: dict, seed: int):
    """Fit only on train; rank on validation macro F1 then accuracy, stable grid order breaks ties."""
    records, best, best_score = [], None, (-1., -1.)
    for params, model in candidates(spec, seed):
        start = perf_counter(); model.fit(X_train, y_train); train_time = perf_counter() - start
        start = perf_counter(); pred = model.predict(X_val); pred_time = perf_counter() - start
        score = (f1_score(y_val, pred, average='macro', zero_division=0), accuracy_score(y_val, pred))
        records.append({'params': params, 'val_macro_f1': score[0], 'val_accuracy': score[1], 'train_seconds': train_time, 'val_predict_seconds': pred_time})
        if score > best_score:
            best, best_score = model, score
    return best, records
