from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np, json

def evaluate_all(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)

        # Directional accuracy: did the model predict the right direction?
        # More meaningful than R² for trading decisions
        actual_dir  = np.diff(y_test.values) > 0
        pred_dir    = np.diff(preds) > 0
        dir_acc     = np.mean(actual_dir == pred_dir)

        results[name] = {
            'RMSE': round(rmse, 4),
            'MAE':  round(mae, 4),
            'R2':   round(r2, 4),
            'Directional_Accuracy': round(dir_acc, 4)
        }
        print(f"{name}: R²={r2:.3f} RMSE={rmse:.4f}")

    with open('models/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results