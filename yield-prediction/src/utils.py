import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names):
    """Plot feature importance for trained model"""
    importances = model.feature_importances_
    indices = importances.argsort()

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance (Random Forest)")
    plt.show()
