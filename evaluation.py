import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc
)


def evaluate_and_report(y_train, y_test, y_pred_custom, y_proba_custom, 
                       y_pred_sklearn, y_proba_sklearn, weights_custom, feature_names):

    acc_custom = accuracy_score(y_test, y_pred_custom)
    prec_custom = precision_score(y_test, y_pred_custom)
    rec_custom = recall_score(y_test, y_pred_custom)
    f1_custom = f1_score(y_test, y_pred_custom)
    auc_custom = roc_auc_score(y_test, y_proba_custom)
    
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    prec_sklearn = precision_score(y_test, y_pred_sklearn)
    rec_sklearn = recall_score(y_test, y_pred_sklearn)
    f1_sklearn = f1_score(y_test, y_pred_sklearn)
    auc_sklearn = roc_auc_score(y_test, y_proba_sklearn)
    
    baseline_acc = max(y_train.mean(), 1 - y_train.mean())
    cm_custom = confusion_matrix(y_test, y_pred_custom)
    

    print(f"\nBaseline (majority class): {baseline_acc:.4f}")
    
    print("\nCustom Implementation:")
    print(f"  Accuracy:  {acc_custom:.4f} (+{acc_custom - baseline_acc:.4f} vs baseline)")
    print(f"  Precision: {prec_custom:.4f}")
    print(f"  Recall:    {rec_custom:.4f}")
    print(f"  F1-Score:  {f1_custom:.4f}")
    print(f"  ROC-AUC:   {auc_custom:.4f}")
    
    print("\nsklearn:")
    print(f"  Accuracy:  {acc_sklearn:.4f} (+{acc_sklearn - baseline_acc:.4f} vs baseline)")
    print(f"  Precision: {prec_sklearn:.4f}")
    print(f"  Recall:    {rec_sklearn:.4f}")
    print(f"  F1-Score:  {f1_sklearn:.4f}")
    print(f"  ROC-AUC:   {auc_sklearn:.4f}")
    
    print(f"\nDifference (Custom - sklearn):")
    print(f"  Accuracy:  {acc_custom - acc_sklearn:+.4f}")
    print(f"  Precision: {prec_custom - prec_sklearn:+.4f}")
    print(f"  Recall:    {rec_custom - rec_sklearn:+.4f}")
    print(f"  F1-Score:  {f1_custom - f1_sklearn:+.4f}")
    print(f"  ROC-AUC:   {auc_custom - auc_sklearn:+.4f}")
    
  
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_custom, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['No Sauce', 'Sauce'])
    plt.yticks([0, 1], ['No Sauce', 'Sauce'])
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm_custom[i, j]), ha='center', va='center', 
                    color='white' if cm_custom[i, j] > cm_custom.max()/2 else 'black', 
                    fontsize=24, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.jpg', dpi=300, bbox_inches='tight')
    plt.close()

    fpr_custom, tpr_custom, _ = roc_curve(y_test, y_proba_custom)
    fpr_sklearn, tpr_sklearn, _ = roc_curve(y_test, y_proba_sklearn)
    
    roc_auc_custom_curve = auc(fpr_custom, tpr_custom)
    roc_auc_sklearn_curve = auc(fpr_sklearn, tpr_sklearn)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_custom, tpr_custom, color='#1f77b4', lw=2.5, 
             label=f'Custom Implementation (AUC = {roc_auc_custom_curve:.4f})')
    plt.plot(fpr_sklearn, tpr_sklearn, color='#ff7f0e', lw=2.5, linestyle='--',
             label=f'sklearn LogisticRegression (AUC = {roc_auc_sklearn_curve:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Logistic Regression Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nROC Curves saved as 'roc_curve.jpg'")

    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': weights_custom
    }).sort_values('coefficient', key=abs, ascending=False)
    
    coef_df.to_csv('coefficients.csv', index=False)

    
    print("\nTop 10 Features (increase/decrease probability of Crazy Sauce):")
    for _, row in coef_df.head(10).iterrows():
        effect = "increases" if row['coefficient'] > 0 else "decreases"
        print(f"  {row['feature']:30s} {row['coefficient']:+.4f} ({effect})")
