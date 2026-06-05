import shap
print("use KernelExplainer to SHAP analysis ...")

def model_predict(data):
    return clf_best.predict(data)
background_data = X_train_final.sample(min(100, len(X_train_final)), random_state=42)

explainer = shap.KernelExplainer(model_predict, background_data)

shap_values = explainer.shap_values(X_train_final[:80])  

mean_shap_values = np.mean(np.abs(shap_values), axis=0)  

feature_importance_df = pd.DataFrame({
    'Feature': X_train_final.columns,
    'Mean_SHAP_Value': mean_shap_values
})

feature_importance_df = feature_importance_df.sort_values('Mean_SHAP_Value', ascending=False)

print("=== SHAP特征重要性表格 (Mean SHAP Value) ===")
print(feature_importance_df.to_string(index=False))
print("\n" + "="*50)


feature_importance_df.to_csv('SHAP_feature_importance_incorp.csv', index=False)
print("The feature importance table has been saved as: SHAP_feature_importance_incorp.csv")

# SHAP summary plot (SHAP value impact on model output)
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_train_final[:80], show=False)
plt.xlabel('SHAP Value (average impact on model output)',
           fontdict={'family': 'Arial', 'size': 18, 'weight': 'bold'})
plt.ylabel('Features',
           fontdict={'family': 'Arial', 'size': 18, 'weight': 'bold'})
plt.tick_params(axis='both', labelsize=14, labelcolor='black')
plt.xticks(fontname='Arial', fontweight='bold')
plt.yticks(fontname='Arial', fontweight='bold')
plt.tight_layout()
plt.savefig('SHAP_summary_OIR_XGB.png', dpi=400, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10, 6))
bars = plt.barh(feature_importance_df['Feature'][::-1], feature_importance_df['Mean_SHAP_Value'][::-1],
                color='#1f77b4', alpha=0.8)
plt.xlabel('Mean |SHAP Value| (impact on model output)',
           fontdict={'family': 'Arial', 'size': 18, 'weight': 'bold'})
plt.ylabel('Features',
           fontdict={'family': 'Arial', 'size': 18, 'weight': 'bold'})
plt.tick_params(axis='both', labelsize=14, labelcolor='black')
plt.xticks(fontname='Arial', fontweight='bold')
plt.yticks(fontname='Arial', fontweight='bold')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)

for bar, value in zip(bars, feature_importance_df['Mean_SHAP_Value'][::-1]):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f'{value:.2f}', ha='left', va='center', fontweight='bold', family='Arial', fontsize=10)

plt.tight_layout()
plt.savefig('SHAP_bar_OIR_XGB_with_values.png', dpi=400, bbox_inches='tight')
plt.show()


print("\n=== Example of SHAP scores for a single sample (first test sample) ===")
sample_idx = 0  
sample_data = X_test_eval.iloc[[sample_idx]] 
sample_shap_values = explainer.shap_values(sample_data)

print(f"Actual sample value: {Y_test_eval[sample_idx]:.4f}")
print(f"Predicted sample value: {clf_best.predict(sample_data)[0]:.4f}")
print("\nSHAP value decomposition:")
for feature, shap_val in zip(X_test_eval.columns, sample_shap_values[0]):
    print(f"{feature}: {shap_val:.4f}")

print(f"\nBase prediction (expected value): {explainer.expected_value:.4f}")
print(f"Sum of SHAP values: {np.sum(sample_shap_values[0]):.4f}")
print(f"Final prediction = Base + SHAP sum = {explainer.expected_value + np.sum(sample_shap_values[0]):.4f}")