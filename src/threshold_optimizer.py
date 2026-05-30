# src/threshold_optimizer.py
def find_optimal_threshold(y_true, y_pred_proba, 
                          cost_fn_miss=10,  # Cost untuk miss fall (high!)
                          cost_fn_false_alarm=1):  # Cost untuk false alarm
    """
    Select threshold berdasarkan cost-sensitive analysis
    Critical untuk fall detection: miss detection lebih berbahaya dari false alarm
    """
    thresholds = np.arange(0.2, 0.8, 0.01)
    costs = []
    
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Cost calculation
        total_cost = (fn * cost_fn_miss) + (fp * cost_fn_false_alarm)
        costs.append(total_cost)
    
    # Find threshold with minimum cost
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    
    # Visualisasi
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, marker='o')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
               label=f'Optimal: {optimal_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Total Cost')
    plt.title('Cost-Sensitive Threshold Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/05_threshold_optimization.png', dpi=300, bbox_inches='tight')
    
    print(f"✅ Optimal Threshold: {optimal_threshold:.2f}")
    print(f"   Expected Cost: {costs[optimal_idx]:.2f}")
    
    return optimal_threshold