# Challenge Extensions
These extensions go beyond the base assignment requirements and explore more practical, real-world machine learning workflows.


## Tier 1 — Threshold Optimization for Deployment
In the base task, predictions are made using the default threshold of 0.5. However, in real-world applications, the decision threshold should be aligned with business constraints.
In this extension, I evaluated model performance across thresholds from 0.1 to 0.9. For each threshold, I computed:
- Precision
- Recall
- F1 score
- Expected number of alerts per 1,000 customers
A key business constraint was introduced:
The retention team can contact at most 150 customers per month (equivalent to 15 alerts per 1,000 customers).
Based on this constraint, I selected the threshold that:
- Satisfies the capacity limit
- Maximizes recall

### Result
The selected threshold was approximately:
- Threshold ≈ 0.70
- Alerts per 1,000 ≈ 10
- Recall is low but within operational constraints

### Insight
Lower thresholds increase recall but generate too many alerts, exceeding operational capacity. Higher thresholds reduce workload but miss more churners. This demonstrates that threshold selection is a business decision, not just a modeling choice.



## Tier 2 — Permutation Importance and Model Explanation
To better understand how models make decisions, I used permutation importance, which measures how much each feature contributes to model performance.
Unlike built-in feature importance in tree models, permutation importance is:
- Model-agnostic
- More reliable for comparison across models
I computed permutation importance for the top models using the test set.

### Key Findings
The most important features were:
- num_support_calls
- contract_months
- monthly_charges

### Interpretation
- Customers with more support calls are more likely to churn.
- Short contract duration (e.g., month-to-month) is strongly associated with churn.
- Pricing (monthly charges) influences customer decisions.

### Model Comparison Insight
While both models identified similar important features, the tree-based model captured more complex relationships between features, which helped it achieve better performance.



## Tier 3 — Model Selection Framework
Instead of hardcoding models directly in the script, I implemented a configuration-driven model selection system.

### Key Features
- Models are defined in a JSON configuration file
- The system dynamically builds pipelines from configuration
- Supports multiple model types without modifying core code
- Automatically:
  - Runs cross-validation
  - Generates comparison tables
  - Logs experiment results
  - Saves outputs to timestamped directories

### Example Capabilities
- Easily add new models (e.g., Gradient Boosting) by updating the config file
- Run multiple experiments with different configurations
- Keep results organized and reproducible

### Insight
This approach reflects real-world ML engineering practices, where:
- Experiments are configurable
- Code is reusable
- Results are traceable


## Summary
These extensions demonstrate how to move from a basic modeling task to a more production-oriented workflow by:
- Optimizing decision thresholds based on business constraints
- Interpreting model behavior using permutation importance
- Designing a flexible and scalable model selection system
These extensions provide a deeper understanding of how machine learning systems are applied in practice.