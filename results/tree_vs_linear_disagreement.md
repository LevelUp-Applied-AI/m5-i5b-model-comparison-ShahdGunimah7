# Tree vs. Linear Disagreement Analysis

## Sample Details

- **Test-set index:** 4060
- **True label:** 0
- **RF predicted P(churn=1):** 0.5998
- **LR predicted P(churn=1):** 0.1700
- **Probability difference:** 0.4299

## Feature Values

- **tenure:** 36.0
- **monthly_charges:** 20.0
- **total_charges:** 1077.33
- **num_support_calls:** 2.0
- **senior_citizen:** 0.0
- **has_partner:** 0.0
- **has_dependents:** 0.0
- **contract_months:** 1.0

## Structural Explanation
The Random Forest assigns a much higher churn probability than Logistic Regression for this sample (0.5998 vs 0.1700). This is likely because the tree captures a threshold-based pattern involving short contract duration (1 month) combined with moderate tenure and low charges, which places the customer in a higher-risk segment. In contrast, Logistic Regression treats features additively and cannot capture this interaction, leading to a much lower predicted probability.