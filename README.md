#  Customer Churn Prediction Project (XGBoost + Feature Engineering)

##  Project Overview
This project predicts customer churn based on the **Telco Customer Churn Dataset**.  
Using a full ML pipeline with `XGBoost`, we handle class imbalance, evaluate model performance (AUC 0.82), and interpret predictions through feature importance.

> 🎯 **Goal:** Help businesses identify customers likely to leave and take proactive retention actions.

...

##  Feature Importance

Top influential features:
1. Contract_Two year
2. InternetService_Fiber optic
3. Contract_One year
4. InternetService_No

> 📌 Long-term contracts significantly reduce churn risk.

![Feature Importance](assets/importance.png)

---

##  External Dashboard

🔗 [Interactive Tableau Dashboard](https://public.tableau.com/app/profile/yourname/viz/SuperstoreDashboard)

---

##  Conclusion

- Adjusting for class imbalance **improved churn recall by ~37%**
- Feature analysis reveals business drivers of churn
- Model is interpretable, actionable, and production-ready

---

##  Skills Demonstrated

-  Data preprocessing (`StandardScaler`, `OneHotEncoder`)
-  ML pipeline construction (`Pipeline`, `ColumnTransformer`)
-  XGBoost modeling + tuning
-  Model evaluation (precision, recall, ROC, AUC)
-  Interpretability (feature importance)
-  Data storytelling & presentation

---

>  Built to support churn reduction strategies in real-world scenarios.
