# Explainable-AI-for-Credit-Risk-Management

## Overview

This project aims to use Explainable AI (XAI) techniques to enhance the transparency and interpretability of credit risk management models. By using these techniques, financial institutions can better understand their decision-making processes and ensure compliance with regulatory standards.

## Context
The rapid advancement of artificial intelligence (AI) has significantly transformed various industries, with the financial sector being one of the most impacted. Among the many applications of AI in finance, credit risk management stands out as a critical area where AI can provide substantial benefits. Accurate credit scoring models are essential for financial institutions to assess the reliability of borrowers and mitigate potential risks. However, the complexity and opacity of many AI models pose challenges in understanding and trusting their predictions, particularly in a domain where transparency is crucial.

Explainable Artificial Intelligence (XAI) has emerged as a promising solution to address these challenges. By making AI models more interpretable, XAI enables stakeholders to understand the underlying factors influencing predictions, thereby enhancing trust and facilitating better decision-making. This project aims to explore the viability of integrating XAI into credit risk management, providing insights into its practical applications and potential benefits.

## Objective
The primary objective of this project is to evaluate the feasibility and effectiveness of employing Explainable AI (XAI) in the context of credit risk management. Specifically, we want to develop a robust credit scoring models using various machine learning algorithms, including Random Forest, Support Vector Machine (SVM), XGBoost, and Neural Networks.
We want to implement and compare different XAI frameworks, such as SHAP (SHapley Additive exPlanations), Quantus, and Captum, to enhance the interpretability of the credit scoring models.
We will also analyzed the insights provided by XAI to identify critical features influencing credit risk predictions and assess the consistency and logic of the models' behavior.
Finally, the idea is to evaluate the practical implications of using XAI in credit risk management, including its potential to improve transparency, trust, and decision-making processes within financial institutions.
By achieving these objectives, the project seeks to demonstrate the value of XAI in enhancing the reliability and transparency of AI-driven credit scoring systems.

## Data

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/Data_Model.png)

## Modelling

### Model Approach

For our classification task, we have chosen LightGBM model due to its efficiency and balance between accuracy and interpretability. 
The originality, of the LightGBM is that it constructs an ensemble of decision trees sequentially. Each subsequent tree is built to correct the errors made by the previous trees. Unlike traditional decision trees, which grow level-wise, LightGBM grows trees leaf-wise. This means it expands the leaf with the maximum loss reduction, leading to better accuracy and faster training. 

### Loss function

As the goal of our classification task is to predict whether a borrower will default (class: 1) or not (class: 0), we use the Binary Cross Entropy loss function. This loss function is well-suited for binary classification tasks.

During the training process, the model tries to minimize the \textbf{\textit{Binary Cross Entropy Loss}}. This loss gives useful feedback to the model during the training process to adjust the predictions of the model by measuring the performance of a classification model whose output is a probability value between 0 and 1. It compares the predicted probabilities with the actual class labels and penalizes the model based on the difference between them. In fact, during the training process the model tries to minimize the loss function by adding new trees that predict the residual errors (gradients) of the previous trees.

Given that this loss function is convex, it is designed to be minimized thanks to a optimization algorithm such Descent Gradient (find local minimum). Lower values indicate that the predicted probabilities are closer to the actual class labels, meaning the model is performing well. It effectively penalizes large deviations between the predicted probabilities and the actual outcomes, guiding the model to improve its predictions.

$$
L = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

### Hyperparameters Fine-Tuning

To ensure that our LightGBM model performs optimally in predicting credit risk, we rely on the Optuna framework for hyperparameter optimization. Optuna is a  flexible framework that uses Bayesian optimization to search for the best hyperparameters. The objective of this optimization process is to maximize the accuracy of our model.

To do so, Optuna uses Bayesian optimization, a method that builds a probabilistic model of the objective function and uses it to select the most promising hyperparameters to evaluate in the next iteration.
* **Define the Search Space:** The range of possible values for each hyperparameter is defined. For example, this could include the learning rate, number of leaves, and maximum depth for LightGBM.
* **Objective Function:** The objective function is defined to maximize accuracy. This function trains the LightGBM model with a given set of hyperparameters and evaluates its accuracy on a validation set.
* **Bayesian Optimization:** Optuna uses past evaluation results to build a surrogate model of the objective function. It selects hyperparameters to evaluate by balancing exploration (trying new areas of the search space) and exploitation (focusing on areas known to perform well).
* **Evaluation and Iteration:** The selected hyperparameters are evaluated by training the model and calculating the accuracy. This process is iterated many times, continually updating the surrogate model and selecting new hyperparameters to evaluate.
* **Best Hyperparameters:** After a predefined number of iterations or when the improvement in accuracy plateaus, Optuna identifies the set of hyperparameters that yielded the highest accuracy.
\end{itemize}

The best hyperpameters found by Optuna are the following:

| **Hyperparameter** | **Value** |
|-------------------|-----------|
| n_estimators      | 86        |
| max_depth         | 410       |
| learning_rate     | 0.134     |
| subsample         | 0.467     |
| num_leaves        | 42        |
| feature_fraction  | 0.757     |
| sub_bin           | 84053     |

### Model Calibration

In the context of credit risk classification, it is crucial to ensure that the model's predicted probabilities are well-calibrated. Calibration is the process of aligning the predicted probabilities with the actual likelihood of the predicted classes. A well-calibrated model ensures that the predicted probabilities reflect reality and are more reliable. For instance, if a model predicts a probability of 0.8 for a borrower defaulting, it should be the case that 80\% of the time, the borrower actually defaults.

To achieve this level of reliability, we apply isotonic regression to calibrate the model. Isotonic regression is a non-parametric method that fits a piecewise constant, non-decreasing function to the predicted probabilities. The methodology involves the following steps:
* **Fit the Model:** Initially, the LightGBM model is trained and makes predictions, producing raw predicted probabilities.
* **Create Bins** The predicted probabilities are divided into bins. For each bin, we calculate the observed frequency of the positive class (e.g., default).
* **Apply Isotonic Regression** Isotonic regression is then applied to these bins. It adjusts the predicted probabilities to ensure they are monotonically increasing. This means that higher predicted probabilities correspond to higher actual frequencies of the positive class.
* **Transform Predictions:** The raw predicted probabilities are transformed using the fitted isotonic regression function. This step aligns the predicted probabilities with the actual likelihood observed in the training data.
\end{itemize}

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/calibration.png)

By applying isotonic regression, we enhance the reliability of the predicted probabilities. The calibrated probabilities provide a more accurate representation of the true risk, which is essential for making informed decisions in credit risk management. This postprocessing step ensures that the model's outputs are not only accurate but also trustworthy and actionable for stakeholders.

### Choose the decision frontier

After obtaining a calibrated model, the next step is to adjust the decision threshold for classification. Adjusting the threshold helps align the model's predictions with the bank's risk management strategy. Specifically, in the context of credit risk, the worst-case scenario is granting a loan to a person who will default. To mitigate this risk, we adjust the thresholds to maximize the recall of the model, aligning with the bank's risk tolerance and liquidity considerations.

The bank's primary concern is to minimize the occurrence of defaults. By adjusting the decision threshold, we can control the balance between precision and recall. In this case, maximizing recall (the ratio of true positives to the sum of true positives and false negatives) is crucial because it ensures that most borrowers who are likely to default are correctly identified. This reduces the risk of granting loans to high-risk individuals.

In this way, setting a lower threshold, the model becomes more sensitive to identifying defaults (positive class). This means that even if there is a slight indication of default, the model will classify the borrower as a default risk. Although this might increase false positives, the bank's strategy prioritizes avoiding defaults over misclassifying safe borrowers as risky. 

The threshold is adjusted based on the bank's risk tolerance and liquidity position. If the bank can afford to be conservative and avoid defaults at all costs, a lower threshold is appropriate. Conversely, if the bank can tolerate some defaults in exchange for higher loan approval rates, the threshold might be set higher.

Therefore, a well-calibrated model provides predicted probabilities that accurately reflect the likelihood of default. This makes the model's predictions easier to interpret. For example, if the model predicts an 0.8 probability of default, stakeholders can trust that approximately 80\% of similar cases historically resulted in defaults. This transparency helps in making informed decisions.

Knowing the exact decision threshold used by the model adds another layer of transparency. It clarifies the criteria under which loans are approved or denied. This transparency is crucial for regulatory compliance and for explaining decisions to stakeholders, including customers and auditors. Adjusting the threshold according to the bank's strategic goals ensures that the model's decisions are not only data-driven but also aligned with business objectives.

There is a trade-off between measures such as default recall, non-default recall and model accuracy. An easy way to approximate a good starting threshold value is to examine a graph of these three measures. With this graph, we can see how each of these measures looks when we change the threshold values, and find the point at which the performance of all three is good enough to be used for credit data.

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/threshold.png)

### Model Performance evaluation

#### Classification Report

| **Class**    | **Precision** | **Recall** | **F1-Score** | **Support** |
|--------------|---------------|------------|--------------|-------------|
| Non-Default  | 0.96          | 0.70       | 0.81         | 70687       |
| Default      | 0.17          | 0.70       | 0.27         | 6191        |
| **Accuracy** |               |            |              | **0.70**    |

Overall, the classification report highlights that while the model performs well in identifying non-default cases with high precision and moderate recall, its performance in identifying default cases is less satisfactory. The model has a tendency to misclassify some non-default cases as default, resulting in a relatively low precision for the default class. However, the low false negative rate suggests that the model effectively identifies the majority of default cases, which is crucial in credit risk management. Overall, there is room for improvement, particularly in enhancing the precision of default predictions, to ensure a more balanced and reliable credit risk classification system.

#### Confusion Matrix

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/confusion_matrix.png)

While the model has a good recall for the Default class, its precision is quite low, leading to many false positives. This might be acceptable in scenarios where the cost of missing a default is very high compared to the cost of incorrectly predicting a default. However, improving precision would be important to reduce the number of Non-Default instances incorrectly labeled as Default, thereby improving customer experience and reducing unnecessary credit denials.

#### ROC-AUC Curve

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/roc_curve.png)

The ROC curve shows that the model has a reasonably good performance with an AUC of 0.77. This indicates that the model is fairly effective at distinguishing between defaults and non-defaults. However, efforts can be made to further improve the model's accuracy, particularly in reducing the false positive rate while maintaining or improving the true positive rate.

#### Lift Curve

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/lift_curve.png)

At the beginning of the curve (left part), the lift is pretty high. This indicates that the model is very effective at identifying defaulters in the top-ranked predictions.The curve gradually declines as more of the population is considered. This decline occurs because as we move to the right, we are including more of the population, which will include more non-defaulters.

Overall, we can see that the lift curve remains above 1. This indicates that the model is consistently better than random guessing.

By focusing on the top-ranked predictions with the lift curve we can identify borrowers who are much more likely to default. This helps in making informed lending decisions, such as declining high-risk loans or offering them at higher interest rates to compensate for the risk.

## Credit Risk Optimization Strategy

So far, we've used simple assumptions and checks to set threshold values to determine loan status based on the predicted probability of default. With these values, we used code like this to define a new loan status based on probability and threshold. These new loan status values have an impact on our model's performance measures, as well as on the estimated financial impact on the portfolio. If we have three loans with these default probabilities, those above the threshold are considered defaults and those below are considered non-defaults.

We have seen before that our models have already predicted default probabilities, and we can use these probabilities to calculate the threshold. Since the threshold is used to determine what constitutes a default or non-default, it can also be used to approve or reject new loans as they come in. As an example, let's assume that our test set is a new batch of new loans. Before calculating the threshold, we need to understand a concept known as the acceptance rate. This is the percentage of new loans we accept in order to keep the number of defaults in a portfolio below a certain number.

To do so, we implement a strategy which consists of accepting or rejecting a loan according to an acceptance rate. The acceptance rate, is the percentage of new loans we accept in order to keep the number of defaults in a portfolio below a certain number.
By adjusting this acceptance rate, we find the best thresholds that minimize the bad rate which is the percentage of accepted loans that are defaults. Hence, minimizing the bad rate leads to maximize the portfolio estimated net value and allow us to better manage the risk according to the economic conditions. 

### Acceptance Rate

If we want to accept 85\% of all loans with the lowest probability of default, our acceptance rate is 85\%. This means that we reject 15\% of all loans with the highest probability of default. Instead of setting a threshold value, we want to calculate it to separate the loans we accept using our acceptance rate from the loans we reject. This value will not be the same as the 85\% we used as our acceptance rate.

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/hist_default.png)

Here we can see where the threshold lies within the range of predicted probabilities. We can see not only how many loans will be accepted (left-hand side), but also how many loans will be rejected (right-hand side). I recommend that you rerun this code with different threshold values to better understand how this affects the acceptance rate.

In our example, the threshold is 0.804. This means that all our new loans with a probability of default of less than 80\% are accepted, and all higher probabilities are rejected.

### Bad Rate Calculation

The calculation of the bad rate is the number of defaults in our accepted loans divided by the total number of accepted loans.

$$\frac{\text{Accepted Defaults}}{\text{Total Accepted Loans}}$$

Even if we have calculated an acceptance rate for our loans and set a threshold, there will always be defaults in our accepted loans. These are often in probability ranges where our model has not been properly calibrated. In our example, we have accepted 85\% of the loans, but not all of them are non-defaulting loans as we would like. The bad rate is the percentage of accepted loans that are actually defaults. Thus, our bad rate is a percentage of the 10,016 loans accepted.

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/bad_rate.png)

Then we look at the amount of each loan to understand the impact of acceptance rates on the portfolio. To do so, we use cross-tabulations with calculated values, such as the average loan amount, of the new set of loans. To do this, multiply the number of each loan by the average loan amount.

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/crosstab.png)

The next step is to compute the bad rate for several acceptance rate. Hence we can obtain a strategy table. To build this strategy table, for each acceptance rate we define, we calculate the threshold, store it for later, apply it to the loans to separate our set of loans into two subsets: accepted loans and rejected loans. Then we create a subset called accepted loans and we calculate and store the bad rate. According each bad rates, we compute the estimated portfolio value by calculating the difference between the average value of non-defaulting loans accepted and the average value of defaulting loans accepted.

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/strategy_table.png)

### Results of the portfolio optimization strategy

Then we visualize the results of our portfolio optimization strategy with several plots

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/port_opti_box_plot.png)

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/port_opti_curve.png)

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/strategy_table.png)

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/port_opti_plot_surface.jpg)

The final way of measuring the financial impact of our predictions is the total expected loss. This represents the amount we expect to lose if a loan defaults, given its probability of default. We take the product of the probability of default, the loss given default and the exposure at default for each loan, and add them together. In our predictive data framework, we'll use the probability of default, the exposure will be assumed to be the total value of the loan, and the loss given default will be equal to 1 for a total loss on the loan.

$$
EL = PD \times LGD \times EAD
$$

where:
*$PD$ is the Probability of Default.
*$LGD$ is the Loss Given Default (expressed as a fraction or percentage).
*$EAD$ is the Exposure at Default (the amount of money the lender is exposed to at the time of default).
\end{itemize}

The Total expecred loss of our portfolio is Total expected loss:  \$10,737,417.19

## Explainable AI (XAI)

### Why XAI matters
The significance of XAI can be seen in several areas. First, it builds trust and confidence by enabling users to understand AI decision-making processes. Second, it helps in evaluating model accuracy and fairness, ensuring that AI systems do not perpetuate biases related to race, gender, age, or location. XAI is also crucial for meeting regulatory requirements, providing transparency, and allowing individuals affected by AI decisions to challenge or comprehend those decisions. Finally, XAI supports operational accountability, enabling organizations to maintain auditability and manage risks associated with AI deployment.

### Features Permuration Importances

The importance of a feature $X_j$ is calculated as the difference between the model's performance on the original data and the performance after shuffling the values of feature $X_j$. Mathematically, it can be expressed as:

$$
\text{Importance}(X_j) = \text{Perf}_{\text{baseline}} - \text{Perf}_{\text{shuffled}(X_j)}
$$

where:
-$\text{Perf}_{\text{baseline}}$ is the performance metric (e.g., accuracy) of the model on the original data.
-$\text{Perf}_{\text{shuffled}(X_j)}$ is the performance metric after shuffling the values of feature $X_j$.

### Partial Dependence Plots

Partial dependence plots (PDPs) are a visualization technique used to understand the relationship between a feature and the predictions made by a machine learning model while marginalizing the effects of all other features. PDPs show how the predicted outcome changes as a single feature varies, while averaging out the effects of all other features.

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/partial_dependence_plot.png)

The partial dependence of the predicted outcome$\hat{y}$ on a feature$X_j$ is calculated by averaging the predictions of the model over all possible values of$X_j$, while keeping the values of all other features fixed. Mathematically, it can be expressed as:

$$
\text{Partial Dependence}(X_j) = \frac{1}{N} \sum_{i=1}^{N} \hat{y}_{\text{partial}(X_j)}(X_{\text{other}, i})
$$

where:
* $N$ is the number of observations in the dataset.
* $hat{y}_{\text{partial}(X_j)}(X_{\text{other}, i})$ is the predicted outcome when feature$X_j$ is varied across all possible values, while the values of all other features$X_{\text{other}, i}$ are fixed.
- The sum is taken over all observations in the dataset, and the average is computed.

This equation represents the partial dependence of the predicted outcome on a single feature, providing insights into how changes in that feature affect the model's predictions.

While we ran partial dependence plots for all the features in our dataset, we thought it would be more appropriate to show them for some of the features.

We can clearly notice that the higher the price of the good the power the probability of the dependence, the higher the yearly instalment, the higher the probability of default up to the threshold of 35K USD. We also notice that the higher the estimated value of the car the lower the probability of default. 

### Shapley Additive exPlanations

SHAP (SHapley Additive exPlanations) values are a technique used to explain the output of machine learning models by attributing the contribution of each feature to the model's predictions. They provide a unified measure of feature importance and explainability.

SHAP values are based on Shapley values from cooperative game theory, which allocate the contribution of each feature to the prediction by considering all possible combinations of features. They quantify the impact of each feature by comparing the model's prediction when including the feature with its actual value against a baseline prediction.

The SHAP value $\phi_j$ for a feature $X_j$ can be calculated as:

$$
\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [\text{val}(S \cup \{j\}) - \text{val}(S)]
$$

where:
- $N$ is the set of all features.
- $S$ represents a subset of features excluding $X_j$.
- $\text{val}(S \cup \{j\})$ is the model's prediction when including feature $X_j$ along with the features in subset $S$.
- $\text{val}(S)$ is the model's prediction when considering only the features in subset $S$.
- $|S|$ and $|N|$ denote the cardinality of sets $S$ and $N$, respectively.

This equation computes the SHAP value for each feature, indicating its contribution to the model's prediction. SHAP values provide insights into the importance and impact of individual features on the model's output.

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/shap_values.png)

* Features like **N\_CREDIT\_ACTIVE\_RESID**, and **YEAR\_EMPLOYED** have clear, strong impacts on the prediction, indicating their importance in the model.
* The spread of SHAP values for features like **RATIO\_CREDIT\_ANNUITY** and **MEAN\_RATE\_DOWN\_PAYMENT** suggests they have a more nuanced influence on the prediction, depending on their specific values.
* **AGE** Being older is associated with a lower impact towards predicting default. 
* Having a higher than expected number of active loans contributes significantly towards a default prediction by the model. 
* A higher duration of employment tends towards a non-default prediction. 
* Previous behavior of late payments, contributes towards a default prediction. 

### Conformal Learning

Conformal learning is a statistical framework that provides a method for constructing predictive models with reliable measures of uncertainty. It is particularly useful in situations where understanding the confidence of predictions is crucial. Unlike traditional machine learning models, which output point predictions or probabilities without a clear indication of their reliability, conformal learning aims to produce prediction sets that are valid with a specified probability.

The core idea behind conformal learning is to use past data to define a conformity measure, which quantifies how typical or atypical a new example is compared to the training data. This measure is then used to determine the prediction interval or set for new examples, ensuring that the true label is contained within this interval with a specified confidence level.

* **Conformity Score:** A measure of how typical or atypical a new credit applicant's data is compared to the historical data used to train the model. Lower scores indicate higher conformity and lower risk.
* **Calibration:** The process of adjusting conformity scores using a separate calibration set to achieve the desired coverage probability.
* **Prediction Set:** A range of possible outcomes for a new applicant that is guaranteed to include the true outcome with a specified confidence level (e.g., 95\%).

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/conformal_learning.png)

#### Coverage Values

These coverage values help evaluate the reliability and accuracy of the conformal prediction model at different confidence levels.

The statistics for each alpha value provide insights into the distribution of conformity scores relative to the specified confidence intervals. Let's interpret these results:
* **Alpha: 0.2**
   * Below: 61,502
   * Above: 15,376

**Interpretation:** 
For $\alpha = 0.2$, the threshold is set such that 80\% of the conformity scores in the calibration set fall below it. Therefore, 80\% of the predictions for the test set are within the specified confidence interval, indicating moderate confidence.

* **Alpha: 0.1**
   * Below: 69,190
   * Above: 7,688
     
**Interpretation:**
For $\alpha = 0.1$, the threshold is set to include 90\% of the conformity scores. Thus, 90\% of the test set predictions fall within the confidence interval, providing higher confidence compared to $\alpha = 0.2$.

* **Alpha: 0.05**
   * Below: 73,034
   * Above: 3,844

**Interpretation:** 
For $\alpha = 0.05$, the threshold includes 95\% of the conformity scores, resulting in 95\% of the test set predictions being within the confidence interval, indicating the highest level of confidence.

### Counterfactual Analysis with DICE

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/dicedice.png)

Counterfactual analysis is a crucial tool for enhancing decision-making processes in credit scoring. The DICE (Diverse Counterfactual Explanations) framework operates by generating alternative scenarios that could lead to different outcomes. For example, if a customer's loan application is rejected, DICE can suggest specific changes the customer can make to their financial profile to improve their chances of approval in the future.

For the bank, DICE helps identify risk boundaries, inform decision-making, and ensure compliance with regulations. By understanding the counterfactual scenarios, banks can delineate clear guidelines on what changes customers need to make to qualify for loans. This not only streamlines the approval process but also aligns with regulatory requirements by providing transparent and justifiable decisions.

For the customer, DICE offers personalized advice, enhances trust, and provides valuable data insights. By showing customers what specific actions they need to take, such as reducing debt or increasing savings, DICE empowers them with actionable steps to improve their financial standing and reapply with a higher likelihood of success. This transparency and guidance foster a stronger, trust-based relationship between the customer and the bank.

The graph depicts the flow of credit request submission, analysis by human or AI, and the outcomes based on counterfactual adjustments. Customers receive installment payments if approved, and if not, they get insights on how to improve and reapply. This cyclical process of adjustment and reapplication underscores the continuous improvement facilitated by DICE, ultimately leading to better financial health for customers and more informed lending decisions for banks.

#### How to use it : an example ?
![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/Diceexplanation.png)

Counterfactual analysis with DICE can be illustrated through the following example. 

In the provided tabs, we see a customer's credit application data and we decided to work on two scenarios. These scenarios suggest changes that could lead to different credit approval outcomes.

**1. Initial Data:** The customer's original data includes all the variables. We decided to constraint the model to only leave `AMT\_CREDIT`, `AMT\_ANNUITY`, `AMT\_GOODS\_PRICE`, `RATIO\_CREDIT\_ANNUITY`, `N\_CREDIT\_ACTIVE\_RESID`, and `CAR\_ESTIMATED\_VALUE` to vary. The initial values indicate that the customer's loan request might not be approved due to certain financial metrics and we want to give him some recommendations. 

**2. Scenario 1:** DICE proposes a counterfactual scenario where the AMT\_ANNUITY is significantly reduced to 5259.4. This adjustment increases the `RATIO\_CREDIT\_ANNUITY` to 44.8, which might improve the customer's chances of loan approval by indicating better affordability.

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/SC1.png)

**3. Scenario 2:** Another counterfactual scenario suggests a different adjustment, with `AMT\_ANNUITY` remaining high at 40320.0 but modifying `CAR\_ESTIMATED\_VALUE` to 0.063068. This scenario also modifies the `RATIO\_CREDIT\_ANNUITY` to 39.3, which could also be favorable for the customer's creditworthiness.

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/scenario2.png)

These examples demonstrate how DICE generates alternative scenarios by adjusting key financial variables. By exploring these counterfactuals, customers can understand the specific changes needed to improve their credit scores and increase their chances of loan approval. For the bank, this process helps in providing personalized advice and making informed lending decisions.

### Feature Contribution with Shapash

Shapash is a user friendly extension built on top of the Shap framework. It's main advantage is the interactive report that allows users to analyse the most influential features, with a built-in parameter that inverse transforms the preprocessing steps for easier interpretability.

#### Consistency and Logic of Predictions

As we proceed to evaluate the predictive efficacy of our models, it is imperative to consider three foundational pillars that are critical for credit assessment. These pillars not only guide our expectations but also influence how we interpret the model's performance.

**1. Stability**
The first pillar is the stability of our clients. Given the inherent risks in lending, it is crucial to minimize uncertainties. A primary source of uncertainty is the potential for significant changes in a client's circumstances, such as employment shifts or relocation. Such changes could render the initial credit assessment data obsolete, thereby increasing the financial risks over the loan's duration.

**2. Finances**
The financial soundness of the client's project is the second pillar. Questions to consider include whether the client is financially overextending themselves, whether the project is viable, and if the client can maintain their standard of living while repaying the debt on time. These considerations are vital for a thorough and effective credit assessment.

**3. Trust**
The third pillar focuses on trust. Credit granting is fraught with risks, including unpredictable risks affecting that emanates from unexpected events and accidents, and risks stemming from information asymmetries due to clients concealing relevant information. To foster a sound financial system, efforts must be made to reduce these asymmetries and cultivate trustworthy relationships with clients, thus allowing more flexibility on our part.

For each of these pillars, we have identified a few features that are directly linked to these dimensions and have looked at their impact on credit risks. 
We believe Age and the Number of years a client has been employed for to have a negative impact on the probability of default. Younger clients are more likely to experience impactful changes in their lives as they have more flexible lives. We may also expect this relationship to reverse at latter points in life as health becomes a growing concern. The same reasoning holds for the number of years a client has been employed for, the longer you have been employed the more likely you are to keep being employed. 

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/feature_contribution2.png)

Following the same thought process, we acknowledge that certain professions are more exposed to instabilities potentially due to high turnover in their workplace. We are mostly thinking about low-skill labors and unsecure industries with high turnover such as restaurants, drivers... As we can see in the following graph the model's predictions align with our hypotheses. Low-skill laborers (drivers, cleaning staff, cooking staff, low-skill laborers, laborers) are expected to have a higher probability of default by the model whereas high-skill labor (accountants, medicine staff, core staff) have a negative impact on a clients' probability of default. 

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/feature_contribution.png)

The number of excess credit a client has is expected to be indicative of over-indebtedness and we expect it to be positively correlated with one's probability of default. As we can see from the model's results, this feature is the most influential feature for predicting credit default, thus aligning with our thought process.

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/activeresid.png)

The excess amount annuity of a client could be indicative of one's overconfidence in its ability to repay its loan rapidly and thus we expect it to be positively linked to the probability of default. Once again, the model aligns with our expectations with clients having excessive annuity payments being more prone to defaulting.

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/amountann.png)

We believe that the number of days our clients misses due payments of loans that meet certain value thresholds is either linked to financial distress, or would indicate a clients lack of commitments towards his contractual obligations. There seems to be a clear positive decreasing relationships between our target and the number of days past due.

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/historicaldays.png)

On the other hand, we expect excess down payment to be a signal of our client's commitment towards his contractual obligations by having "skin in the game" and sharing part of the risk with the bank. Furthermore it also indicates that our client's is likely to have cash on hands which could serve as a safety cushion if he were to face financial difficulties. Thus we would expect it to be negatively correlated to credit default. As we can see from the following graph the relation seems clear and aligns with our thought process.

![Alt text](https://github.com/hugo-mi/Explainable_AI_for_credit_risk_management/blob/main/img/meandpayresid.png)

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/Gwenlolive/Explainable-AI-for-Credit-Risk-Management.git
   cd Explainable-AI-for-Credit-Risk-Management
   ```

2. **Install the required packages:**

   Make sure you have `pip` installed. Then run:

   ```sh
   pip install -r requirements.txt
   ```
