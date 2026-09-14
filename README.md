
# Project: Predicting Product Returns

## Objective
Product returns are a significant cost for Parts Avatar. We want to proactively identify orders that have a high probability of being returned. By predicting these returns, we can flag high-risk orders for review, adjust marketing, or investigate potential issues with product listings.

Your goal is to build a machine learning model that predicts whether a product sold will be returned.

## The Challenge
The provided dataset is a simplified sample of historical sales. The primary challenge is not just to build a classifier, but to engineer meaningful features, select an appropriate model and evaluation metrics (especially given that returns are less common than successful sales), and interpret the model's predictions to provide actionable business insights.

## Dataset
* `data/sales_data.csv`: A sample of historical sales transactions.

## Your Tasks
1.  **Data Exploration & Feature Engineering:**
    * Perform an exploratory data analysis (EDA) to understand the data.
    * Engineer at least two new features from the existing data that you believe could be predictive of returns. For example, you might consider price-related features or interactions between variables.

2.  **Model Training & Evaluation:**
    * Build a machine learning pipeline that preprocesses the data, trains a classification model, and evaluates its performance.
    * **Problem-Solving:** Choose a model (e.g., Logistic Regression, Random Forest, XGBoost) and justify your choice. Given the class imbalance (fewer returns), what evaluation metrics are most important (e.g., Precision, Recall, F1-Score, AUC-ROC)? Explain why accuracy alone is not a good metric here.

3.  **Interpretation & Reporting:**
    * Analyze the results of your best model. What are the most important features that predict a return?
    * Use techniques like feature importance plots or SHAP values to interpret your model's decisions.

4.  **Documentation:**
    * Update this `README.md` to be a comprehensive report of your project.
    * Include key findings from your EDA.
    * Describe your feature engineering process.
    * Justify your choice of model and evaluation metrics.
    * Present the final model's performance and, most importantly, provide **actionable recommendations** for Parts Avatar based on your model's insights. (e.g., "Our model shows that products in the 'Electronics' category over $200 have a high return probability. We should review the product descriptions for these items.").
    * Provide clear instructions on how to run your code.

## Evaluation Criteria
* **Problem-Solving & ML Concepts:** Your approach to feature engineering, model selection, and handling class imbalance.
* **ML Pipeline:** The quality and structure of your code for training and evaluation.
* **Model Evaluation & Interpretation:** Your choice of metrics and your ability to extract business insights from the model.
* **Communication & Reporting:** The clarity of your analysis and the actionability of your recommendations in the README report.

## Disclaimer: Data and Evaluation Criteria
Please be advised that the datasets utilized in this project are synthetically generated and intended for illustrative purposes only. Furthermore, they have been significantly reduced in terms of sample size and the number of features to streamline the exercise. They do not represent or correspond to any actual business data. The primary objective of this evaluation is to assess the problem-solving methodology and the strategic approach employed, not necessarily the best possible tailored solution for the data.

## Project Report
#### Data Exploration & Feature Engineering
The main findings of my EDA are:
1. Negative examples outweigh positive examples 10 to 1.
2. There are 7 different values of `product_category` and 200 different values of `product_id`
3. All possible combinations of the above are present in the dataset.
4. For each combination of `product_category` and `product_id` almost all entries within the group have distinct prices.

Here are the features I decided to add to the data and why:

1. A new product category feature I named product_id created from a combination of the product_id and product_category into a single numerical value.The formula I used to calculate this value is: 1000 * product_category (converted to int) + product_id (converted to int)
2. Percentile rank of the price of the order within all orders with the same product_category_hash.

I decided to create these two features because they give the model information about the context of each sale. In other words, the information in the original .csv file only relates to each row individually, so adding these two features allows the model to quickly evaluate if two orders are similar (two products from the same category will have similar values on the order of 10s-100s, whereas different categories will differ by thousands), and if the order is "expensive". The utility of these features will be demonstrated further along in this report.

#### Choice of Model and Evaluation Metric
The model I chose to use for this classification task is XGBoost and the evaluation metric I chose is recall. I decided to use XGBoost because it offers higher accuracy and requires less data pre-processing such as feature scaling compared to logistic regression. On the other hand, I chose recall as my target metric because the use case of predictions from this model is to flag orders with a high risk of a return occuring. Indeed, since all orders that are flagged will be manually reviewed by a human, it is best to push for the model to identify as many returns as possible, even if some orders that won't end up being returned get flagged.

Finally, accuracy is not a good metric for this particular task because of the ratio of positives to negatives in the data: the overwhelming number of negative cases allows a model trained to maximize accuracy to simply assign the negative label to all orders and achieve a 90% accuracy, despite utterly failing the task.

#### Model Interpretation
![image](https://github.com/BenjaminRoderick/product-return-prediction/blob/main/data/summary_plot.png)

I used SHAP values to evaluate the importance of different features within my model. As can be seen in the image above, I have included the summary plot of feature importance in my model as a function of the value of each feature. My analysis of this plot for each feature is as follows:
- **category_id**: The plot shows that certain values of this feature provide a very strong signal that the order will be returned or not. This indicates that, in the provided data, certain lower `category_id` items such as brakes and electronics will be returned very frequently, whereas higher codes, such as suspension and hvac, will be rarely returned.
- **is_first_time_customer**: The most clear-cut separation of all the features. Despite the signal not being as strong as some of the other features, the clear separation indicates that being a new customer will almost always increase the odds of an order being returned.
- **order_day_of_the_week**: I was surprised to see that Friday-Saturday-Sunday are the days that are most likely to result in a returned order.
- **price**: Orders with High prices carry a strong signal for a return.
- **price_percentile**: High price percentiles cluster around weak signal, whereas low price percentiles take on more extreme signal values.

To illustrate the signal of each feature more clearly, I've included the waterfall plots for each of a true negative, a true positive, a false negative and a false positive.

#### True Negative
![image](https://github.com/BenjaminRoderick/product-return-prediction/blob/main/data/true_negative.png)

#### True Positive
![image](https://github.com/BenjaminRoderick/product-return-prediction/blob/main/data/true_positive.png)

#### False Negative
![image](https://github.com/BenjaminRoderick/product-return-prediction/blob/main/data/false_negative.png)

#### False Positive
![image](https://github.com/BenjaminRoderick/product-return-prediction/blob/main/data/false_positive.png)

#### Action Items
In conclusion, the most consistent predictors of an order being returned are if the customer is making their first ever purchase and the day of the week. Other factors to remain wary of are the category of the item and the price, as they can provide a very strong signal, but are less clearly separated into positive and negative signal based on the value of the feature.

### How to run the code
```
pip install -r requirements.txt
```

```
python ./src/ml_pipeline.py
```

Then run all the cells in `evaluate_model.ipynb` to see the SHAP values and classification report for the model.
