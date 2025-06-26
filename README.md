# Project Setup and Instructions

This repository is part of a master's thesis focused on incorporating fairness into machine learning.  
The project explores methods for incorporating various fairness definitions into model training through a modified loss function, enabling multi-objective optimization that balances predictive accuracy with selected fairness criteria.  
The goal is to significantly improve fairness while minimizing the loss in accuracy.

## Create and activate a virtual environment

1. Navigate to your working directory:
```bash
cd "working directory"
```

2. Create a virtual environment:
```bash
python -m venv afenv
```

3. Activate the virtual environment:
```bash
afenv\Scripts\activate
```

4. Install the required dependencies:
```bash
pip install -r requirements.txt
```

5. To run the confusion_matrix_generator.py script:
```bash
python confusion_matrix_generator.py 30 8
```
Generates all possible distributions of 30 classification outcomes across an 8-cell confusion matrix representing binary classification results for protected and unprotected groups
The results are saved in Parquet format and used for large-scale fairness measures analysis.

6. To run the fairness_measures_analysis.py script:
```bash
python fairness_measures_analysis.py
```
Analyzes the relationship between model predictive performance metrics (Accuracy, Precision, Recall) and fairness measures (Accuracy Equality, Statistical Parity, Equal Opportunity, Predictive Equality) using synthetic confusion matrices. 
Outputs include heatmaps, histograms and summary tables.

7. To run the experiment_adult_income_dataset.py script:
```bash
python experiment_adult_income_dataset.py
```
Performs a grid search over model configurations defined in `config.yaml` using the Adult Income dataset.  
Each configuration trains a model (e.g., MLP, Logistic Regression) with a fairness-aware loss function, parameterized by different fairness definitions (AE, SP, EOP, PE) and alpha weighting strategies (e.g., constant, linear increase, linear decrease). 
At the end of training, results (CSV) and three best performing models are saved: the ones that achieved the highest accuracy, highest fairness, and best combined score.

8. To run the experiments_results_visualizations.py script:
```bash
python experiments_results_visualizations.py
```
Loads results from the experiment CSV and visualizes performance–fairness trade-offs. 
Includes grouped summaries, line plots over alpha values, and Pareto front scatter plots for model comparison.

To launch Jupyter Notebooks and start exploring:
```bash
jupyter notebook
```
As an alternative to running scripts, you can use the interactive notebooks available in the notebooks/ directory.
They are ideal for step-by-step exploration, visualization, and experimentation.
