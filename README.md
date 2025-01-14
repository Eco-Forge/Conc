# Concrete Compressive Strength Prediction

The quality of concrete is determined by its compressive strength, which is measured using a conventional crushing test on a concrete cylinder. The strength of the concrete is also a vital aspect in achieving the requisite longevity. It will take 28 days to test strength, which is a long period.

This project uses Data Science and Machine Learning technology to predict concrete compressive strength based on the quantities of raw materials, potentially saving significant time and effort in concrete testing.

## Data Source
https://www.kaggle.com/elikplim/concrete-compressive-strength-data-set

## Prerequisites
- Anaconda or Miniconda installed (for conda package management)
- Git (for cloning the repository)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Eco-Forge/Conc.git
cd Conc
```

2. Create and activate the conda environment:
```bash
# Create the environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate conc-env
```

Note for Apple Silicon (M1/M2) Mac users: The conda-forge channel is used by default in environment.yml for better compatibility with Apple Silicon processors.

## Running the Application

1. Start the FastAPI application:
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

## Usage

1. In the web interface, enter the following concrete mixture parameters:
   - Age (in days)
   - Cement quantity (in kg)
   - Water quantity (in kg)
   - Fly ash quantity (in kg)
   - Superplasticizer quantity (in kg)
   - Blast furnace slag quantity (in kg)

2. Click "Predict" to get the estimated concrete compressive strength in MPa

3. The result will be displayed showing the predicted strength in MPa (Megapascals)

## Technical Approach

1. Data Processing:
   - Loaded dataset using Pandas
   - Performed data type validation and missing value checks
   - Handled outliers using IQR technique, particularly in the 'age' column

2. Exploratory Data Analysis:
   - Analyzed target feature distribution (Concrete compressive strength)
   - Identified key relationships:
     - Direct correlation between cement content and strength
     - Inverse correlation between water content and strength
   - Performed correlation analysis using both Pearson and Spearman methods

3. Machine Learning Implementation:
   - Experimented with multiple algorithms:
     - Linear regression with feature selection (Backward elimination, RFE, LassoCV)
     - Tree-based models with hyperparameter tuning
   - Selected XGBoost regressor as the final model based on performance metrics
   - Saved model performance metrics and feature importance in results directory

4. Model Deployment:
   - Implemented using FastAPI backend for improved compatibility
   - Modern HTML interface with Bootstrap for input and prediction display
   - ASGI server (uvicorn) for better performance

## References
 
- [Concrete Basics: Essential Ingredients For A Concrete Mixture](https://concretesupplyco.com/concrete-basics/)
- [Applications of Fly ash](https://www.thespruce.com/fly-ash-applications-844761)
- [Blast furnace slag cement](https://theconstructor.org/concrete/blast-furnace-slag-cement/23534/)
- [Applications of Superplasticizer in concrete making](https://en.wikipedia.org/wiki/Superplasticizer)
- [Factors that affect strength of concrete](https://gharpedia.com/blog/factors-that-affect-strength-of-concrete/)
- [Feature selection with sklearn and pandas](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b)
- [sklearn's LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
- [Hyper parameter tuning in XGBoost](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)

  
