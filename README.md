# Load Forecasting Web System


  The project aims to use machine learning technology to achieve high-precision day-ahead load forecasting, to provide a reliable basis for market scheduling and market pricing, and to provide support for the formulation of urban power equipment maintenance plans. 

  At present, the following forecasing algorithms are built into the system:  
  (a) Multiple linear regression: MLR;  
  (b) Gradient boosting algorithm: XGBoost, LightGBM;  
  (c) Integrated learning algorithm: Stacking;  
  (d) Deep learning: FC_DNN, LSTM, RNN, TCN, Seq2Seq, Attention_Seq2Seq;  
  (e) Algorithm proposed by this project: proposed_1, proposed_2.  
  
  Since meteorological information has an important influence on the accuracy of short-term load forecasting, the system also provides an analysis module to analyze the relationship between load curves and meteorological conditions. 
 
 # Usage
 
 1. Load Forecasting  
    -Enter the city number to be predicted # eg: P, C1  
    -Enter the date to be predicted # format: XXXX-XX-XX, eg: 2018-01-10  
    -Input prediction algorithm # eg: FC_DNN  
    -Click the "Forecast" button to make a forecast  
    -Waiting for the prediction result to return  

2. Analysis  
   -Enter the city number to be analyzed # eg: P, C1  
   -Enter the start date of the analysis # format: XXXX-XX-XX, eg: 2018-01-01  
   -Enter the deadline for analysis # format: XXXX-XX-XX, eg: 2018-01-31  
   -Click the "Analyze" button to analyze  
   -Waiting for the analysis result to return
   

# Packages Requirement

tensorflow==1.9.0  
numpy==1.16.0  
dtw==1.4.0  
tqdm==4.59.0  
pandas_stubs==1.1.0.5  
seaborn==0.11.1  
matplotlib==3.3.4  
Flask==1.1.2  
pandas==1.2.4  
scikit_learn==0.24.2  
