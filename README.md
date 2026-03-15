This project focuses on multivariate time series analysis and forecasting using the Electricity Transformer Temperature (ETTh1) dataset. The primary objective is to build deep learning models capable of capturing seasonality and complex trends in time series data, thereby providing accurate forecasts for multiple future time steps.

## Methodological Overview

To solve this forecasting problem, the project is structured into several progressive stages, ranging from exploratory data analysis to the deployment of advanced deep learning architectures. 

Initially, exploratory data analysis (EDA) is performed to evaluate underlying data patterns. This includes assessing seasonality, checking for stationarity using the Augmented Dickey-Fuller (ADF) test, and analyzing autocorrelation and partial autocorrelation plots. Based on these analytical insights, feature engineering is applied to the raw data. New temporal features are created, such as cyclical time encoding, lag features, moving averages, and specific interaction variables, which assist the neural networks in learning complex temporal rules more effectively.

Furthermore, to enhance the overall robustness of the models and mitigate the risk of overfitting during the training phase, the project incorporates a data augmentation strategy. This is achieved by systematically injecting Gaussian noise directly into the numerical features of the training dataset, allowing the models to generalize better on unseen test data.

Regarding the core modeling phase, the system implements two distinct architectures tailored for different forecasting objectives. The first approach utilizes a Long Short-Term Memory (LSTM) network integrated with a Variational Dropout mechanism, which helps stabilize the hidden states during the recurrent learning process. The second approach tackles the more complex multi-step forecasting task by deploying a Sequence-to-Sequence (Seq2Seq) architecture paired with an Attention mechanism. To optimize the training of this Seq2Seq model, a Scheduled Sampling technique utilizing Teacher Forcing decay is implemented, significantly improving both the convergence speed and the stability of the predictions.

## Source Code Structure

The source code is organized into a modular structure to separate processing pipelines, making the project easier to maintain and extend:

```text
├── src/
│   ├── augmentation.py  (DataAugmenter class managing the generation and injection of Gaussian noise)
│   ├── config.py        (Handles hardware device configuration and fixes random seeds for reproducibility)
│   ├── data_utils.py    (Executes feature engineering, scaling, and sliding window generation)
│   ├── eda.py           (OTEDA class dedicated to time series statistical analysis and visualization)
│   ├── models.py        (Defines PyTorch architectures: VariationalDropout, LSTM, and Seq2Seq)
│   └── train_utils.py   (Constructs the training loops and evaluation metric calculations)
├── clean.ipynb          (Notebook dedicated to initial data preprocessing and resampling)
├── main.ipynb           (The central notebook integrating the entire training and evaluation pipeline)
└── README.md
