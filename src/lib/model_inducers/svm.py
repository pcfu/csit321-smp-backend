import pandas as pd
import numpy as np
import json
from .base_inducer import BaseInducer

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

"""
    SVM model is use for next day recommendation prediction
"""

class SVM(BaseInducer):
        
    def __init__(self, training_id):
        
        """
            Parameters
            ----------
            training_id: int | str
                Training id of corresponding ModelTraining in Frontend
        """

        model_save_path = f'trained_models/{training_id}'
        super().__init__(model_save_path)

    def get_data(self, stock_id, stock_id1, stock_id2, date_start, date_end, model_data_fields):
        """
            Returns a list of dicts, each containing a price history entry

            Parameters
            ----------
            stock_id: int | str
                Stock id of price histories' parent Stock in Frontend
                
            stock_id1: int | str
                Second stock from the same industry as the main stock, to get only "Close" price
                
            stock_id2: int | str
                Second stock from the same industry as the main stock, to get only "Close" price

            date_start: str
                format: 'YYYY-MM-DD'

            date_end: str
                format: 'YYYY-MM-DD'
        """
        # Set stock_id1 and stock_id2 column headers for later
        CloseID1 = 'close_1'
        CloseID2 = 'close_2'
        
        params = [ stock_id, date_start, date_end, model_data_fields ]
        params1 = [ stock_id1, date_start, date_end, ['date','close'] ]
        params2 = [ stock_id2, date_start, date_end, ['date','close'] ]
        params_ti = [ stock_id, date_start, date_end ]
        params_sis = [ stock_id, date_start, date_end ]

        res = self.frontend.get_price_histories(*params)
        res1 = self.frontend.get_price_histories(*params1)
        res2 = self.frontend.get_price_histories(*params2)
        res_ti = self.frontend.get_technical_indicators(*params_ti)
        res_sisters = self.frontend.get_sister_prices(*params_sis) # return array

        # Preprocess df (original stock)
        df = pd.json_normalize(res['price_histories'])
        df['date'] = pd.to_datetime(df['date'])
        df.reset_index(inplace=True)
        df = df.set_index('date')
        del df['index']

        df1 = pd.json_normalize(res1['price_histories'])
        df1['date'] = pd.to_datetime(df1['date'])
        df1.rename(columns={'close':CloseID1}, inplace=True)
        df1.reset_index(inplace=True)
        df1 = df1.set_index('date')
        del df1['index']

        df2 = pd.json_normalize(res2['price_histories'])
        df2['date'] = pd.to_datetime(df2['date'])
        df2.rename(columns={'close':CloseID2}, inplace=True)
        df2.reset_index(inplace=True)
        df2 = df2.set_index('date')
        del df2['index']

        # Merge the 3 dataframes
        df_merge = pd.merge(df,df1, how='inner', left_index=True, right_index=True)
        df_merge1 = pd.merge(df_merge,df2, how='inner', left_index=True, right_index=True)
        
        # Obtain TI and merge
        df_ti = pd.json_normalize(res_ti['technical_indicators'])
        df_ti['date'] = pd.to_datetime(df_ti['date'])
        df_ti.reset_index(inplace=True)
        df_ti = df_ti.set_index('date')
        del df_ti['index']
        del df_ti['id']
        del df_ti['stock_id']

        # Final merge for 3 closing price DF with TI df
        df_final = pd.merge(df_merge1,df_ti, how='inner', left_index=True, right_index=True)

        return df_final
        #if res.get('status') == 'error':
        #    raise RuntimeError(res.get('reason'))
        #return res.get('price_histories')

    def preprocess_data(self, get_df):
        """
            Setup variables to be use in helper functions
            
            Variables
            ----------
            targetPriceForward: int 
                Predict forward n days
            
            targetCol: float
                Create new column - y target column 
                
            refPriceCol / refPriceCol1 / refPriceCol2: float
                Column to reference - Close price
                
            tar_Days_close_1 / tar_Days_close_2: float
                Create new y target column for stock 2 and 3
                
            marginCol / marginCol1 / marginCol2: float
                Create margin column 
                
            opinion1 / opinion2: int
                Create opinion column to store stock 2 and 3 "recommendation"
                
            recommendationCol: int
                Create recommendation column for main stock

            Returns
            -------
            Final processed data with all the required features for model training and prediction
        """
        # To be added as adjustable variables for sys admin (KIV)
        buyThreshold = 1
        sellThreshold = -1
        targetPriceForward = 1
        
        targetCol = 'targetCol'
        refPriceCol = 'close'
        marginCol = 'margin_percent'
        recommendationCol = 'recommendation'        
        
        #Setup new column for x days predicted values
        self.newPriceColumn(get_df,-targetPriceForward,refPriceCol,targetCol)
        get_df.dropna(inplace=True)
        
        #Getting margain difference Buy Sell Recommendatiaon
        self.getMarginChange(get_df,targetCol,refPriceCol,marginCol)
        
        #Getting ND Margain Buy Sell Recommendatiaon
        self.recommendation(get_df,marginCol,recommendationCol, sellThreshold, buyThreshold)

        return get_df

    def build_train_test_data(self, data, trainSize):
        """
            Return dataframe of train and test set. 
            
            Variables
            ---------
            X = Original dataset, without the target column
            Y = Original dataset, only with target column
            trainSize = Train size percentage for the split

            Returns
            -------
            x_train, y_train, x_test and y_test
        """
        
        # Declare features and variables
        X_features=['sma_5','sma_8','sma_10','wma_5','wma_8',
                    'wma_10','stoch_k','stoch_d','williams','macd',
                    'rsi','ad','roc','cci','atr','close_1','close_2']
        Y_feature = 'recommendation'
        
        # Filter dataframe and get only required features
        X = data[X_features]
        Y = data[Y_feature]
       
        #Split to Train and Test Set
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = trainSize, shuffle=True)

        # Normalization train set
        self.scaler = MinMaxScaler(feature_range=(0, 1)) 
        arr_scaled = self.scaler.fit_transform(x_train) 
        x_train = pd.DataFrame(arr_scaled, columns=x_train.columns,index=x_train.index)
        
        # Normalization test set
        arr_scaled_test = self.scaler.transform(x_test)
        x_test = pd.DataFrame(arr_scaled_test, columns=x_test.columns,index=x_test.index)

        return [x_train, x_test, y_train, y_test]

    def build_model(self, *args):
        """
            BUild the model of SVM by initializing it
        """
        svm_model = svm.SVC()

        return svm_model

    def train_model(self, svm_model, model_params, data):
        """
            model_params['build_args']
            Train the model using the final processed data and options (in dict)

            Returns
            -------
            gs_svm: NA
                Trained and fitted model 

            best_params: dict
                Best parameters, to be display in frontend
            
            f1_score_svm: float
                performance metric on F1 micro average aka accuracy 
        """
        # Train, test datasets
        [x_train, x_test, y_train, y_test] = data

        # Get model params
        params = model_params.get('build_args')

        # Declare time series split for crossValidation
        btss = BlockingTimeSeriesSplit(n_splits=4)
        
        # Cross validation randomized gridsearch
        gs_svm = RandomizedSearchCV(estimator=svm_model, 
                                    cv=btss,
                                    param_distributions=params,
                                    scoring='accuracy')
        #Fit model
        gs_svm.fit(x_train, y_train)

        # Best hyperparameters
        best_params = gs_svm.best_params_
        
        # Get performance metric
        y_pred_svm = gs_svm.predict(x_test)
        f1_score_model = f1_score(y_test, y_pred_svm, average='micro')
        

        return {
            'model_fit': gs_svm,
            'f1_score': f1_score_model,
            'best_params': best_params
        }

    def build_prediction_data(self, get_df):
        """
            Get the latest data from the DB for prediction (last row in table)

            Returns
            -------
            A row of latest data
        """   
        # Get last entry
        X_features=['sma_5','sma_8','sma_10','wma_5','wma_8',
                    'wma_10','stoch_k','stoch_d','williams','macd',
                    'rsi','ad','roc','cci','atr','close_1','close_2']
        
        to_predict = get_df[X_features]
        arr_scaled_predicted = self.scaler.transform(to_predict) 
        to_predict = pd.DataFrame(arr_scaled_predicted, columns=to_predict.columns,index=to_predict.index)

        return to_predict

    def get_prediction(self, model_fit, data):
        """
            Perform predicting using the saved fitted model. Prediction input data will be normalize first
            then pass into the model for prediction.

            Return
            ------
            Prediction result for 1 day ahead
        """
        predicted_result = model_fit.predict(data)

        return print(predicted_result)

        if predicted_result == -1:
            return 'Sell'
        elif predicted_result == 1:
            return 'Buy'
        else:
            return 'Neutral'

    def _build_features_and_labels(self, data):
        None

    # Helper functions
    """
        Helper functions to be use in preprocessing of data.
        
        newPriceColumn = Creation of new Y target column for n forward (applying shift)
        getMarginChange = Get the difference between current close price against n forward close price in newPriceColumn
        recommendation = Create new column to determine whether the stock should buy, sell or hold
        BlockingTimeSeriesSplit = To split train set into n number of train/test set for model training
    """
    
    #Defining Function to Create New Columns of Prices for X Days
    def newPriceColumn(self, df, Xdays, getColName, setNewColName):
        shift_df = df.shift(periods=Xdays)
        df[setNewColName] = shift_df[getColName]

    #Defining Function: Buy or Sell Based on 1% Change from X day
    def getMarginChange(self, df,columnGetValues, columnCompare, colInsertName): #df,10DaysClose,Close,10DaysCloseMargin
        df[colInsertName] = 100.00
        counter = 0
        while counter<len(df):
            marginChange = ((df[columnGetValues].iloc[counter]-df[columnCompare].iloc[counter])/df[columnCompare].iloc[counter])*100
            df[colInsertName].iloc[counter] = marginChange
            counter=counter+1

    #Defining Function: Populate the recommendation of -1, 0, 1 to indicate sell, neutral or buy. 
    def recommendation(self, df, marginColName, colInsertName, sellThreshold, buyThreshold): #df,10DaysCloseMargin,recommendationCol, sellThreshold, buyThreshold
        counter = 0
        df[colInsertName] = 0
        while counter<len(df):
            if(df[marginColName].iloc[counter]<sellThreshold):
                df[colInsertName].iloc[counter]=-1
            elif (df[marginColName].iloc[counter]>buyThreshold):
                df[colInsertName].iloc[counter]=1
            else:
                df[colInsertName].iloc[counter]=0
            counter=counter+1

# Helper class
"""
    BlockingTimeSeriesSplit further split the train set into n number of train/test to train the model
"""

class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start # 20% test
            yield indices[start: mid], indices[mid + margin: stop]