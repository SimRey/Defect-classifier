import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class Pipeline:
    """
    The functionality of the following Pipeline was defined by the previous EDA performed on the training dataset
    This Pipeline was built in order to process different dataframes to make them analysis ready
    """

    def __init__(self, df, is_train):
        self.df = df
        self.is_train = is_train


        self.X_cont = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas','X_Perimeter', 
                'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 
                'Length_of_Conveyer','Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index',
                'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index','LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
                'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas']
        self.X_cat = ["steel_type", "Outside_Global_Index"]

        self.y_cols = ['Pastry','Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
       

    def preprocess(self):
        # Filters

        if self.is_train:
            filt_pa = self.df["Pixels_Areas"] <= int(30e3)
            self.df = self.df[filt_pa]

            filt_xp = self.df["X_Perimeter"] <= int(2e3)
            self.df = self.df[filt_xp]

            filt_sl = self.df["Sum_of_Luminosity"] < int(0.38e7)
            self.df = self.df[filt_sl]

        # Steel type
        self.df["steel_type"] = self.df.apply(lambda x: "A300" if x["TypeOfSteel_A300"] == 1 
            else ("A400" if x["TypeOfSteel_A400"] == 1 else "Other"), axis=1)
        self.df.drop(columns=["TypeOfSteel_A300", "TypeOfSteel_A400", "id"], inplace=True)
       
        self.df['Outside_Global_Index'] = np.where(self.df['Outside_Global_Index'] == 0.7, 
            0.5, self.df['Outside_Global_Index'])
        
        self.df.reset_index(drop=True, inplace=True)
        
        return self.df
    

    def label_encode(self, label_encoders={}):
        if self.is_train:
            for col in self.X_cat:
                label_encoder = LabelEncoder()
                self.df[col] = label_encoder.fit_transform(self.df[col])
                self.df[col] = self.df[col].astype("category")
                label_encoders[col] = label_encoder
        else:
            for col in self.X_cat:
                label_encoder = label_encoders[col]
                self.df[col] = label_encoder.transform(self.df[col])
                self.df[col] = self.df[col].astype("category")
        
        return self.df, label_encoders
    

    def scaling(self, scaler=None):
        if self.is_train:
            scaler = MinMaxScaler()
            X = self.df[self.X_cont]
            X = scaler.fit_transform(X)          
            X = pd.DataFrame(data=X, columns=self.X_cont)        
        else:
            X = self.df[self.X_cont]
            X = scaler.transform(X)          
            X = pd.DataFrame(data=X, columns=self.X_cont)

        return X, scaler
    
    def faults_encoder(self):
        def encoder(x):
            if x["Pastry"] == 1:
                return "Pastry"
            elif x["Z_Scratch"] == 1:
                return "Z_Scratch"
            elif x["K_Scatch"] == 1:
                return "K_Scatch"
            elif x["Stains"] == 1:
                return "Stains"
            elif x["Dirtiness"] == 1:
                return "Dirtiness"
            elif x["Bumps"] == 1:
                return "Bumps"
            elif x["Other_Faults"] == 1:
                return "Other_Faults"
            else:
                return "No_Faults"

        self.df["faults"] = self.df.apply(encoder, axis=1)    
        
        fe = LabelEncoder()
        self.df["faults"] = fe.fit_transform(self.df["faults"])
        return self.df, fe
        
      

    def run(self, label_encoders=None, scaler=None):
        if self.is_train:
            self.df = self.preprocess()
            self.df, label_encoders = self.label_encode()
            X, scaler = self.scaling()

            df_train = X 
            df_train.loc[:, self.X_cat] = self.df[self.X_cat]
            self.df, fe = self.faults_encoder()
            df_train.loc[:, "faults"] = self.df["faults"]

            df_train = df_train.drop(columns=["X_Maximum", "Y_Maximum", "Luminosity_Index", "Log_Y_Index", 
                "Maximum_of_Luminosity", "LogOfAreas"]) # Columns dropped due to VIF analysis

            return df_train, label_encoders, scaler, fe
        
        else:
            self.df = self.preprocess()
            self.df, _ = self.label_encode(label_encoders)
            X, _ = self.scaling(scaler)

            df_test = X 
            df_test.loc[:, self.X_cat] = self.df[self.X_cat]
            
            df_test = df_test.drop(columns=["X_Maximum", "Y_Maximum", "Luminosity_Index", "Log_Y_Index", 
                "Maximum_of_Luminosity", "LogOfAreas"]) # Columns dropped due to VIF analysis

            return df_test
