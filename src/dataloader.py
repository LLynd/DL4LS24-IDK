import os
import anndata
import pyometiff as pyff
import numpy as np
import xgboost as xgb
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class CustomDataLoader():
    def __init__(self, config):
        self.config = config
        
        self.img_data_path = os.path.join(config.img_data_path, 'img')
        self.mask_data_path = os.path.join(config.img_data_path, 'masks')
            
    def load_anndata(self):    
        return anndata.read_h5ad(self.config.ann_data_path)
    
    def transform_image(self, img):
        return np.arcsinh(img / 5.)
    
    def load_img_and_mask(self, data, imgID):
        image_name = data.obs.iloc[imgID]['image']
        
        image_path = os.path.join(self.img_data_path, image_name)
        mask_path = os.path.join(self.mask_data_path, image_name)
        
        image_reader = pyff.OMETIFFReader(fpath=image_path)
        mask_reader = pyff.OMETIFFReader(fpath=mask_path)
        
        image_array, _, _ = image_reader.read()
        mask_array, _, _ = mask_reader.read()
        
        return image_array, mask_array, image_name
        
    def load_img_batch(self, batchsize, split=False, transform=True):
        pass
    
    def get_data(self, preprocess=True):
        if self.config.method == 'xgboost' or self.config.method == 'linear':
            data = self.load_anndata()   
            if preprocess:
                data = self.preprocess_anndata_linear_xgboost(data)
            return data
        
        elif self.config.method == 'starling':
            data = self.load_anndata()
            if preprocess:
                data = self.preprocess_anndata_starling(data)
            return data
        
        elif self.config.method == 'cnn':
            pass
           
    def preprocess_anndata_linear_xgboost(self, data):
        df = data.obs
        counts = data.layers['counts']
        exprs = data.layers['exprs']
        list_cols_exprs = ['exprs'+str(i) for i in range(40)]
        df_exprs = pd.DataFrame(data=exprs, columns=list_cols_exprs, index=df.index)
        result = pd.concat([df, df_exprs], axis=1)
        df=result.copy()

        int_columns = df.select_dtypes(include='int').columns.tolist()
        df[int_columns] = df[int_columns].astype(float)
        non_float_columns = df.select_dtypes(exclude='float').columns.tolist()
        category_columns = df.select_dtypes(include='category').columns.tolist()
        ids = df['sample_id']
        Y = df['cell_labels']

        columns_to_remove=['SampleId','image','sample_id','SlideId','BatchId','SubBatchId','Batch','Box.Description','cell_labels','celltypes']

        df = df.drop(columns=columns_to_remove)
        df['CD20_patches'] = df['CD20_patches'].replace('', '0')
        df['CD20_patches'] = df['CD20_patches'].astype(float)

        category_columns = df.select_dtypes(include='category').columns.tolist()

        one_hot_encoded1 = pd.get_dummies(df['Indication'], prefix='Indication')
        one_hot_encoded1= one_hot_encoded1.astype(float)
        df = df.join(one_hot_encoded1)

        one_hot_encoded2 = pd.get_dummies(df['Study'], prefix='Study')
        one_hot_encoded2= one_hot_encoded2.astype(float)

        df = df.join(one_hot_encoded2)
        df = df.drop(columns=['Indication','Study'])
        category_columns = df.select_dtypes(include='category').columns.tolist()

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(Y)
        num_classes = len(np.unique(y_encoded))

        X=df.copy()

        seed = self.config.seed
        if self.config.test_size is not None:  
            X_train, X_test, Y_train, Y_test = train_test_split(X, y_encoded, 
                                                                test_size=self.config.test_size, 
                                                                random_state=seed)
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()
        else:
            X_train,  Y_train = X, y_encoded
            X_train = X_train.to_numpy()

        print(X_train.shape,X_test.shape)

        nan_indices = np.isnan(X_train)
        num_nans = np.sum(nan_indices)
        column_means = np.nanmean(X_train, axis=0)

        X_train[nan_indices] = np.take(column_means, np.where(nan_indices)[1])
        nan_indices = np.isnan(X_train)
        num_nans = np.sum(nan_indices)
        
        if self.config.test_size is not None:
            nan_indices = np.isnan(X_test)
            num_nans = np.sum(nan_indices)

            column_means = np.nanmean(X_test, axis=0)

            # We replace NaN values with column means
            X_test[nan_indices] = np.take(column_means, np.where(nan_indices)[1])
            nan_indices = np.isnan(X_test)
            num_nans = np.sum(nan_indices)
        
        if self.config.test_size is not None: 
            return X_train, Y_train, X_test, Y_test
        else:
            return X_train, Y_train
        
    def preprocess_anndata_starling(self, data):
        data.X = anndata.layers['exprs']
        return data