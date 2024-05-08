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
        self.dataset = self.load_h5ad()
        
        self.img_data_path = os.path.join(config.imgdata_path, 'img')
        self.mask_data_path = os.path.join(config.imgdata_path, 'masks')
            
    def load_anndata(self):    
        return anndata.read_h5ad(self.config.anndata_path)
    
    def load_starling(self):
        self.dataset.X = anndata.layers['exprs']
        return self.dataset
    
    def load_xgboost(self, preprocess=True, split=False):
        data = self.dataset.to_df()    
        if preprocess:
            # select features etc
            pass
        if split:
            train, test = self.split_tabular(data)
            return xgb.DMatrix(train), xgb.DMatrix(test)
        
        return xgb.DMatrix(data)
        
    def load_linear(self, preprocess=True, split=False):
        pass
        
    def split_tabular(self, data):
        # split into train and test according to config
        pass
    
    def transform_image(self, img):
        return np.arcsinh(img / 5.)
    
    def load_img_and_mask(self, imgID):
        image_name = self.dataset.obs.iloc[imgID]['image']
        
        image_path = os.path.join(self.img_data_path, image_name)
        mask_path = os.path.join(self.mask_data_path, image_name)
        
        image_reader = pyff.OMETIFFReader(fpath=image_path)
        mask_reader = pyff.OMETIFFReader(fpath=mask_path)
        
        image_array, _, _ = image_reader.read()
        mask_array, _, _ = mask_reader.read()
        
        return image_array, mask_array, image_name
        
    def load_img_batch(self, batchsize, split=False, transform=True):
        pass
    
    def get_data(self):
        
        
    def preprocess_anndata(self):
        data = self.load_anndata()
        
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

        seed = 10
        test_size = self.config.test_size
        X_train, X_test, Y_train, Y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=seed)

        X_train=X_train.to_numpy()
        X_test=X_test.to_numpy()

        print(X_train.shape,X_test.shape)

        nan_indices = np.isnan(X_train)
        num_nans = np.sum(nan_indices)
        column_means = np.nanmean(X_train, axis=0)

        X_train[nan_indices] = np.take(column_means, np.where(nan_indices)[1])
        nan_indices = np.isnan(X_train)
        num_nans = np.sum(nan_indices)
        
        nan_indices = np.isnan(X_test)
        num_nans = np.sum(nan_indices)

        column_means = np.nanmean(X_test, axis=0)

        # We replace NaN values with column means
        X_test[nan_indices] = np.take(column_means, np.where(nan_indices)[1])
        nan_indices = np.isnan(X_test)
        num_nans = np.sum(nan_indices)