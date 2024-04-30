import os
import anndata
import pyometiff as pyff
import numpy as np
import xgboost as xgb


class DataLoader():
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