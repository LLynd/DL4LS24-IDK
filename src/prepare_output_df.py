import torch
import numpy as np
import pandas as pd


def new_predictions(model,sample_ids,df_X,encoder):
    output_df = pd.DataFrame()
    X_np = df_X.to_numpy()
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions_probabilities = torch.softmax(outputs,axis=1)
        predictions_probabilities = predictions_probabilities.detach().numpy() 
        predictions = np.argmax(predictions_probabilities, axis=1)

    output_df['sample_id'] = sample_ids
    output_df['predictions'] = predictions
    output_df['predictions'] = encoder.inverse_transform(output_df['predictions'])
    output_df['predictions'] = predictions_probabilities

    return output_df