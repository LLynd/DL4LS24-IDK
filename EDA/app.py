import streamlit as st
import anndata

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd
import numpy as np


adata = anndata.read_h5ad('train/cell_data.h5ad')


st.title('Exploratory data analysis')

df_markers = pd.DataFrame(adata.layers['exprs'], columns=adata.var.marker.unique())

ct_code, ct_label = pd.factorize(np.array(adata.obs.cell_labels))
df_markers['cell_code'] = ct_code
df_markers['cell_label'] = np.array(adata.obs.cell_labels)


composition_plot = px.bar(x=df_markers.groupby('cell_label').count().index,
                          y=df_markers.groupby('cell_label').count()['cell_code'],
                          title='Celltype composition',
                          color=ct_label,
                          text_auto='.2s'
)
composition_plot.update_layout(autosize=False,
                               width=800,
                               height=600,
                               showlegend=False)
composition_plot.update_traces(textposition='outside')
st.plotly_chart(composition_plot)


corr_matrix = df_markers.drop(columns=['cell_label']).corr()
corr_plot = px.imshow(corr_matrix, text_auto=True, color_continuous_scale=px.colors.sequential.Sunset)


corr_plot.update_layout(
    autosize=False,
    width=1000,
    height=1000,
)

# najbardziej skorelowane
most_correlated = list(corr_matrix[abs(corr_matrix['cell_code']) > 0.4].index[:-1])
st.plotly_chart(corr_plot)


celltype_opt = st.selectbox(
    'Choose cell type',
    ct_label,
)

df = df_markers[df_markers.cell_label == celltype_opt]
marker_fig = px.bar(x=df.columns[:-2],
                    y=df.drop(columns=['cell_code', 'cell_label']).mean(),
                    title=f'Marker expression in {celltype_opt} cells',
                    color=df.columns[:-2],
                    )
marker_fig.update_layout(
    autosize=False,
    width=800,
    height=800,
    showlegend=False,
)

st.plotly_chart(marker_fig)

marker_opt = st.selectbox(
    'Choose cell type',
    df_markers.columns[:-2],
)

celltypes_fig = px.bar(# x=ct_label,
                       x=df_markers.groupby('cell_label').mean().index,
                       y=df_markers.groupby('cell_label').mean()[marker_opt],
                       title=f'{marker_opt} expression in cell types',
                       color=ct_label,
                       )
print(df_markers.groupby('cell_label').mean()[marker_opt])

celltypes_fig.update_layout(
    autosize=False,
    width=800,
    height=800,
    showlegend=False,
)


st.plotly_chart(celltypes_fig)
