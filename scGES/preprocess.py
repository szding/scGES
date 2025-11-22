import anndata
import torch
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.WARNING)
from scipy import sparse

import scanpy as sc
import numpy as np

def remove_sparsity(adata):
    """
        If ``adata.X`` is a sparse matrix, this will convert it in to normal matrix.
        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated data matrix.
        Returns
        -------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
    """
    if sparse.issparse(adata.X):
        new_adata = sc.AnnData(X=adata.X.A, obs=adata.obs.copy(deep=True), var=adata.var.copy(deep=True))
        return new_adata

    return adata

def select_hvg(query,atlas_hvg):
    selected_index = list(set(atlas_hvg).intersection(query.var.index))
    if len(selected_index) == 2000:
        selected_result = []
        for i in query.var.index:
            if i in atlas_hvg:
                selected_result.append(True)
            else:
                selected_result.append(False)

    return selected_result

def label_code(adata, label, condition_key=None):

    unique_conditions = list(np.unique(adata.obs[condition_key]))
    labels = np.zeros(adata.shape[0])

    if not set(unique_conditions).issubset(set(label.keys())):
        missing_labels = set(unique_conditions).difference(set(label.keys()))
        print(f"Warning: Labels in adata.obs[{condition_key}] is not a subset of label-encoder!")
        print(f"The missing labels are: {missing_labels}")
        print("Therefore integer value of those labels is set to -1")
        for data_cond in unique_conditions:
            if data_cond not in label.keys():
                labels[adata.obs[condition_key] == data_cond] = -1

    for condition, label in label.items():
        labels[adata.obs[condition_key] == condition] = label

    return labels


def preprocess(adata,
               tech_key,
               n_high_var=2000,
               scran=True,
               query=False,
               atlas_hvg=None,
               LVG=True):
    if adata.layers != None and 'counts' in adata.layers:
        adata.X = adata.layers['counts'].copy()

    tech = np.asarray(list(adata.obs[tech_key]))
    tech_list = np.unique(tech)

    if scran:
        sc.pp.normalize_total(adata, inplace=True)
    else:
        adata.obs['size_factors'] = np.ones((adata.shape[0],))

    sc.pp.log1p(adata)

    if not query:
        sc.pp.highly_variable_genes(adata,
                                    inplace=True,
                                    n_top_genes=n_high_var,  # 2000
                                    batch_key=tech_key)  # tech_list
        hvg = adata.var['highly_variable'].values
    else:
        hvg = select_hvg(adata, atlas_hvg)
    adata.var['Variance Type'] = [['LVG', 'HVG'][int(x)] for x in hvg]


    if issparse(adata.X):
        adata.X = adata.X.toarray()

    if issparse(adata.layers['counts']):
        adata.layers['counts'] = adata.layers['counts'].toarray()

    C = adata.layers['counts'].copy()
    adata.obs['size_factors'] = torch.tensor(np.ravel(C.sum(1)), dtype=torch.float32)


    for tech_ in tech_list: 
        indices = [x == tech_ for x in tech]
        sub_adata = adata[indices]
        sc.pp.scale(sub_adata)
        adata[indices] = sub_adata.X


    if not LVG:
        adata = adata[:, adata.var['Variance Type'] == 'HVG']

    return adata


def preprocess_data(adata=None, tech_key='study',celltype_key = 'cell_type',query_name=None,
                    preprocess_judge=True):


        if query_name is not None:
            atlas = adata[adata.obs[tech_key] != query_name]
            query = adata[adata.obs[tech_key] == query_name]

            if preprocess_judge:
                atlas = preprocess(atlas, tech_key = tech_key)
                atlas_hvg_index = atlas.var[atlas.var['Variance Type'] == 'HVG'].index
                query = preprocess(query, tech_key = tech_key, atlas_hvg=atlas_hvg_index, query = True)
        else:
            if preprocess_judge:
                atlas = preprocess(adata, tech_key = tech_key)
                query = None


        atlas_batch = atlas.obs[tech_key].unique().tolist()
        batch_atlas = {k: v for k, v in zip(atlas_batch, range(len(atlas_batch)))}
        atlas.obs[tech_key + '_label'] = label_code(atlas, batch_atlas, condition_key=tech_key)


        atlas_celltype = atlas.obs[celltype_key].unique().tolist()
        if 'Unassigned' in atlas_celltype:
            atlas_celltype.remove('Unassigned')
        celltype_encoder = {k: v for k, v in zip(atlas_celltype, range(len(atlas_celltype)))}
        atlas.obs[celltype_key] = label_code(atlas, celltype_encoder, condition_key=celltype_key)

        if query_name is not None:
            query_batch_ = query.obs[tech_key].unique().tolist()
            query_batch = atlas_batch
            for query_ in query_batch_:
                if query_ not in atlas_batch:
                    query_batch.append(query_)
            batch_query = {k: v for k, v in zip(query_batch, range(len(query_batch)))}
            query.obs[tech_key + '_label'] = label_code(query, batch_query, condition_key=tech_key)
            query.obs[celltype_key] = -1

            atlas.obs['renewed atlas'] = 0
            query.obs['renewed atlas'] = 1
            atlas.obs['batch'] = "atlas"
            query.obs['batch'] = "query"
            adata_new = anndata.concat([atlas, query], join="outer",
                                       index_unique=None, fill_value=None)
            set(atlas.var['Variance Type'].values == query.var['Variance Type'].values)
            adata_new.var['Variance Type'] = atlas.var['Variance Type']

            return atlas, query, adata_new, batch_atlas, celltype_encoder

        else:
            return atlas, None, atlas, batch_atlas, celltype_encoder

