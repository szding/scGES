import numpy as np
import pandas as pd
from collections import Counter
import scanpy as sc
from collections import Counter
from sklearn.neighbors import NearestNeighbors

def Metric_label(true, predicted):
    """
    Returns
    -------
    dict with:
    - Conf: confusion matrix
    - MedF1 : median F1-score
    - F1 : F1-score per class
    - Acc : accuracy
    - PercUnl : percentage of unlabeled cells
    - PopSize : number of cells per cell type
    """
    
    true_lab = np.array(true).flatten()
    pred_lab = np.array(predicted).flatten()
    

    unassigned_labels = {'unassigned', 'Unassigned', 'Unknown', 'rand', 'Node', 'None','ambiguous', 'unknown'}

    unique_all = np.unique(np.concatenate([true_lab, pred_lab]))

    conf = pd.crosstab(true_lab, pred_lab)

    pop_size = conf.sum(axis=1)

    conf_F1 = conf.drop(columns=unassigned_labels, errors='ignore')
    conf_F1 = conf_F1[~conf_F1.index.isin(unassigned_labels)]
    
    F1 = {}
    sum_acc = 0
    
    for true_label in conf_F1.index:
        if pop_size[true_label] == 0:
            F1[true_label] = np.nan
            continue

        pred_match = true_label if true_label in conf_F1.columns else None
        
        if pred_match is not None:
            tp = conf_F1.loc[true_label, pred_match]
            fp = conf_F1[pred_match].sum() - tp
            fn = conf_F1.loc[true_label].sum() - tp
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if prec == 0 or rec == 0:
                F1[true_label] = 0
            else:
                F1[true_label] = 2 * (prec * rec) / (prec + rec)
            
            sum_acc += tp
        else:
            F1[true_label] = 0
    
    med_F1 = np.nanmedian(list(F1.values()))
    
    num_unlab = sum(pred_lab == label for label in unassigned_labels)
    per_unlab = num_unlab / len(pred_lab)

    acc = sum_acc / conf_F1.sum().sum() if conf_F1.sum().sum() > 0 else 0
    
    return {
        'Conf': conf,
        'MedF1': med_F1,
        'F1': F1,
        'Acc': acc,
        'PercUnl': per_unlab,
        'PopSize': pop_size
    }


def Label_transfer(RL_query, RL_ref=None, n_neighbors=20, index = None, label_key = 'predictions', emb_key = 'HVG', ref_label_key = 'celltype'):
    
    if RL_ref is None:
        all_indices = RL_query.obs.index
        remove_indices = all_indices[index]
        keep_mask = ~all_indices.isin(remove_indices)

        RL_ref_ = RL_query[keep_mask, :].copy()  
        RL_query = RL_query[~keep_mask, :].copy()
        
        ref_adata = sc.AnnData(RL_ref_.obsm[emb_key])
        ref_adata.obs[ref_label_key] = RL_ref_.obs[label_key].values
        query_adata = sc.AnnData(RL_query.obsm[emb_key])
    else:
        ref_adata = sc.AnnData(RL_ref.obsm[emb_key])
        ref_adata.obs[ref_label_key] = RL_ref.obs[ref_label_key].values
        query_adata = sc.AnnData(RL_query.obsm[emb_key])

    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(ref_adata.X)
    distances, indices = knn.kneighbors(query_adata.X)
    

    all_celltypes = np.unique(ref_adata.obs[ref_label_key])
    
    label = []
    for neighbor_indices in indices:
        neighbor_types = ref_adata.obs[ref_label_key].values[neighbor_indices]
        type_counts = Counter(neighbor_types)
        most_common_type, count = type_counts.most_common(1)[0]  
        
        if RL_ref is None:
            label.append(most_common_type)
        else:
            total_neighbors = len(neighbor_indices) 
            ratio = count / total_neighbors  
            if ratio > 0.5:
                label.append(most_common_type)
            else:
                label.append(0)
    return label