from torch.utils.data import Dataset, DataLoader
from scipy.sparse import issparse
import logging
import random
import numpy as np
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix, hstack

logger = logging.getLogger(__name__)

def variable_exists(var_name):
    try:
        eval(var_name)
        return True
    except NameError:
        return False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AnnDataDataset(Dataset):
    def __init__(self, adata, model_type = 'HVG', emb_key = None, batch_key = None, labels_key = None, map_batch_label = None):

        if 'size_factors' in adata.obs.keys().tolist():
            self.size_factors = torch.tensor(adata.obs['size_factors'], dtype=torch.float32)
        else:
            print("Lack of size_factors")

        if model_type == 'HVG':
            adata_ = adata[:,adata.var['Variance Type']=='HVG'].copy()
        if model_type == 'LVG':
            adata_ = adata[:,adata.var['Variance Type']=='LVG'].copy()

        if issparse(adata_.X):
            A = adata_.X.toarray()
            B = adata_.layers['counts'].toarray()
        else:
            A = adata_.X
            B = adata_.layers['counts']
        self.data = torch.tensor(A, dtype=torch.float32)

        self.data_c = torch.tensor(B, dtype=torch.float32)

        if batch_key is not None:
            if map_batch_label is not None:
                self.batch_labels = torch.tensor([map_batch_label] * adata_.shape[0], dtype=torch.long)
            else:
                self.batch_labels = torch.tensor(adata_.obs[batch_key], dtype=torch.long)
        else:
            self.batch_labels = None

        if labels_key is not None:
            self.celltype_label = torch.tensor(adata_.obs[labels_key], dtype=torch.long)
        else:
            self.celltype_label = None

        # hvg emb
        if emb_key in adata_.obsm:
            self.emb = torch.tensor(adata_.obsm[emb_key], dtype=torch.float32)
        else:
            self.emb = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data_item = self.data[idx]
        datac_item = self.data_c[idx]
        size_factors_item = self.size_factors[idx]
        batch_label_item = self.batch_labels[idx] if self.batch_labels is not None else torch.tensor([], dtype=torch.float32)
        emb_item = self.emb[idx] if self.emb is not None else torch.tensor([], dtype=torch.float32)
        celltype_label_item = self.celltype_label[idx] if self.celltype_label is not None else torch.tensor([], dtype=torch.long)

        return data_item, datac_item, size_factors_item, batch_label_item, emb_item, celltype_label_item



class LabeledKnnTripletDataset(Dataset):
    def __init__(self, adata_train, adata_all, dictionary, batch_list, batch_indices,
                 mapping=False, map_part=False, atlas_part=False,
                 train_1=False, train_2=False, supervised=True,
                 batch_key = 'study_label', celltype_key = 'cell_type',
                 reconstruction=True, use_celltype=True, batch_size=32):

        C = adata_train.layers['counts']
        if issparse(adata_train.X):
            A = adata_train.X.toarray()
            B = adata_train.layers['counts'].toarray()
        else:
            A = adata_train.X
            B = adata_train.layers['counts']
        self.X_train = torch.tensor(A, dtype=torch.float32)  # scale

        self.X_train_C = torch.tensor(B, dtype=torch.float32) # counts

        if issparse(adata_all.X):
            self.X_all = torch.tensor(adata_all.X.toarray(), dtype=torch.float32)
        else:
            self.X_all = torch.tensor(adata_all.X, dtype=torch.float32)

        size_factors = np.ravel(C.sum(1))
        self.size_factors = torch.tensor(size_factors, dtype=torch.float32)

        # one-hot
        self.batch_label_train = torch.tensor(adata_train.obs[batch_key], dtype=torch.float32)
        self.batch_label_all = torch.tensor(adata_all.obs[batch_key], dtype=torch.float32)

        # label
        if use_celltype:
            self.Y = torch.tensor(adata_train.obs[celltype_key], dtype=torch.long)
        else :
            self.Y = torch.tensor(np.array([-1] * adata_train.shape[0]), dtype=torch.long)


        self.dictionary = dictionary
        self.batch_list = batch_list
        self.batch_indices = batch_indices

        self.batch_size = batch_size
        self.num_cells = len(self.dictionary)


        self.use_celltype = use_celltype
        self.mapping = mapping
        self.supervised = supervised
        self.reconstruction = reconstruction
        self.map_part = map_part
        self.atlas_part = atlas_part
        self.train_1 = train_1
        self.train_2 = train_2


    def __len__(self):
        return int(np.ceil(self.num_cells / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.num_cells))

        triplet = []
        onehot = []
        tripletc = []
        for row_index in batch_indices:
            triplet_, onehot_, tripletc_ = self.knn_triplet_from_dictionary(row_index=row_index,
                                                                            neighbour_list=self.dictionary[row_index],
                                                                            batch=self.batch_list[row_index])
            triplet.append(triplet_)
            onehot.append(onehot_)
            tripletc.append(tripletc_)


        triplet_list = np.array(triplet)
        onehot_list = np.array(onehot)

        X_list = self.X_train[batch_indices]
        label = self.Y[batch_indices]
        size_factors_item = self.size_factors[batch_indices]

        return (torch.tensor(triplet_list[:, 0]),
                torch.tensor(triplet_list[:, 1]),
                torch.tensor(triplet_list[:, 2])), \
               (torch.tensor(onehot_list[:, 0]),
                torch.tensor(onehot_list[:, 1]),
                torch.tensor(onehot_list[:, 2])), \
               (torch.tensor(size_factors_item), torch.tensor(label), torch.tensor(X_list), torch.tensor(tripletc))

    def knn_triplet_from_dictionary(self, row_index, neighbour_list, batch):
        anchor = row_index
        positive = np.random.choice(neighbour_list)
        negative = self.batch_indices[batch][np.random.randint(len(self.batch_indices[batch]))]

        triplets_a = np.array(self.X_train[anchor])
        tripletsc_a = np.array(self.X_train_C[anchor])
        onehot_a = np.array(self.batch_label_train[anchor])

        triplets_p = np.array(self.X_all[positive])
        onehot_p = np.array(self.batch_label_all[positive])

        triplets_n = np.array(self.X_all[negative])
        onehot_n =  np.array(self.batch_label_all[negative])

        return [triplets_a, triplets_p, triplets_n],[onehot_a, onehot_p, onehot_n], tripletsc_a






