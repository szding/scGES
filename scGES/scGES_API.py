import os
import torch.optim as optim
from tqdm import tqdm
from time import time
import psutil
import scanpy as sc
import pandas as pd
from anndata import AnnData
from copy import deepcopy
from torch.utils.data import DataLoader
import numpy as np
import gc
from .preprocess import preprocess_data
from .MNN import generator_from_index
from .untils import AnnDataDataset, LabeledKnnTripletDataset, set_seed
from .layers import *



class scGES_API:
    def __init__(self, adata, tech_key='study', query_name=None, celltype_key='cell_type', knn_num = 20,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 recon_loss: Optional[str] = 'nb',use_bn: bool = False, use_ln: bool = True, dr_rate: float = 0,
                 batch_size=64, hidden_dim=[128, 32], hvg_nodes=32, gradient_clipping=5,
                 atlas_params=None, map_params=None, atlas_lvg_params=None, map_lvg_params=None,
                 train_params={'num_epochs': 8, 'lr': 5e-4}, model_name='pretrain_model', seed=202409,
                 pretrain_params=None, mode_save_dir='model_weights', results_save_dir = None):

        self.seed = seed
        self.batch_size = batch_size
        self.device = device
        self.celltype_key = celltype_key
        self.knn_num = knn_num
        self.recon_loss = recon_loss
        self.dr_rate = dr_rate
        self.use_bn = use_bn
        self.use_ln = use_ln

        self.query_name = query_name
        self.tech_key = tech_key
        self.gradient_clipping = gradient_clipping
        self.mode_save_dir = mode_save_dir
        self.results_save_dir = results_save_dir
        os.makedirs(self.mode_save_dir, exist_ok=True)

        self.reconstruction_criterion = nn.MSELoss()
        self.triplet_criterion = nn.TripletMarginLoss(margin=1.0, p=2)

        self.classification_criterion = nn.CrossEntropyLoss()
        classification_loss = nn.CrossEntropyLoss(reduction='none')
        self.masked_loss_function = semi_supervised_loss(classification_loss)

        print('\033[1;33m... data preprocess ...\033[0m')
        self.atlas, self.query, self.adatamap, self.batch_map, self.celltype_map = preprocess_data(adata, tech_key=tech_key,
                                                                                                   celltype_key=celltype_key,
                                                                                                   query_name=query_name)

        a = self.atlas.obs[tech_key + '_label'].value_counts()
        self.cell_max = int(a.index[0])
        self.index_to_celltype = {v: k for k, v in self.celltype_map.items()}

        HVG_dim = sum(self.atlas.var['Variance Type'] == 'HVG')
        self.LVG_dim = sum(self.atlas.var['Variance Type'] == 'LVG')

        cls_num = len(np.unique(self.adatamap.obs[celltype_key][self.adatamap.obs[celltype_key] != np.array(-1)]))  #
        self.atlas_batch_num = len(set(self.atlas.obs[tech_key]))

        if query_name is not None:
            self.query_batch_num = len(set(self.query.obs[tech_key]))
        else:
            self.query_batch_num = 0


        self.pretrain_params = pretrain_params if pretrain_params is not None else [HVG_dim, hidden_dim[0],
                                                                                    hidden_dim[1],
                                                                                    cls_num, 0, 0, 0]
        self.atlas_params = atlas_params if atlas_params is not None else [HVG_dim, hidden_dim[0], hidden_dim[1],
                                                                       cls_num,
                                                                       self.atlas_batch_num, 0, 0]
        self.atlas_lvg_params = atlas_lvg_params if atlas_lvg_params is not None else [self.LVG_dim, hidden_dim[0],
                                                                                       hidden_dim[1],
                                                                                       cls_num,
                                                                                       self.atlas_batch_num, 0,
                                                                                       hvg_nodes]


        if query_name is not None:
            self.map_params = map_params if map_params is not None else [HVG_dim, hidden_dim[0], hidden_dim[1],
                                                                         cls_num, self.atlas_batch_num,
                                                                         self.query_batch_num, 0]
            self.map_lvg_params = map_lvg_params if map_lvg_params is not None else [self.LVG_dim, hidden_dim[0],
                                                                                     hidden_dim[1],
                                                                                     cls_num,
                                                                                     self.atlas_batch_num,
                                                                                     self.query_batch_num,
                                                                                     hvg_nodes]

        self.pretrain_with_HVG(self.atlas, train_params, model_name)


        low_dimensional_representations, _, _ = self.predict(self.pretrain_params, model_name, self.atlas, 'HVG')
        self.atlas.obsm['pretransfer embedding'] = low_dimensional_representations

        if query_name is not None:
            low_dimensional_representations, _, _ = self.predict(self.pretrain_params, model_name, self.adatamap, 'HVG')
            self.adatamap.obsm['pretransfer embedding'] = low_dimensional_representations

    def train_model_with_HVG(self, model, data_loader, train_params, model_name):

        num_epochs = train_params['num_epochs']
        lr = train_params['lr']
        weight = train_params['weight']

        if model_name.split('_')[0] == 'map':
            triplet_model = TripletNetworkTNN(model)
            if list(triplet_model.parameters()) == list(model.parameters()):
                print("Fixed weight completion!")
            else:
                print("The weight has not been fixed yet!")
            optimizer = optim.Adam(triplet_model.parameters(), lr=lr)

        else:
            triplet_model = TripletNetworkTNN(model)  # model
            optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in tqdm(range(0, num_epochs)):
            for input_X, onehot,(size_factors, labels, _, input_XC) in data_loader:

                X = []
                for i in range(len(input_X)):
                    X.append(input_X[i].to(self.device))
                input_X = tuple(X)
                L = []
                for i in range(len(onehot)):
                    L.append(onehot[i].to(self.device))
                onehot = tuple(L)

                size_factors = size_factors.to(self.device)
                labels = labels.to(self.device)
                input_XC = input_XC.to(self.device)

                model.train()
                recon_loss, z0, z1, z2, cls, dec_mean = triplet_model(input_X, input_XC, size_factors, onehot,
                                                                      minus_num = self.LVG_dim)
                reconstruction_loss = recon_loss
                triplet_loss = self.triplet_criterion(z0, z1, z2)

                if model_name.split('_')[0] == 'atlas':
                    classification_loss = self.masked_loss_function(labels, cls)  # true, predict

                    total_loss = reconstruction_loss * weight[0] + \
                                 classification_loss * weight[1] + \
                                 triplet_loss * weight[2]
                else:
                    total_loss = reconstruction_loss * weight[0] + triplet_loss * weight[1]

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pretrain_model.parameters(), self.gradient_clipping)
                optimizer.step()

            if model_name.split('_')[0] == 'atlas':
                print(f'{model_name} - Epoch [{epoch + 1}/{num_epochs}], '
                      f'Tatol Loss: {total_loss.item():.4f}')
            else:
                print(f'{model_name} - Epoch [{epoch + 1}/{num_epochs}], '
                      f'Tatol Loss: {total_loss.item():.4f}')


        return model

    def train_model_with_LVG(self, model, data_loader, train_params, model_name):

        num_epochs = train_params['num_epochs']
        lr = train_params['lr']
        weight = train_params['weight']
        each_epochs = train_params['each_epochs']
        optimizer = optim.Adam(model.parameters(), lr=lr)


        for epoch in tqdm(range(0, num_epochs)):
            if each_epochs is not None:

                for i in range(each_epochs[1]):
                    for inputs, inputs_c, size_factor, batch_labels, hvg_emb, targets in data_loader:

                        inputs = inputs.to(self.device)
                        inputs_c = inputs_c.to(self.device)
                        size_factor = size_factor.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        hvg_emb = hvg_emb.to(self.device)
                        targets = targets.to(self.device)

                        model.train()

                        recon_loss, z, cls, dec_mean = model(inputs, inputs_c, size_factor, batch_labels, hvg_emb, self.LVG_dim)
                        classification_loss = self.masked_loss_function(targets, cls)
                        optimizer.zero_grad()
                        classification_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.pretrain_model.parameters(), self.gradient_clipping)
                        optimizer.step()

                for i in range(each_epochs[0]):
                    for inputs, inputs_c, size_factor, batch_labels, hvg_emb, targets in data_loader:

                        inputs = inputs.to(self.device)
                        inputs_c = inputs_c.to(self.device)
                        size_factor = size_factor.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        hvg_emb = hvg_emb.to(self.device)

                        model.train()

                        recon_loss, z, cls, dec_mean = model(inputs, inputs_c, size_factor, batch_labels, hvg_emb, self.LVG_dim)
                        # reconstruction_loss = self.reconstruction_criterion(outputs, inputs)
                        reconstruction_loss = recon_loss
                        optimizer.zero_grad()
                        reconstruction_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.pretrain_model.parameters(), self.gradient_clipping)
                        optimizer.step()

                print(f'{model_name} - Epoch [{epoch + 1}/{num_epochs}], '
                      f'Reconstruction Loss: {reconstruction_loss.item():.4f}, '
                      f'Classification Loss: {classification_loss.item():.4f}')

            else:
                for inputs, inputs_c, size_factor, batch_labels, hvg_emb, targets in data_loader:
                    inputs = inputs.to(self.device)
                    inputs_c = inputs_c.to(self.device)
                    size_factor = size_factor.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    hvg_emb = hvg_emb.to(self.device)
                    targets = targets.to(self.device)
                    model.train()
                    recon_loss, z, cls, dec_mean = model(inputs, inputs_c, size_factor, batch_labels, hvg_emb, self.LVG_dim)
                    reconstruction_loss = recon_loss
                    classification_loss = self.masked_loss_function(targets, cls)
                    total_loss = reconstruction_loss * weight[0] + classification_loss * weight[1]

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.pretrain_model.parameters(), self.gradient_clipping)
                    optimizer.step()

                print(f'{model_name} - Epoch [{epoch + 1}/{num_epochs}], '
                      f'Tatol Loss: {total_loss.item():.4f}, '
                      f'Reconstruction Loss: {reconstruction_loss.item():.4f}, '
                      f'Classification Loss: {classification_loss.item():.4f}')

        return model



    def train_atlas(self, adata = None,train_params = {'num_epochs': 8, 'lr': 5e-4, 'weight':[1,1,1]},
                    model_name = 'atlas_model_', model_type = 'HVG'):

        if model_type == 'HVG':

            if adata == None:
                adata = self.atlas[:, self.atlas.var['Variance Type'] == 'HVG']
            else:
                adata = adata[:, adata.var['Variance Type'] == 'HVG']

            print('\033[1;33m...  ATLAS HVG train ...\033[0m')

            model_name = model_name + model_type   # 保存权重的时候使用

            X_train, X_all, triplet_list, tech_list, tech_indices = \
                generator_from_index(adata, tech_name=self.tech_key,celltype_name=self.celltype_key,
                                     knn_num=self.knn_num,
                                     mask_batch=None, mapping=False)

            set_seed(self.seed)
            data_loader = LabeledKnnTripletDataset(X_train, X_all, dictionary=triplet_list, batch_list=tech_list,
                                                   batch_key=self.tech_key+'_label', celltype_key=self.celltype_key,
                                                   batch_indices=tech_indices, batch_size=self.batch_size)

            del X_train, X_all, triplet_list, tech_list, tech_indices 
            gc.collect()

            self.atlasHVG_model = scGES(self.atlas_params, dispersion = "gene-batch",
                                        recon_loss = self.recon_loss, dr_rate=self.dr_rate,
                                        use_bn=self.use_bn, use_ln= self.use_ln).to(self.device)

            # self.atlasHVG_model = self.transfer_weights(model_type = model_type, model_name = ['pretrain_model', model_name])

            self.atlasHVG_model = self.train_model_with_HVG(self.atlasHVG_model, data_loader,
                                                            train_params, model_name)


            self._save_weights(self.atlasHVG_model, model_name)

            low_dimensional_representations, cls, dec = self.predict(self.atlas_params, 'atlas_model_HVG', adata,
                                                                     'HVG', map_batch_label = self.cell_max)
            self.atlas.obsm['atlas HVG embedding'] = low_dimensional_representations

            labels_ = torch.argmax(torch.tensor(cls, dtype=torch.float32), dim=1)
            celltype_names = [self.index_to_celltype[idx.item()] for idx in labels_]
            self.atlas.obs['atlas HVG predicted'] = celltype_names
            self.atlas.obsm['atlas HVG denoised'] = dec
        
        elif model_type == 'LVG':

            if adata == None:
                adata = self.atlas[:, self.atlas.var['Variance Type'] == 'LVG']
            else:
                adata = adata[:, adata.var['Variance Type'] == 'LVG']

            gc.collect()

            model_name = model_name + model_type
            print('\033[1;33m...  atlas LVG train ...\033[0m')

            set_seed(self.seed)
            dataset = AnnDataDataset(adata, model_type = model_type, emb_key ='atlas HVG embedding',
                                     batch_key = self.tech_key + '_label',
                                     labels_key = self.celltype_key)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            del dataset  
            gc.collect()  

            # build model
            self.atlasLVG_model = scGES(self.atlas_lvg_params, dispersion = "gene-batch",
                                        recon_loss = self.recon_loss, dr_rate=self.dr_rate,
                                        use_bn=self.use_bn, use_ln= self.use_ln).to(self.device)

            self.atlasLVG_model = self.train_model_with_LVG(self.atlasLVG_model, data_loader, train_params, model_name)


            self._save_weights(self.atlasLVG_model, model_name)

            low_dimensional_representations, cls, dec = self.predict(self.atlas_lvg_params, 'atlas_model_LVG', adata,
                                                                     'LVG', 'atlas HVG embedding', map_batch_label = self.cell_max)


            self.atlas.obsm['atlas LVG embedding'] = low_dimensional_representations
    

            labels_ = torch.argmax(torch.tensor(cls, dtype=torch.float32), dim=1)
            celltype_names = [self.index_to_celltype[idx.item()] for idx in labels_]
            self.atlas.obs['atlas LVG predicted'] = celltype_names


            self.atlas.obsm['atlas LVG denoised'] = dec  # LVG
            denoised_ = np.concatenate((self.atlas.obsm['atlas HVG denoised'], dec), axis=1)
            HVG_gene = self.atlas.var_names[self.atlas.var['Variance Type'] == 'HVG'].to_list()
            LVG_gene = self.atlas.var_names[self.atlas.var['Variance Type'] == 'LVG'].to_list()
            denoised = pd.DataFrame(denoised_, columns=HVG_gene+LVG_gene, index = self.atlas.obs_names.to_list())
            self.atlas.layers["atlas ALL denoised"] = denoised[list(self.atlas.var_names)].to_numpy()

        

    def train_map(self, adata = None, train_params = {'num_epochs': 8, 'lr': 5e-4, 'weight':[1,1]},
                  model_name = 'map_model_', model_type = 'HVG'):

        if model_type == 'HVG':
            if adata == None:
                adata = self.adatamap[:, self.adatamap.var['Variance Type'] == 'HVG']
            else:
                adata = adata[:, adata.var['Variance Type'] == 'HVG']
            print('\033[1;33m...  MAP HVG train ...\033[0m')
            model_name = model_name + model_type


            # 2、
            X_train, X_all, triplet_list, tech_list, tech_indices = \
                generator_from_index(adata, tech_name=self.tech_key,celltype_name=self.celltype_key,
                                     knn_num=self.knn_num,
                                     mask_batch=self.query_name, mapping=True)

            set_seed(self.seed)
            data_loader = LabeledKnnTripletDataset(X_train, X_all, dictionary=triplet_list, batch_list=tech_list,
                                                   batch_key=self.tech_key + '_label', celltype_key=self.celltype_key,
                                                   batch_indices=tech_indices,batch_size=self.batch_size)

            del X_train, X_all, triplet_list, tech_list, tech_indices, self.adatamap.obsm['pretransfer embedding']
            gc.collect()

            # 3、build model
            self.mapHVG_model = scGES(self.map_params,
                                      recon_loss = self.recon_loss, dr_rate=self.dr_rate,
                                      use_bn=self.use_bn, use_ln= self.use_ln).to(self.device)

            self.mapHVG_model = self.transfer_weights(model_type = model_type, model_name=['atlas_model_HVG', model_name])

            self.mapHVG_model = self.train_model_with_HVG(self.mapHVG_model, data_loader, train_params, model_name)



            self._save_weights(self.mapHVG_model, model_name)

            low_dimensional_representations, labels, dec = self.predict(self.map_params, 'map_model_HVG', adata,
                                                                        'HVG', atlas_params = self.atlas_params,
                                                                        atlas_name='atlas_model_HVG',
                                                                        map_batch_label = self.cell_max)

            del adata
            gc.collect()

            self.adatamap.obsm['map HVG embedding'] = low_dimensional_representations
            self.adatamap.obsm['map HVG denoised'] = dec

            labels_ = torch.argmax(torch.tensor(labels, dtype=torch.float32), dim=1)
            celltype_names = [self.index_to_celltype[idx.item()] for idx in labels_]

            self.adatamap.obs['map_labels'] = labels_
            self.adatamap.obs['map_HVG_labels'] = celltype_names


        elif model_type == 'LVG':

            if adata == None:
                adata = self.adatamap[:, self.adatamap.var['Variance Type'] == 'LVG']
            else:
                adata = adata[:, adata.var['Variance Type'] == 'LVG']

            model_name = model_name + model_type

            print('\033[1;33m... map LVG train ...\033[0m')



            set_seed(self.seed)
            dataset = AnnDataDataset(adata, model_type=model_type, emb_key='map HVG embedding',
                                     batch_key = self.tech_key + '_label',
                                     labels_key = 'map_labels')
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            del dataset
            gc.collect()

            self.mapLVG_model = scGES(self.map_lvg_params,
                                      recon_loss = self.recon_loss, dr_rate=self.dr_rate,
                                      use_bn=self.use_bn, use_ln= self.use_ln).to(self.device)


            self.mapLVG_model = self.transfer_weights(model_type = model_type, model_name=['atlas_model_LVG', model_name])


            self.train_model_with_LVG(self.mapLVG_model, data_loader, train_params, model_name)


            self._save_weights(self.mapLVG_model, model_name)

            low_dimensional_representations, labels, dec = self.predict(self.map_lvg_params, 'map_model_LVG', adata,
                                                                        'LVG', 'map HVG embedding',
                                                                        atlas_params = self.atlas_lvg_params,
                                                                        atlas_name='atlas_model_LVG',
                                                                        map_batch_label = self.cell_max)
            del adata
            gc.collect()

            self.adatamap.obsm['map LVG embedding'] = low_dimensional_representations
            self.adatamap.obsm['map LVG denoised'] = dec

            HVG_gene = self.adatamap.var_names[self.adatamap.var['Variance Type'] == 'HVG'].to_list()
            LVG_gene = self.adatamap.var_names[self.adatamap.var['Variance Type'] == 'LVG'].to_list()
            denoised_ = np.concatenate((self.adatamap.obsm['map HVG denoised'], dec), axis=1)
            import pandas as pd
            denoised = pd.DataFrame(denoised_, columns=HVG_gene + LVG_gene, index=self.adatamap.obs_names.to_list())
            self.adatamap.layers["map ALL denoised"] = denoised[list(self.adatamap.var_names)].to_numpy()


            labels_ = torch.argmax(torch.tensor(labels, dtype=torch.float32), dim=1)
            celltype_names = [self.index_to_celltype[idx.item()] for idx in labels_]
            self.adatamap.obs['map_LVG_labels'] = celltype_names



    def transfer_weights(self, atlas_model = None, model_type = None, model_name = None):
        with torch.no_grad():
            if model_type == 'HVG':
                if model_name[0] == 'pretrain_model':
                    map_model = self._transfer_model_weights(model_map = self.atlasHVG_model,
                                                             atlas_params = self.pretrain_params,
                                                             model_name = model_name)
                else:
                    map_model = self._transfer_model_weights(atlas_model = atlas_model,
                                                             model_map = self.mapHVG_model,
                                                             freeze=True,
                                                             atlas_params = self.atlas_params,
                                                             model_name = model_name)
            elif model_type == 'LVG':
                map_model = self._transfer_model_weights(atlas_model = atlas_model,
                                                         model_map = self.mapLVG_model,
                                                         freeze=True,
                                                         atlas_params = self.atlas_lvg_params,
                                                         model_name = model_name)
        return map_model


    def _transfer_model_weights(self, atlas_model = None, model_map = None, freeze = False, freeze_expression = True,
                                atlas_params = None, model_name = None):


        if atlas_model == None:
            if atlas_params[4] != 0 and atlas_params[5] == 0 :
                dispersion = "gene-batch"
            else:
                dispersion = "gene"
            atlas_model = scGES(atlas_params, dispersion = dispersion,
                                recon_loss = self.recon_loss, dr_rate=self.dr_rate,
                                use_bn=self.use_bn, use_ln= self.use_ln).to(self.device)
            atlas_model, atlas_dict = self._load_weights(atlas_model, model_name[0])  # atlas weight

        model_map.to(next(iter(atlas_dict.values())).device)
        device = next(model_map.parameters()).device
        map_dict = model_map.state_dict()


        for key, atlas_w in atlas_dict.items():
            if key == 'theta':
                continue
            else:
                map_w = map_dict[key]
                if map_w.size() == atlas_w.size():
                    continue

                else:
                    atlas_w = atlas_w.to(device)
                    # only one dim diff
                    new_shape = map_w.shape
                    n_dims = len(new_shape)

                    sel = [slice(None)] * n_dims
                    for i in range(n_dims):
                        dim_diff = new_shape[i] - atlas_w.shape[i]
                        axs = i
                        sel[i] = slice(-dim_diff, None)
                        if dim_diff > 0:
                            break
                    fixed_ten = torch.cat([atlas_w, map_w[tuple(sel)]], dim=axs)
                    atlas_dict[key] = fixed_ten

        for key, map_ww in map_dict.items():
            if key == 'theta':
                atlas_dict[key] = map_ww
            else:
                if key not in atlas_dict:
                    atlas_dict[key] = map_ww

        model_map.load_state_dict(atlas_dict)

        if freeze:
            model_map.freeze = True
            for name, p in model_map.named_parameters():
                p.requires_grad = False
                if 'theta' in name:  # theta参数训练
                    p.requires_grad = True
                if freeze_expression:  # 条件weights训练
                    if 'cond_L.weight' in name:
                        p.requires_grad = True
                else:
                    if "L0" in name or "N0" in name:
                        p.requires_grad = True
        # list(filter(lambda p: p.requires_grad, model_map.parameters()))
        return model_map


    def _get_model_filepath(self, model_name):
        return os.path.join(self.mode_save_dir, f"{model_name}.pth")

    def _load_weights(self, model, model_name):

        model_path = self._get_model_filepath(model_name)

        if os.path.exists(model_path):
            model_state_dict = torch.load(model_path)
            model.load_state_dict(model_state_dict)
            print(f"Loaded pre-trained weights for {model_name} from {model_path}.")
            return model, model_state_dict
        else:
            print(f"No pre-trained weights found for {model_name}. Training will be required.")

    def _save_weights(self, model, model_name):
        torch.save(model.state_dict(), self._get_model_filepath(model_name))

        print("Model weights have been saved.")

    def pretrain_with_HVG(self, adata, train_params={'num_epochs': 8, 'lr': 5e-4}, model_name='pretrain_model'):

        set_seed(self.seed)
        dataset = AnnDataDataset(adata)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        del dataset
        gc.collect()

        self.pretrain_model = scGES(self.pretrain_params,
                                    recon_loss = self.recon_loss, dr_rate=self.dr_rate,
                                    use_bn=self.use_bn, use_ln= self.use_ln).to(self.device)
        num_epochs = train_params['num_epochs']
        lr = train_params['lr']
        optimizer = optim.Adam(self.pretrain_model.parameters(), lr=lr)

        for epoch in tqdm(range(0, num_epochs)):
            for inputs, inputs_c, size_factor, _ , _ , _ in data_loader:
                inputs = inputs.to(self.device)
                inputs_c = inputs_c.to(self.device)
                size_factor = size_factor.to(self.device)

                self.pretrain_model.train()

                recon_loss, _, _, _= self.pretrain_model(x=inputs, x_c = inputs_c, sizefactor=size_factor)

                loss = recon_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pretrain_model.parameters(), self.gradient_clipping)
                optimizer.step()

            # print(f'Pretraining - Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        self._save_weights(self.pretrain_model, model_name)

    def predict(self, model_params, model_name, adata, model_type='HVG', emb_key = None,
                atlas_params = None, atlas_name = None,map_batch_label = None):

        if model_name.split('_')[0] == 'atlas':
            dispersion = "gene-batch"
        else:
            dispersion = "gene"

        set_seed(self.seed)

        # map model
        model_ = scGES(model_params, dispersion = dispersion,
                       recon_loss = self.recon_loss, dr_rate=self.dr_rate,
                       use_bn=self.use_bn, use_ln= self.use_ln).to(self.device)
        model, weight_dict = self._load_weights(model_, model_name)

        # atlas model
        if dispersion == "gene-batch":  # 合并到同一个batch中
            dim_1, dim_2 = weight_dict['theta'].shape
            weight_new = weight_dict['theta'][:,map_batch_label]
            weight_new = weight_new.expand(dim_2, -1).T
            weight_dict['theta'] = weight_new
            model.load_state_dict(weight_dict)

        # map model
        if atlas_params is not None:
            model_atlas = scGES(atlas_params, dispersion="gene-batch",
                                recon_loss = self.recon_loss, dr_rate=self.dr_rate,
                                use_bn=self.use_bn, use_ln= self.use_ln).to(self.device)
            model_atlas, weight_dict_atlas = self._load_weights(model_atlas, atlas_name)
            weight_new = weight_dict_atlas['theta'][:, map_batch_label]
            weight_dict['theta'] = weight_new
            model.load_state_dict(weight_dict)

        dataset = AnnDataDataset(adata, model_type=model_type, emb_key=emb_key,
                                 batch_key = self.tech_key + '_label',
                                 labels_key=self.celltype_key,
                                 map_batch_label = map_batch_label)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        del dataset, adata
        gc.collect()

        encoded_results = []
        classified_results = []
        decoded_results = []
        with torch.no_grad():  # Disable gradient computation for inference
            for inputs, inputs_c, size_factor, batch_labels, emb, y in data_loader:
                if model_params[4] == 0:  # pretrain-HVG
                    inputs = inputs.to(self.device)
                    inputs_c = inputs_c.to(self.device)
                    size_factor = size_factor.to(self.device)
                    recon_loss, z1, cls, dec = model(x=inputs, x_c = inputs_c, sizefactor=size_factor)
                elif model_params[4] != 0 and model_params[6] != 0: # atlas-map-LVG
                    inputs = inputs.to(self.device)
                    inputs_c = inputs_c.to(self.device)
                    size_factor = size_factor.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    emb = emb.to(self.device)
                    recon_loss, z1, cls, dec = model(x=inputs, x_c = inputs_c, sizefactor=size_factor,
                                                     batch=batch_labels, hvg_emb=emb)
                else: # atlas-map-HVG
                    inputs = inputs.to(self.device)
                    inputs_c = inputs_c.to(self.device)
                    size_factor = size_factor.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    recon_loss, z1, cls, dec = model(x=inputs, x_c = inputs_c, sizefactor=size_factor,
                                                     batch=batch_labels)

                encoded_results.append(z1.cpu().numpy())
                classified_results.append(cls.cpu().numpy())
                decoded_results.append(dec.cpu().numpy())

        return (np.concatenate(encoded_results, axis=0),
                np.concatenate(classified_results, axis=0),
                np.concatenate(decoded_results, axis=0))


