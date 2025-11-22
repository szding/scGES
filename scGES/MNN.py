import pandas as pd
import numpy as np
import itertools
import networkx as nx
import hnswlib 
from sklearn.preprocessing import LabelEncoder  
from scipy.sparse import issparse
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.WARNING)
le = LabelEncoder()


def generator_from_index(adata,
                         use_celltype=True,
                         tech_name='study',
                         celltype_name='cell_type',
                         mask_batch=None,
                         mapping=False,
                         k_to_m_ratio=0.75,
                         knn_num = 20,
                         label_ratio=1):
    cells = adata.obs_names

    # atlas
    if not mapping:
        if use_celltype:
            label_dict_original = create_dictionary_label(adata,
                                                          celltype_name=celltype_name,
                                                          tech_name=tech_name,
                                                          mapping=mapping,
                                                          mask_batch=mask_batch)  # 选取点对数量基本为5*batch数量
            num_label = round(label_ratio * len(label_dict_original))
            cells_for_label = np.random.choice(list(label_dict_original.keys()), num_label, replace=False)
            label_dict = {key: value for key, value in label_dict_original.items() if key in cells_for_label}
            print('label dict', len(label_dict))
        else:
            label_dict = {}

        if len(label_dict) < len(cells):

            print('atlas MNN')
            mnn_dict = create_dictionary_mnn(adata, techname=tech_name, label_cell=label_dict, each=True, knn=knn_num)
            print('mnn dict', len(mnn_dict))


            knn_dict = {}
            num_k = round(k_to_m_ratio * len(mnn_dict))
            cells_for_knn = list(set(cells) - (set(list(label_dict.keys())) | set(
                list(mnn_dict.keys()))))
            num_knn_k = min(len(cells_for_knn), 10)
            if len(cells_for_knn) > 0:
                if (len(cells_for_knn) > num_k):
                    cells_for_knn = np.random.choice(cells_for_knn, num_k, replace=False)
                adata_knn = adata[cells_for_knn]
                knn_dict = create_dictionary_knn(adata_knn,
                                                 cells_for_knn,
                                                 k=num_knn_k)
                print('knn dict', len(knn_dict))

        else:
            print("all labels are known with atlas")
            mnn_dict = {}
            knn_dict = {}
    # map
    else:
        query = adata[adata.obs[tech_name] == mask_batch]
        query_cell = query.obs_names
        if len(set(query.obs['cell_type'])) > 1:
            label_dict_original = create_dictionary_label(adata,
                                                          celltype_name=celltype_name,
                                                          mapping=mapping,
                                                          tech_name=tech_name,
                                                          mask_batch=mask_batch)  # 选取点对数量基本为5*batch数量
            num_label = round(label_ratio * len(label_dict_original))
            cells_for_label = np.random.choice(list(label_dict_original.keys()), num_label, replace=False)
            label_dict = {key: value for key, value in label_dict_original.items() if key in cells_for_label}
            print('label dict', len(label_dict))
        else:
            label_dict = {}

        if len(set(query.obs['cell_type'])) == 1 or len(label_dict) < len(query_cell):
            print("mnns and knns of mapping")
            mnn_dict = create_dictionary_mnn(adata=adata, each=False, knn=knn_num,
                                             label_cell=label_dict,
                                             techname=tech_name,
                                             mask_batch=mask_batch)
            print('mnn dict', len(mnn_dict))


            knn_dict = {}
            num_k = round(k_to_m_ratio * len(mnn_dict))
            cells_for_knn = list(set(query_cell) - (set(list(label_dict.keys())) | set(list(mnn_dict.keys()))))
            num_knn_k = min(len(cells_for_knn), 10)
            if len(cells_for_knn) > 0:
                if (len(cells_for_knn) > num_k):
                    cells_for_knn = np.random.choice(cells_for_knn, num_k, replace=False)
                adata_knn = adata[cells_for_knn]
                knn_dict = create_dictionary_knn(adata_knn,
                                                 cells_for_knn,
                                                 k=num_knn_k)
                print('knn dict', len(knn_dict))
        else:
            print("all labels are known with map")
            mnn_dict = {}
            knn_dict = {}



    final_dict_original = merge_dict(mnn_dict, label_dict)
    final_dict_original.update(knn_dict)
    final_dict = {k: v for k, v in sorted(final_dict_original.items(), key=lambda item: item[0])}

    cells_for_train = list(final_dict.keys())
    print('cells for train:', len(cells_for_train))
    adata_train = adata[cells_for_train]


    cell_as_dict = dict(zip(list(sorted(adata.obs_names)), range(0, adata.shape[0])))  # cell 的顺序

    def get_indices2(name):
        return ([cell_as_dict[x] for x in final_dict[name]])

    triplet_list = list(map(get_indices2, cells_for_train))


    tech_list = adata_train.obs[tech_name]
    tech_indices = []
    for i in tech_list.unique():
        tech_indices.append(list(np.where(tech_list == i)[0]))
    tech_as_dict = dict(zip(list(tech_list.unique()), range(0, len(tech_list.unique()))))

    tmp = map(lambda _: tech_as_dict[_], tech_list)
    tech_list = list(tmp)

    X_train = adata_train[:, adata_train.var['Variance Type'] == 'HVG']
    X_all = adata[:, adata_train.var['Variance Type'] == 'HVG']

    return X_train, X_all, triplet_list, tech_list, tech_indices



def create_dictionary_label(adata=None,
                            celltype_name=None,
                            mapping=False,
                            tech_name=None,
                            mask_batch=None,
                            k=50):
    if not mapping:
        adata = adata[adata.obs[tech_name] != mask_batch]
        tech_list = adata.obs[tech_name]
        cell_types = adata.obs[celltype_name]
        if len(set(cell_types==-1)) >1:
            cell_types = cell_types[cell_types!=-1]
            tech_list = tech_list[cell_types.index]

        types = []
        for i in tech_list.unique():
            types.append(cell_types[tech_list == i])


        labeled_dict = dict()
        for comb in list(itertools.permutations(range(len(types)), 2)):

            i = comb[0]
            j = comb[1]
            ref_types = types[i]
            new_types = types[j]
            common = set(ref_types) & set(new_types)

            for each in common:
                ref = list(ref_types[ref_types == each].index)
                new = list(new_types[new_types == each].index)
                num_k = min(int(k / 10), 5, len(new))

                for key in ref:
                    new_cells = np.random.choice(new, num_k, replace=False)
                    if key not in labeled_dict.keys():
                        labeled_dict[key] = list(new_cells)
                    else:
                        labeled_dict[key] += list(new_cells)
    else:
        atlas = adata[adata.obs[tech_name] != mask_batch]
        query = adata[adata.obs[tech_name] == mask_batch]


        atlas_cell = atlas.obs_names
        atlas_tech = atlas.obs[tech_name]
        atlas_ct = atlas.obs[celltype_name]
        tech_new = list(set(atlas.obs[tech_name].unique()))



        query_cell = query.obs_names
        query_ct = query.obs[celltype_name]

        ct_atlas = []
        cell_atlas = []
        for i in tech_new:
            ct_atlas.append(atlas_ct[atlas_tech == i])
            cell_atlas.append(atlas_cell[atlas_tech == i])


        labeled_dict = dict()
        for i in range(len(tech_new)):

            ref_types = ct_atlas[i]
            cell_atlas_ = cell_atlas[i]

            query_types = query_ct
            common = set(ref_types) & set(query_types)

            for each in common:
                query = list(query_cell[query_types == each])
                ref = list(cell_atlas_[ref_types == each])
                num_k = min(int(k / 10), 5, len(ref))

                for key in query:
                    new_cells = np.random.choice(ref, num_k, replace=False)
                    if key not in labeled_dict.keys():
                        labeled_dict[key] = list(new_cells)
                    else:
                        labeled_dict[key] += list(new_cells)

    return (labeled_dict)



def mnn(ds1, ds2, names1, names2, knn = 50):
    match1 = nn(ds1, ds2, names1, names2, knn = knn)
    match2 = nn(ds2, ds1, names2, names1, knn = knn)
    mutual = match1 & set([(b,a) for a,b in match2])
    return mutual


def nn(ds1, ds2, names1, names2, knn=50):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]


    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M=16)
    p.set_ef(10)
    p.add_items(ds2)

    min_knn = min(ds1.shape[0], ds2.shape[0])
    if min_knn < knn:
        knn = min_knn
    ind, distances = p.knn_query(ds1, k=knn)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))
    return match
def mnn_(ds1, ds2, names1, names2, knn=50):
    match1, D1 = nn_(ds1, ds2, names1, names2, knn=knn)
    match2, D2 = nn_(ds2, ds1, names2, names1, knn=knn)
    match22 = [list(reversed(sublist)) for sublist in match2]
    merged_data1 = {tuple(row): value for row, value in zip(match1, D1)}
    merged_data2 = {tuple(row): value for row, value in zip(match22, D2)}


    intersection = set(map(tuple, match1)).intersection(map(tuple, match22))

    filtered_data1 = {key: value for key, value in merged_data1.items() if key in intersection}
    filtered_data2 = {key: value for key, value in merged_data2.items() if key in intersection}

    return intersection, filtered_data1


def nn_(ds1, ds2, names1, names2, knn=50):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=50, M=16)
    p.set_ef(10)

    p.add_items(ds2)
    ind, distances = p.knn_query(ds1, k=knn)

    match = []
    D = []
    for a, b in zip(range(ds1.shape[0]), ind):
        D = D + list(distances[a])
        for b_i in b:
            match.append([names1[a], names2[b_i]])

    return match, D

def create_dictionary_mnn(adata=None, techname=None, mask_batch=None, label_cell = None, each=True, knn = 20):



    if mask_batch == None:
        if label_cell == None:
            cell_names = adata.obs_names
            tech_list = adata.obs[techname]

            cells = []
            for i in tech_list.unique():
                cells.append(cell_names[tech_list == i])

            mnns = dict()
            for comb in list(itertools.combinations(range(len(cells)), 2)):
                i = comb[0]
                j = comb[1]

                new = list(cells[j])
                ref = list(cells[i])

                ds1 = adata[new].obsm['pretransfer embedding']
                ds2 = adata[ref].obsm['pretransfer embedding']
                names1 = new
                names2 = ref
                match = mnn(ds1, ds2, names1, names2, knn=knn)

                G = nx.Graph()
                G.add_edges_from(match)
                node_names = np.array(G.nodes)
                anchors = list(node_names)
                adj = nx.adjacency_matrix(G)

                tmp = np.split(adj.indices, adj.indptr[1:-1])
                for i in range(0, len(anchors)):
                    key = anchors[i]
                    i = tmp[i]
                    names = list(node_names[i])
                    mnns[key] = names

        else:

            cell_names = adata.obs_names
            tech_list = adata.obs[techname]
            cells = []
            for i in tech_list.unique():
                cells.append(cell_names[tech_list == i])

            cells_query = list(set(cell_names) - set(list(label_cell.keys())))
            cells_query = pd.Index(cells_query)
            query_list = adata[cells_query, :].obs[techname]
            query_tech = query_list.unique()
            cells_q = []
            for i in query_tech:
                cells_q.append(cells_query[query_list == i])


            if each:

                mnns = dict()
                for l in range(len(query_tech)):

                    tech_new = list(tech_list.unique())
                    tech_new.remove(query_tech[l])

                    for j in range(len(tech_new)):

                        new = list(cells_q[l])
                        ref = list(cells[j])

                        # cell embedding
                        ds1 = adata[new].obsm['pretransfer embedding']
                        ds2 = adata[ref].obsm['pretransfer embedding']
                        names1 = new
                        names2 = ref
                        match = mnn(ds1, ds2, names1, names2, knn=knn)


                        G = nx.Graph()
                        G.add_edges_from(match)
                        node_names = np.array(G.nodes)
                        anchors = list(node_names)
                        adj = nx.adjacency_matrix(G)

                        tmp = np.split(adj.indices, adj.indptr[1:-1])
                        for i in range(0, min(len(new), len(anchors))):
                            key = new[i]
                            i = tmp[i]
                            names = list(node_names[i])
                            mnns[key] = names


            else:
                mnns = dict()
                for j in range(len(query_tech)):

                    ref_ = cell_names[tech_list != query_tech[j]]
                    new = list(cells_q[j])
                    ref = list(ref_)


                    ds1 = adata[new].obsm['pretransfer embedding']
                    ds2 = adata[ref].obsm['pretransfer embedding']
                    names1 = new
                    names2 = ref
                    match = mnn(ds1, ds2, names1, names2, knn=knn)


                    G = nx.Graph()
                    G.add_edges_from(match)
                    node_names = np.array(G.nodes)
                    anchors = list(node_names)
                    adj = nx.adjacency_matrix(G)

                    tmp = np.split(adj.indices, adj.indptr[1:-1])
                    for i in range(0, min(len(new), len(anchors))):
                        key = new[i]
                        i = tmp[i]
                        names = list(node_names[i])
                        mnns[key] = names


    else:

        if each:
            cell_names = adata.obs_names  # all cell
            tech_list = adata.obs[techname]  # all tech
            tech_new = list(set(adata.obs[techname].unique()))
            tech_new.remove(mask_batch)
            cells_ref = cell_names[tech_list == mask_batch]


            cells_new = []
            for i in tech_new:
                cells_new.append(cell_names[tech_list == i])

            mnns = dict()
            for j in range(len(tech_new)):
                new = list(cells_new[j])
                ref = list(cells_ref)

                ds1 = adata[new].obsm['pretransfer embedding']
                ds2 = adata[ref].obsm['pretransfer embedding']
                names1 = new
                names2 = ref
                match = mnn(ds1, ds2, names1, names2, knn=knn)

                G = nx.Graph()
                G.add_edges_from(match)
                node_names = np.array(G.nodes)
                anchors = list(node_names)

                adj = nx.adjacency_matrix(G)

                tmp = np.split(adj.indices, adj.indptr[1:-1])

                for i in range(0, min(len(cells_ref), len(anchors))):
                    key = cells_ref[i]
                    i = tmp[i]
                    names = list(node_names[i])
                    mnns[key] = names


        else:
            cell_names = adata.obs_names  # all cell
            tech_list = adata.obs[techname]  # all tech
            cells_new = cell_names[tech_list != mask_batch]
            cells_ref = cell_names[tech_list == mask_batch]
            if label_cell != None:
                cells_ref = list(set(cells_ref) - set(list(label_cell.keys())))
                cells_ref = pd.Index(cells_ref)

            mnns = dict()
            new = list(cells_new)
            ref = list(cells_ref)

            ds1 = adata[new].obsm['pretransfer embedding']
            ds2 = adata[ref].obsm['pretransfer embedding']
            names1 = new
            names2 = ref
            match = mnn(ds1, ds2, names1, names2, knn=knn)

            G = nx.Graph()
            G.add_edges_from(match)
            node_names = np.array(G.nodes)
            anchors = list(node_names)
            adj = nx.adjacency_matrix(G)

            tmp = np.split(adj.indices, adj.indptr[1:-1])
            for i in range(0, min(len(cells_ref), len(anchors))):
                key = cells_ref[i]
                i = tmp[i]
                names = list(node_names[i])
                mnns[key] = names

    return ({k: v for k, v in sorted(mnns.items(), key=lambda item: item[0])})



def create_dictionary_knn(adata, cell_subset, k=20):
    dataset = adata[cell_subset]
    pcs = dataset.obsm['pretransfer embedding']


    dim = pcs.shape[1]
    num_elements = pcs.shape[0]
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=50, M=16)
    p.set_ef(10)
    p.add_items(pcs)
    ind, distances = p.knn_query(pcs, k=k)

    cell_subset = np.array(cell_subset)
    names = list(map(lambda x: cell_subset[x], ind))
    knns = dict(zip(cell_subset, names))

    return (knns)


def merge_dict(x, y):
    for k, v in x.items():
        if k in y.keys():
            y[k] += v
        else:
            y[k] = v
    return y

