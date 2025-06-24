
import pandas as pd
import networkx as nx
import numpy as np
from pickle_file import save_obj, load_obj
import jsonlines
from tqdm import tqdm
from collections import defaultdict, Counter
from random import sample
from CoLoc_class import CoLoc #import the CoLoc class
import scipy

##!=========================================================================
##!cluster sort, color
##!=========================================================================
# region - sort clusters, color graphs
def color_network_clusters(nt, nc):
    
    G = nt

    metro_colors = ['#EA0437', '#87D300', '#FFD100', '#4F1F91', '#A24CC8', '#FF7200', '#009EDB', '#78C7EB', '#BC87E6', '#7C2230', '#007B63', '#D71671', '#F293D1', '#7F7800', '#BBA786', '#32D4CB', '#B67770', '#D6A461', '#DFC765', '#666666', '#999999', '#009090', '#EE352E', '#00933C', '#B933AD', '#808183', '#0039A6', '#FF6319', '#6CBE45', '#996633', '#A7A9AC', '#FCCC0A', '#00ADD0', '#00985F', '#60269E', '#4D5357', '#6E3219', '#CE8E00', '#006983', '#00AF3F', '#C60C30', '#A626AA', '#00A1DE', '#009B3A', '#EE0034', '#8E258D', '#FF7900', '#6E267B', '#A4343A', '#004B87', '#D90627', '#008C95', '#AA0061', '#B58500', '#FFC56E', '#009B77', '#97D700', '#0092BC', '#FF8674', '#9C4F01', '#F4DA40', '#CA9A8E', '#653279', '#6BA539', '#00ABAB', '#D3A3C9', '#F4C1CA', '#D0006F', '#D86018', '#A45A2A', '#D986BA', '#476205', '#D22630', '#A192B2', '#0049A5', '#FF9500', '#F62E36', '#B5B5AC', '#009BBF', '#00BB85', '#C1A470', '#8F76D6', '#00AC9B', '#9C5E31', '#003DA5', '#77C4A3', '#F5A200', '#0C8E72', '#204080', '#C30E2F', '#1CAE4C', '#5288F5', '#E06040', '#3D99C2', '#80E080', '#3D860B', '#3698D2', '#074286', '#1D2A56', '#753778', '#F9BE00', '#2B3990', '#0052A4', '#009D3E', '#EF7C1C', '#00A5DE', '#996CAC', '#CD7C2F', '#747F00', '#EA545D', '#A17E46', '#BDB092', '#B7C452', '#B0CE18', '#0852A0', '#6789CA', '#941E34', '#B21935', '#8A5782', '#9A6292', '#59A532', '#7CA8D5', '#ED8B00', '#FFCD12', '#22246F', '#FDA600', '#0065B3', '#0090D2', '#F97600', '#6FB245', '#509F22', '#ED1C24', '#D4003B', '#8EC31F', '#81A914', '#E04434', '#F04938', '#A17800', '#C7197D', '#5A2149', '#2A5CAA', '#F06A00', '#81BF48', '#BB8C00', '#217DCB', '#D6406A', '#68BD45', '#8652A1', '#004EA2', '#D93F5C', '#00AA80', '#FFB100', '#F08200', '#009088', '#009362', '#007448', '#FF0000', '#009900'] + ['#000000', '#FFFF00', '#1CE6FF', '#FF34FF', '#FF4A46', '#008941', '#006FA6', '#A30059', '#FFDBE5', '#7A4900', '#0000A6', '#63FFAC', '#B79762', '#004D43', '#8FB0FF', '#997D87', '#5A0007', '#809693', '#FEFFE6', '#1B4400', '#4FC601', '#3B5DFF', '#4A3B53', '#FF2F80', '#61615A', '#BA0900', '#6B7900', '#00C2A0', '#FFAA92', '#FF90C9', '#B903AA', '#D16100', '#DDEFFF', '#000035', '#7B4F4B', '#A1C299', '#300018', '#0AA6D8', '#013349', '#00846F', '#372101', '#FFB500', '#C2FFED', '#A079BF', '#CC0744', '#C0B9B2', '#C2FF99', '#001E09', '#00489C', '#6F0062', '#0CBD66', '#EEC3FF', '#456D75', '#B77B68', '#7A87A1', '#788D66', '#885578', '#FAD09F', '#FF8A9A', '#D157A0', '#BEC459', '#456648', '#0086ED', '#886F4C', '#34362D', '#B4A8BD', '#00A6AA', '#452C2C', '#636375', '#A3C8C9', '#FF913F', '#938A81', '#575329', '#00FECF', '#B05B6F', '#8CD0FF', '#3B9700', '#04F757', '#C8A1A1', '#1E6E00', '#7900D7', '#A77500', '#6367A9', '#A05837', '#6B002C', '#772600', '#D790FF', '#9B9700', '#549E79', '#FFF69F', '#201625', '#72418F', '#BC23FF', '#99ADC0', '#3A2465', '#922329', '#5B4534', '#FDE8DC', '#404E55', '#0089A3', '#CB7E98', '#A4E804', '#324E72', '#6A3A4C']

    if isinstance(nc, dict) or isinstance(nc, defaultdict):
        for i, ci_cc in enumerate(nc.items()):
            for n in ci_cc[1]:
                G.nodes[n]['cluster_id'] = ci_cc[0] 
                c = metro_colors[i] if i < len(metro_colors) else '#808080'
                
                G.nodes[n]['viz'] = {}
                G.nodes[n]['viz']['color'] = {'r':int(c[1:3], 16), 'g':int(c[3:5],16),'b':int(c[5:7],16), 'a':1}

    if isinstance(nc, list):
        for i,cc in nc:
            for n in cc:
                G.nodes[n]['cluster_id'] = str(i) 
                c = metro_colors[i] if i < len(metro_colors) else '#808080'
                
                G.nodes[n]['viz'] = {}
                G.nodes[n]['viz']['color'] = {'r':int(c[1:3], 16), 'g':int(c[3:5],16),'b':int(c[5:7],16), 'a':1}
    
    return G


def sort_clusters(C, sort_dict):
    lenc = [len(c) for c in C]
    lenc, C = zip(*sorted(zip(lenc, C), reverse=True))
    Cs = []
    for c in C:
        temp_dict = {k:sort_dict[k] for k in c}
        ctemp = sorted(temp_dict.items(), key = lambda kv:(-kv[1], kv[0]))
        Cs.append([ct[0] for ct in ctemp])
        
    return lenc, Cs

# endregion
##!=========================================================================


def get_tag_pair(year_bool, data_path_save, so_type, tag_bool):
    
    tag_pair_list = []

    for yr, ybool in tqdm(year_bool.items()):
        if ybool:
            with jsonlines.open(f'{data_path_save}all_{so_type}_so_tags_{yr}.json', 'r') as fcc_file:
                for line in fcc_file:
                    k = list(line.keys())[0]
                    v = list(line.values())[0]
                    tag_temp = [t for t in v[0] if tag_bool[t]]
                    tag_pair_list += [(t1,t2) for t1 in tag_temp for t2 in tag_temp if t1 < t2]

            fcc_file.close()

    from collections import Counter
    tag_pair_count_temp = dict(Counter(tag_pair_list))
    tag_pair_count = sorted(tag_pair_count_temp.items(), key = lambda kv:(-kv[1], kv[0]))

    tag1 = [a[0][0] for a in tag_pair_count]
    tag2 = [a[0][1] for a in tag_pair_count]
    tc = [a[1] for a in tag_pair_count]
    sum_c = sum(tc)

    tag_freq = {a:0 for a in tag1 + tag2}
    for t,c in zip(tag1 + tag2, tc + tc):
        tag_freq[t] += c

    tag_freq_sort = sorted(tag_freq.items(), key = lambda kv:(-kv[1], kv[0]))
    tag_index_dict = {ts[0]:i for i,ts in enumerate(tag_freq_sort)}
    index_tag_dict = {i:t for t,i in tag_index_dict.items()}

    tag_pair_index_count = [(tag_index_dict[a[0][0]], tag_index_dict[a[0][1]], a[1]) for a in tag_pair_count]

    
    return tag_pair_index_count, tag_index_dict, index_tag_dict



def get_tag_edgelist_pmi(device_df, p_value = 0.0005):

    dft = device_df.pivot(index = 'tag1', columns = 'tag2', values = 'tag_count')
    dft = dft.fillna(0)
    
    #Q = CoLoc(df_q, prior = 'uniform', nr_prior_obs = np.size(df_q))
    Q = CoLoc(dft)

    df_Q = Q.make_sigPMIpci(p_value)

    df_index = df_Q.index
    df_columns = df_Q.columns

    res = scipy.sparse.coo_matrix(df_Q.fillna(0).values)

    df_res = pd.DataFrame({'tag2':df_columns[res.col], 'tag1':df_index[res.row], 'tag_pair_weight':res.data})

    df_edgelist = df_res[df_res['tag_pair_weight'] > 0]
    df_edgelist = df_edgelist[df_edgelist['tag1'] != df_edgelist['tag2']]

    #df_variance = Q.make_stdPMIpci()

    return df_edgelist, Q



def build_network_from_tag_pair_posterior(tag_pair_index_count, index_tag_dict):

    tpc_threshold = 0
    tag_pair_count_threshold = [a for a in tag_pair_index_count if a[2] > tpc_threshold]

    tag_df = pd.DataFrame({"tag1":[index_tag_dict[a[0]] for a in tag_pair_count_threshold] + [index_tag_dict[a[1]] for a in tag_pair_count_threshold],
                            "tag2":[index_tag_dict[a[1]] for a in tag_pair_count_threshold] + [index_tag_dict[a[0]] for a in tag_pair_count_threshold],
                            "tag_count":[a[2] for a in tag_pair_count_threshold] + [a[2] for a in tag_pair_count_threshold] })

    ##! 计算pmi并建立网络
    ##! 计算pmi并建立网络
    df_edgelist, Q = get_tag_edgelist_pmi(tag_df)
    G = nx.from_pandas_edgelist(df_edgelist, 'tag1', 'tag2', ['tag_pair_weight'])

    print("graph size: ", G.number_of_nodes(), G.number_of_edges())

    C = [c for c in nx.connected_components(G)]
    len_c = [len(c) for c in C]
    len_c, C = zip(*sorted(zip(len_c,C), reverse=True))
    major_component_graph = nx.subgraph(G,[n for n in list(C[0])])
    print("major component size: ", major_component_graph.number_of_nodes(), major_component_graph.number_of_edges())
    
    return major_component_graph, Q


def update_tag_bool(tag_bool, G_tag):
    tag_bool_G = {t:False for t in list(tag_bool.keys())}
    for n in G_tag.nodes():
        tag_bool_G[n] = True

    return tag_bool_G



def get_community_at_level(df_levels, level_index):
    levels_community = {str(ci):[] for ci in df_levels[level_index]}
    for t,ci in zip(df_levels['TAG'], df_levels[level_index]):
        levels_community[str(ci)].append(t)

    return levels_community



def build_network_from_tag_cooccurrence(tag_pair_index_count, index_tag_dict):

    tpc_threshold = 0
    tag_pair_count_threshold = [a for a in tag_pair_index_count if a[2] > tpc_threshold]

    tag_edgelists = [(index_tag_dict[a[0]], index_tag_dict[a[1]], a[2]) for a in tag_pair_count_threshold]

    G = nx.Graph()
    G.add_weighted_edges_from(tag_edgelists, weight = 'tag_pair_weight')

    print("graph size: ", G.number_of_nodes(), G.number_of_edges())

    C = [c for c in nx.connected_components(G)]
    len_c = [len(c) for c in C]
    len_c, C = zip(*sorted(zip(len_c,C), reverse=True))
    major_component_graph = nx.subgraph(G,[n for n in list(C[0])])
    print("major component size: ", major_component_graph.number_of_nodes(), major_component_graph.number_of_edges())
    
    return major_component_graph





def get_tag_community_rca(G, community_list_std, community_tags):
    tag_list_std = []
    for c in community_list_std:
        tag_list_std += community_tags[c]

    tag_dict = {t:i for i,t in enumerate(tag_list_std)}
    community_dict = {c:i for i,c in enumerate(community_list_std)}
    tag_community_dict = {}
    for c,ts in community_tags.items():
        for t in ts:
            tag_community_dict[t] = c

    cooccur_matrix = np.zeros((len(tag_list_std), len(community_list_std)))
    for eg in G.edges():
        n1 = eg[0]
        n2 = eg[1]
        c1 = tag_community_dict[n1]
        c2 = tag_community_dict[n2]

        cooccur_matrix[tag_dict[n1], community_dict[c2]] += G.edges[eg]['tag_pair_weight']
        cooccur_matrix[tag_dict[n2], community_dict[c1]] += G.edges[eg]['tag_pair_weight']
    
    sum1 = np.sum(cooccur_matrix, axis = 1)
    sum2 = np.sum(cooccur_matrix, axis = 0)
    sum0 = np.sum(cooccur_matrix)
    
    rca_matrix = np.zeros(cooccur_matrix.shape)
    
    for i in range(cooccur_matrix.shape[0]):
        for j in range(cooccur_matrix.shape[1]):
            if sum1[i] != 0 and sum2[j] != 0:
                rca_matrix[i][j] = (cooccur_matrix[i][j]/sum0) / (sum1[i]/sum0 * sum2[j]/sum0)
            else:
                rca_matrix[i][j] = 0
    
    rca_values = defaultdict(float)
    for eg in G.edges():
        n1 = eg[0]
        n2 = eg[1]
        c1 = tag_community_dict[n1]
        c2 = tag_community_dict[n2]

        rca_values[(n1, c2)] = rca_matrix[tag_dict[n1], community_dict[c2]]
        rca_values[(n2, c1)] = rca_matrix[tag_dict[n2], community_dict[c1]]

    #rca_matrix = np.where(rca_matrix > 1, rca_matrix, 0)
    
    return rca_matrix, tag_list_std, rca_values


def get_tags_rca_in_community(rca_values, community_unweighted_level, rca_threshold = 1):
    community_tags = {c:[] for c in community_unweighted_level.keys()}
    empty_community = []
    for c, ts in community_unweighted_level.items():
        temp = []
        temp_t = []
        for t in ts:
            if rca_values[(t,c)] > rca_threshold:
                temp.append(rca_values[(t,c)])
                temp_t.append(t)

        if len(temp) > 0:
            _, a = zip(*sorted(zip(temp,temp_t), reverse=False))
            community_tags[c] = [t for t in a]

        else:
            empty_community.append(c)

    core_rca = {}
    for c, ts in community_unweighted_level.items():
        temp = []
        temp_t = []
        for t in ts:
            core_rca[(t,c)] = rca_values[(t,c)]

    core_rca_sorted = sorted(core_rca.items(), key = lambda kv:(kv[1], kv[0]))

    return community_tags, core_rca_sorted, empty_community





def build_tag_question_bipartite(year_bool, data_path, data_path_save, so_type, tag_bool):
    import csv
    with open(data_path_save + 'question_tag_bipartite.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(('question','tag','ecount'))
        for yr, ybool in tqdm(year_bool.items()):
            if ybool:
                with jsonlines.open(f'{data_path}all_{so_type}_so_tags_{yr}.json', 'r') as fcc_file:
                    for line in fcc_file:
                        k = list(line.keys())[0]
                        v = list(line.values())[0]
                        tag_temp = [t for t in v[0] if tag_bool[t]]
                        for t in tag_temp:
                            writer.writerow([k,t,1])

                fcc_file.close()

    csv_file.close()

    return data_path_save + 'question_tag_bipartite.csv'