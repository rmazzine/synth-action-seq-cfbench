import pandas as pd

from actions.feature import loader as feature_loader
from modelssynas.generic import get_actions as get_actions_generic

class generic_dataset:

    def __init__(self, dataset):
        self.data = dataset.drop(columns=['output']).to_numpy()
        self.labels = pd.concat([dataset['output'], abs(dataset['output']-1)], axis=1).to_numpy()

def setup_generic(dataset, cat_feats, num_feats, bin_feats, dict_feat_idx, used_actions=None):

    raw_features = []
    for idx in range(dataset.shape[1]):
        if str(idx) in num_feats:
            input_data = {'type': 'numeric',
                          'name': str(idx),
                          'idx': list(dataset.columns).index(str(idx)),
                          'i': idx,
                          'num_values': 1,
                          'mean': dataset[str(idx)].mean(),
                          'std': dataset[str(idx)].std()
                          }
            raw_features.append(input_data)
        if str(idx) in cat_feats:
            input_data = {'type': 'nominal',
                          'name': str(idx),
                          'values': [1]*len(dict_feat_idx[str(idx)].keys()),
                          'idx': min(list(dict_feat_idx[str(idx)].values())),
                          'i': idx,
                          'num_values': len(dict_feat_idx[str(idx)].keys())}

            raw_features.append(input_data)

    features = feature_loader('', raw_features)
    actions = get_actions_generic(features)

    dataset = generic_dataset(dataset)

    return dataset, actions, features, [0.0, 1.0]
