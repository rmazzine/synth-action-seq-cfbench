import sys

sys.path.append('../')

import time

import numpy as np
import pandas as pd
from cfbench.cfbench import BenchmarkCF, TOTAL_FACTUAL

from modelssynas import loader
from heuristics.loader import load_heuristics
from recourse.search import SequenceSearch
from recourse.config import base_config

from benchmark.utils import timeout, TimeoutError

# Get initial and final index if provided
if len(sys.argv) == 3:
    initial_idx = sys.argv[1]
    final_idx = sys.argv[2]
else:
    initial_idx = 0
    final_idx = TOTAL_FACTUAL

# Create Benchmark Generator
benchmark_generator = BenchmarkCF(
    output_number=2,
    show_progress=True,
    disable_tf2=True,
    disable_gpu=True,
    initial_idx=int(initial_idx),
    final_idx=int(final_idx)).create_generator()

# The Benchmark loop
synas_current_dataset = None
for benchmark_data in benchmark_generator:
    # Get factual array
    factual_array = benchmark_data['factual_oh']

    # Get train data
    train_data = benchmark_data['df_oh_train']

    # Get columns info
    columns = list(train_data.columns)[:-1]

    # Get factual row as pd.Series
    factual_row = pd.Series(benchmark_data['factual_oh'], index=columns)

    # Get factual class
    fc = benchmark_data['factual_class']

    # Get Keras TensorFlow model
    model = benchmark_data['model']

    # Get Evaluator
    evaluator = benchmark_data['cf_evaluator']

    converter = benchmark_data['oh_converter']

    cat_feats = benchmark_data['cat_feats']
    num_feats = benchmark_data['num_feats']

    session = benchmark_data['tf_session']

    if benchmark_data['dsname'] != synas_current_dataset:

        class model_synas:
            def __init__(self, adapted_nn):
                self.FALSE_LABEL = [0.0, 1.0]
                self.TRUE_LABEL = [1.0, 0.0]
                self.input_dim = adapted_nn.layers[1].get_weights()[0].shape[0]
                self.model = adapted_nn

        model_synas_nn = model_synas(model)

        bin_feats = converter.binary_cats if cat_feats else []
        dict_feat_idx = converter.dict_feat_idx if cat_feats else []

        data, actions, features, target_label = loader.setup_generic(train_data, cat_feats, num_feats, bin_feats,
                                                                     dict_feat_idx)

        for name, feature in features.items():
            feature.initialize_tf_variables()

        heuristics = load_heuristics('vanilla', actions, model_synas_nn, 1)
        cfg = SequenceSearch(model_synas_nn, actions, heuristics, sav_dir=None, config=base_config)

        synas_current_dataset = benchmark_data['dsname']


    @timeout(600)
    def generate_cf():
        try:
            # Create CF using SYNAS' explainer and measure generation time
            start_generation_time = time.time()
            cf_generation_output = cfg.find_correction(
                np.array(factual_array).reshape((1, len(factual_array))),
                np.array([target_label]), session)
            cf_generation_time = time.time() - start_generation_time

            if cf_generation_output.best_result is not None:
                if model.predict(np.array([cf_generation_output.best_result.final_instance]))[0][1] >= 0.5:
                    cf = cf_generation_output.best_result.final_instance.tolist()
                else:
                    cf = factual_array
            else:
                cf = factual_array

            if factual_array != cf:
                print('CF candidate generated')
            else:
                print('No CF generated')

        except Exception as e:
            print('Error generating CF')
            print(e)
            # In case the CF generation fails, return same as factual
            cf = factual_row.to_list()
            cf_generation_time = np.NaN

        # Evaluate CF
        evaluator(
            cf_out=cf,
            algorithm_name='synas',
            cf_generation_time=cf_generation_time,
            save_results=True)
    try:
        generate_cf()
    except TimeoutError:
        print('Timeout generating CF')
        # If CF generation time exceeded the limit
        evaluator(
            cf_out=factual_row.to_list(),
            algorithm_name='synas',
            cf_generation_time=np.NaN,
            save_results=True)
