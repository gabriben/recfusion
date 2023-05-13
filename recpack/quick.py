from recpack.datasets import *
from recpack.scenarios.splitters import *
from recpack.scenarios import WeakGeneralization, StrongGeneralization
from recpack.pipelines import PipelineBuilder
import pandas as pd
from recpack.preprocessing.filters import MinRating, MinItemsPerUser, MinUsersPerItem

from recpack.preprocessing.filters import NMostPopular

import wandb

import random

def quick_train(model: str, 
                dataset: str, 
                prep_hypers: dict,
                # architecture_hypers: dict,
                train_hypers: dict,
                val_metric: dict, 
                test_metrics: dict):

    wandb.init(mode = "online")

    wandb.config.dataset = dataset
    wandb.config.update(prep_hypers)
    wandb.config.update({"model": model})
    wandb.config.update(prep_hypers)   

    
    p = prep_hypers['ds_path'] if 'ds_path' in prep_hypers.keys() else 'datasets/'
    
    d = eval(dataset)(path=p, use_default_filters=False)
    if dataset != "MillionSongDataset":
        d.add_filter(MinRating(prep_hypers['min_rating'], d.RATING_IX))
    d.add_filter(MinItemsPerUser(prep_hypers['min_items_per_user'], d.ITEM_IX, d.USER_IX))
    d.add_filter(MinUsersPerItem(prep_hypers['min_users_per_item'], d.ITEM_IX, d.USER_IX))

    x = d.load()
    if prep_hypers['force_even_items'] and x.shape[1] % 2 != 0:
        d = eval(dataset)(path='datasets/', filename=dataset+'.csv', use_default_filters=False)
        if dataset != "MillionSongDataset":        
            d.add_filter(MinRating(prep_hypers['min_rating'], d.RATING_IX))
        d.add_filter(MinItemsPerUser(prep_hypers['min_items_per_user'], d.ITEM_IX, d.USER_IX))
        d.add_filter(MinUsersPerItem(prep_hypers['min_users_per_item'], d.ITEM_IX, d.USER_IX))
        d.add_filter(NMostPopular(x.shape[1] - 1, d.ITEM_IX))
        x = d.load()

    # first split train_val and test    
    train, val, test = prep_hypers["train_val_test"]
    train_val = train + val
    
    scenario = eval(prep_hypers['generalization'])(train_val, validation=True, seed = random.randint(0, 10e10))
    # then split train and val from train_val
    if prep_hypers['generalization'] == 'StrongGeneralization':
        scenario.validation_splitter = StrongGeneralizationSplitter(in_frac=train/train_val, seed=scenario.seed)
    elif prep_hypers['generalization'] == 'WeakGeneralization':
        scenario.validation_splitter = FractionInteractionSplitter(in_frac=train/train_val, seed=scenario.seed)
    # e.g. 0.9 [train_val] * 0.88 -> 0.8 [train] / 0.1 [val]

    scenario.split(x)
    builder = PipelineBuilder()
    builder.set_data_from_scenario(scenario)
    builder.add_algorithm(model, params=train_hypers) # .update(architecture_hypers)

    if len(val_metric) > 1:
        for m, K in val_metric.items():
            builder.set_optimisation_metric(m, K=K)

    for m, K in test_metrics.items():
      builder.add_metric(m, K=K)

    pipeline = builder.build()
    pipeline.run()

    last_val = wandb.run.summary["val_ndcg@50"] if "val_ndcg@50" in wandb.run.summary.keys() else 0

    table_of_results = pipeline.get_metrics(short=True)

    wandb.finish()

    return (table_of_results, last_val)
