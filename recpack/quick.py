from recpackfusion.recpack.datasets import *
from recpack.scenarios import WeakGeneralization, StrongGeneralization
from recpack.pipelines import PipelineBuilder
import pandas as pd
from recpack.preprocessing.filters import MinRating, MinItemsPerUser, MinUsersPerItem

from recpack.preprocessing.filters import NMostPopular

import wandb

def quick_train(model: str, 
                dataset: str, 
                prep_hypers: dict, 
                train_hypers: dict, 
                val_metric: dict, 
                test_metrics: dict):

    wandb.init(mode = "online")

    wandb.config.dataset = dataset
    wandb.config.update(prep_hypers)

    d = eval(dataset)(path='datasets/', filename=dataset+'.csv', use_default_filters=False)

    d.add_filter(MinRating(prep_hypers['min_rating'], d.RATING_IX))
    d.add_filter(MinItemsPerUser(prep_hypers['min_items_per_user'], d.ITEM_IX, d.USER_IX))
    d.add_filter(MinUsersPerItem(prep_hypers['min_users_per_item'], d.ITEM_IX, d.USER_IX))
    x = d.load()

    scenario = eval(prep_hypers['generalization'])(prep_hypers['train_prop'], validation=True)
    scenario.split(x)
    builder = PipelineBuilder()
    builder.set_data_from_scenario(scenario)
    builder.add_algorithm(model, params=train_hypers)

    for m, K in val_metric.items():
      builder.set_optimisation_metric(m, K=K)

    for m, K in test_metrics.items():
      builder.add_metric(m, K=K)

    pipeline = builder.build()
    pipeline.run()

    return pipeline.get_metrics(short = True)
