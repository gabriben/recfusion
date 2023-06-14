from recpack.quick import quick_train


model = 'RecFusionMLP'
dataset = 'MovieLens100K'
prep_hypers = {
    'ds_path': 'datasets',    
    'min_rating': 4,
    'min_items_per_user': 5,
    'min_users_per_item': 5,
    'generalization': 'WeakGeneralization', # StrongGeneralization,
    "train_val_test": [0.8, 0.1, 0.1],
    "force_even_items": False
}

train_hypers = {
    # "stop_early": False, 
    # "validation_sample_size": 1000,
    # # "dim_bottleneck_layer" : 200,
    # "batch_size" : 200,
    # "T": 100,
    # "schedule_type": 'fixed',
    # 'jascha_bin_process': False,
    # 'b_start' : 0.01,
    # 'max_epochs' : 1,
    # 'b_end' : 0.5,
    # "reparametrization_mu": True,
    # "anneal_steps" : 20    
    # 'xavier_initialization': True
}     
val_metric = {'NDCGK':100}    
test_metrics = {
    'NDCGK' : [10, 20, 50, 100],
    'RecallK' : [10, 20, 50],
    'HitK': [20, 50, 100],
    'CalibratedRecallK': [10, 20, 50]
}

m = quick_train(model, dataset, prep_hypers, train_hypers, val_metric, test_metrics)
m
