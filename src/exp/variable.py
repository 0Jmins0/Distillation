class ExperimentConfig:
    def __init__(self, model_name='MVCNN_CLIP', batch_size=8, lr=1e-06, num_epochs=14):
        self.model_name = model_name
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

class TrainConfig(ExperimentConfig):
    def __init__(self, model_name='MVCNN_CLIP', batch_size=8, lr=1e-06, num_epochs=14,
                 train_dataset='OS-MN40-core', train_num_views=12, model_num=0, margin=1.0, experiment_name="exp1"):
        super().__init__(model_name, batch_size, lr, num_epochs)
        self.train_dataset = train_dataset
        self.train_num_views = train_num_views
        self.model_num = model_num
        self.margin = margin
        self.experiment_name = experiment_name

class TestConfig(ExperimentConfig):
    def __init__(self, model_name='MVCNN_CLIP', batch_size=8, lr=1e-06, num_epochs=14,
                 test_dataset='OS-MN40-core', test_num_views=12, model_num=0, margin=1.0):
        super().__init__(model_name, batch_size, lr, num_epochs)
        self.test_dataset = test_dataset
        self.test_num_views = test_num_views
        self.model_num = model_num
        self.margin = margin