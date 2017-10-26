from ml.ml_model import Model

m = Model()

m.load_data('data')
m.build_model()
m.train()