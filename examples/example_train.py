import mlww.train as train
import numpy as np

def train_CNN_model():
    training_dataset = train.NeutronDataset("../../mcdc_inputs/input_parameters.h5", "../../mcdc_inputs/tally_results.h5", end_idx=500)
    validation_dataset = train.NeutronDataset("../../mcdc_inputs/input_parameters.h5", "../../mcdc_inputs/tally_results.h5", start_idx = 501)

    train_model = train.TrainCNN(training_dataset, validation_dataset, max_epochs=50, tolerance=1e-9, lr=1e-3)
    train_model.train()
    train_model.plot_errors()
    train_model.save_model("../models/", "mlww_5x5x4_030525")

train_CNN_model()
loaded_model = train.ModelLoader("../models/mlww_5x5x4_030525.pt")
actual_output, predicted_output = loaded_model.compare_flux("../../mcdc_inputs/input_parameters.h5", "../../mcdc_inputs/tally_results.h5", case_number=0)
rel_error = np.abs(predicted_output - actual_output)/actual_output * 100
print(np.mean(rel_error))