import mlww.train as train
import mlww.generate as generate
import numpy as np
import h5py


def train_CNN_model(val_cutoff=1400, n_epochs=50):
    training_dataset = train.NeutronDataset("../../mcdc_inputs/input_parameters.h5", "../../mcdc_inputs/tally_results.h5", end_idx=val_cutoff)
    validation_dataset = train.NeutronDataset("../../mcdc_inputs/input_parameters.h5", "../../mcdc_inputs/tally_results.h5", start_idx = val_cutoff+1)
    training_dataset.normalize_source()
    validation_dataset.normalize_source()
    # these normalization methods have not yet improved training results
    #training_dataset.normalize_results()
    #validation_dataset.normalize_results()
    #training_dataset.normalize_xs()
    #validation_dataset.normalize_xs()
    training_dataset.to_torch()
    validation_dataset.to_torch()

    train_model = train.TrainCNN(training_dataset, validation_dataset, max_epochs=n_epochs, tolerance=1e-9, lr=1e-3)
    train_model.train()
    train_model.plot_errors()
    train_model.save_model("../models/", "mlww_5x5x4_030525")

def compare_results(case_num):
    loaded_model = train.ModelLoader("../models/mlww_5x5x4_030525.pt")
    actual_output, predicted_output = loaded_model.compare_flux("../../mcdc_inputs/input_parameters.h5", "../../mcdc_inputs/tally_results.h5", case_number=case_num, normalize_source = True, normalize_results=False, normalize_xs=False)
    loaded_model.plot_compare_flux(actual_output, predicted_output, z_slice=2)
    rel_error = np.abs(predicted_output - actual_output)/actual_output * 100
    print("Mean error of case", case_num, ":", np.mean(rel_error))



#train_CNN_model(val_cutoff = 1999, n_epochs=50)
compare_results(case_num=0)


