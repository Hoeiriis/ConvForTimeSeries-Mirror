from src import SingleParam, Tuner, ParamConfig, ParamLog
from data_loading import NMRDataLoader, cd
from PrimaryModel import PrimaryModel
import numpy as np


def stopper(trials):
    if trials > 4:
        return True

    return False


if __name__ == "__main__":

    # Setting up variables and paths for main runner
    data_set = 1
    data_path = "C:/Users/jeppe/Dropbox/MI/Data{}".format(data_set)
    save_path = "C:/Users/jeppe/Dropbox/MI"

    print("Initializing tuner")

    params_CNN = (
        SingleParam("learning_rate", output_type="double", value_range=(0.0005, 0.002), scaling="log"),
        SingleParam("p_size_input", output_type="discrete", value_range=[0, 1, 2]),
        SingleParam("p_size_b1", output_type="discrete", value_range=[0, 1, 2]),
        SingleParam("p_size_b2", output_type="discrete", value_range=[0, 1, 2]),
        SingleParam("p_size_b3", output_type="discrete", value_range=[0, 1, 2, 3]),
        SingleParam("p_size_b4", output_type="discrete", value_range=[0, 1, 2, 3]),
        SingleParam("p_size_output", output_type="discrete", value_range=[0, 1, 2]),
        SingleParam("k_size_input", output_type="integer", value_range=(3, 7), scaling="incremental", increment=1),
        SingleParam("k_size_b1", output_type="integer", value_range=(3, 7), scaling="incremental", increment=1),
        SingleParam("k_size_b2", output_type="integer", value_range=(3, 7), scaling="incremental", increment=1),
        SingleParam("k_size_b3", output_type="integer", value_range=(3, 7), scaling="incremental", increment=1),
        SingleParam("type_b4", output_type="discrete", value_range=[0, 1]),
        SingleParam("dropout_input_blocks", output_type="double", value_range=(0, 0.7), scaling="incremental", increment=0.1),
        SingleParam("dropout_end_blocks", output_type="double", value_range=(0, 0.7), scaling="incremental", increment=0.1),
        SingleParam("batch_size", output_type="discrete", value_range=[64, 128, 256, 512])
    )

    params_DNN = (
        SingleParam("learning_rate", output_type="double", value_range=(0.0005, 0.002), scaling="log"),
        SingleParam("p_size_input", output_type="discrete", value_range=[0, 1, 2]),
        SingleParam("p_size_b1", output_type="discrete", value_range=[0, 1, 2]),
        SingleParam("p_size_b2", output_type="discrete", value_range=[0, 1, 2]),
        SingleParam("p_size_b3", output_type="discrete", value_range=[0, 1, 2]),
        SingleParam("p_size_b4", output_type="discrete", value_range=[0, 1, 2]),
        SingleParam("p_size_output", output_type="discrete", value_range=[0, 1, 2]),
        SingleParam("dropout_input_blocks", output_type="double", value_range=(0, 0.7), scaling="incremental", increment=0.1),
        SingleParam("dropout_end_blocks", output_type="double", value_range=(0, 0.7), scaling="incremental", increment=0.1),
        SingleParam("batch_size", output_type="discrete", value_range=[64, 128, 256, 512])
    )

    # Make model maker
    king_sam = PrimaryModel(save_path)

    # Suggestors
    suggestors = {"ZoomRandomSearch": {"trials_per_zoom": 40, "n_eval_trials": 30}}

    p_config = ParamConfig()
    rescaler_functions_CNN, param_names_CNN = p_config.make_rescale_dict(params_CNN)
    rescaler_functions_DNN, param_names_DNN = p_config.make_rescale_dict(params_DNN)

    # Loading and initializing param logs
    with cd(save_path):
        CNN_actual = np.load("CNN_models_params_actual.npy")
        CNN_unscaled = np.load("CNN_models_params_unscaled.npy")
        CNN_score = np.load("CNN_models_params_scores.npy")
        DNN_actual = np.load("DNN_models_params_actual.npy")
        DNN_unscaled = np.load("DNN_models_params_unscaled.npy")
        DNN_score = np.load("DNN_models_params_scores.npy")

    CNN_param_log = ParamLog(len(rescaler_functions_CNN), actual=CNN_actual, unscaled=CNN_unscaled, score=CNN_score,
                             param_descriptions=params_CNN)
    DNN_param_log = ParamLog(len(rescaler_functions_DNN), actual=DNN_actual, unscaled=DNN_unscaled, score=DNN_score,
                             param_descriptions=params_DNN)

    # Initializing tuners
    tuner_CNN = Tuner("CNN_models", sam=king_sam, param_config=params_CNN,
                      suggestors=suggestors, save_path=save_path, param_log=CNN_param_log)

    tuner_DNN = Tuner("DNN_models", sam=king_sam, param_config=params_DNN,
                      suggestors=suggestors, save_path=save_path, param_log=DNN_param_log)

    # Main runner
    print("Tuning")
    while True:
        print("Loading data")
        nmr_loader = NMRDataLoader(data_path)
        data_tuple = nmr_loader.get_data_small_val(val_era_size=None)

        king_sam.set_dataset(data_tuple)

        # CNN:
        print("Tuning CNNs")
        king_sam.CNN = True
        tuner_CNN.tune(stopper)

        # DNN:
        print("Tuning DNNs")
        king_sam.CNN = False
        tuner_DNN.tune(stopper)

        # Loading "new" data
        data_set = 2 if data_set == 1 else 1
        data_path = "C:/Users/jeppe/Dropbox/MI/Data{}".format(data_set)
