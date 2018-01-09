from data_loading import NMRDataLoader, make_pred_file, cd
import keras
import numpy as np
import sklearn.metrics as skm

if __name__ == "__main__":
    type_ = "CNN"
    number = 89
    desktop = True

    if desktop:
        path = "C:/SOFTWARE and giggles/NMR_tuning/t_{}".format(number)
        storage_path = "C:/SOFTWARE and giggles/NMR_tuning/1st_tuning_round/preds{}".format(number)
    else:
        path = "D:/SOFTWARE/numerai_data/t_{}".format(number)
        storage_path = "D:/SOFTWARE/MI Jule tests/preds{}".format(number)

    nmr_loader = NMRDataLoader(path)
    X_train, Y_train, X_val, Y_val, X_tournament, id_tournament = nmr_loader.get_data_small_val(val_era_size=None)

    val_pred_list = []
    # top_performingCNN = [27, 46, 49, 50, 84, 85, 99, 129, 169, 194, 204, 223, 228, 238, 260, 270]
    top_performingCNN = []
    top_performingDNN = [45, 62, 83, 87, 96]
    # top_performingDNN = []

    import os

#    for file in os.listdir(storage_path):
#        if file.endswith(".npy"):
#            top_performingCNN.append(file)

    for i in top_performingCNN:
        model_name = "CNN_models_param_{}".format(i)
        print("Loading model {}".format(model_name))

        with cd(storage_path):
            val_pred_list.append(np.load("{}_predictions.npy".format(model_name)))
            #val_pred_list.append(np.load("{}".format(i)))

    for i in top_performingDNN:
        model_name = "DNN_models_param_{}".format(i)
        print("Loading model {}".format(model_name))

        with cd(storage_path):
            val_pred_list.append(np.load("{}_predictions.npy".format(model_name)))

    combined = None
    for entry in val_pred_list:
        if combined is None:
            combined = entry
        else:
            combined = combined + entry

    ensemble_pred = combined/(len(top_performingCNN)+len(top_performingDNN))

    print("test: {}".format(ensemble_pred[0]))

    make_pred_file(storage_path=storage_path, predictions=ensemble_pred, ids=id_tournament,
                   name="DNN_top{}_pred.csv".format(len(top_performingDNN)))