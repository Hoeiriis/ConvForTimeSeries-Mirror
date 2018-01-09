from data_loading import NMRDataLoader, make_pred_file, cd
import keras
import numpy as np

if __name__ == "__main__":
    type = "DNN"
    number = 88
    path = "D:/SOFTWARE/numerai_data/t_{}".format(number)
    nmr_loader = NMRDataLoader(path)
    X_train, Y_train, X_val, Y_val, X_tournament, id_tournament = nmr_loader.get_data_small_val(val_era_size=None)

    dnn_list = [45, 62, 83, 87, 96]

    for i in dnn_list:
        model_name = "{}_models_param_{}".format(type, i)
        model_path = "D:/SOFTWARE/MI Jule tests/{}s/{}".format(type, model_name)

        print("Loading model {}".format(model_name))
        model = keras.models.load_model(model_path)

        #train_eval = model.evaluate(X_train.reshape((-1, 50, 1)), Y_train, batch_size=512)
        print("Evaluating on validation set")
        #val_eval = model.evaluate(X_val.reshape((-1, 50, 1)), Y_val, batch_size=512)
        val_eval = model.evaluate(X_val, Y_val, batch_size=512)

        #print("Train loss: {}".format(train_eval[0]))
        #print("Train accuracy: {}".format(train_eval[1]))

        print("Validation loss: {}".format(val_eval[0]))
        print("Validation accuracy: {}".format(val_eval[1]))

        if val_eval[1] > 0.5175:
            print("Predicting on X_tournament")
            predictions = model.predict(X_tournament, batch_size=512)
            #predictions = model.predict(X_tournament.reshape((-1, 50, 1)), batch_size=512)

        print("Printing results to csv")
        storage_path = "D:/SOFTWARE/MI Jule tests/preds{}".format(number)
        with cd(storage_path):
            with open("{}_evaluations.csv".format(type), "a") as the_file:

                the_file.write("{}: \n Validation accuracy: {} \n Validation loss: {} \n"
                               .format(model_name, val_eval[1], val_eval[0]))

            if val_eval[1] > 0.5175:
                np.save("{}_predictions".format(model_name), predictions)

        if val_eval[1] > 0.5175:
            print("Making prediction file")
            make_pred_file(storage_path, predictions=predictions, ids=id_tournament, name=model_name)

