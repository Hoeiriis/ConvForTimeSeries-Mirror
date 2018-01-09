import numpy as np
import keras
import keras.layers as kl
import Components as cmp
from data_loading import make_pred_file


class PrimaryModel:

    def __init__(self, save_path, input_shape=50):
        # Setting static values that are common for all models
        self.loss = keras.losses.binary_crossentropy
        self.optimizer = keras.optimizers.adam
        self.epochs = 500
        self.callbacks = []
        self.input_shape = input_shape

        # Initializing model
        self.model = None

        # Setting sizes
        self.CNN_sizes = [32, 64, 128, 256]
        self.DNN_sizes = [300, 500, 700]

        # For dropout
        self.dropout = None
        self.dropout_counter = 0

        # Data sets
        self.X_train = None
        self.Y_train = None
        self.X_val = None
        self.Y_val = None
        self.X_tournament = None
        self.id_tournament = None

        # CNN?
        self.CNN = True
        self.save_path = save_path

    def set_callbacks(self, callbacks):
        self.callbacks = callbacks

    def set_dataset(self, data_tuple):
        self.X_train, self.Y_train, self.X_val, self.Y_val, self.X_tournament, self.id_tournament = data_tuple

    def save(self, name):
        print("WubbalubbaDUBDUB {}".format(name))

    def run(self, params, name):
        self._reset_param()
        params = params[0]
        p_size = [params["p_size_input"], params["p_size_b1"], params["p_size_b2"],
                  params["p_size_b3"], params["p_size_b4"], params["p_size_output"]]

        self.dropout = [params["dropout_input_blocks"]]*2 + [params["dropout_middle_blocks"]]*2 + [params["dropout_end_blocks"]]*2

        if self.CNN:
            input_ = kl.Input(shape=(self.input_shape, 1))
            self.reshape_to_1D()
            model = self._construct_CNN(input_, p_size, params=params)
        else:
            input_ = kl.Input(shape=(self.input_shape,))
            self.reshape_to_Dense()
            model = self._construct_DNN(input_, p_size)

        # Adding dense layer
        model = kl.Dense(300, activation="relu", kernel_initializer="he_normal")(model)

        # Adding output layer
        output = kl.Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform")(model)

        # Finishing model
        self.model = keras.models.Model(inputs=input_, outputs=output)

        # Training model
        self._train(params["learning_rate"], params["batch_size"])

        best_model = keras.models.load_model(self.save_path + "/{}".format(name))

        return best_model.evaluate(self.X_val, self.Y_val, batch_size=512)[1]*100

    def _train(self, learning_rate, batch_size):
        # compiling
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer(learning_rate),
                           metrics=["accuracy"])
        # training model
        history = self.model.fit(x=self.X_train, y=self.Y_train,
                                 batch_size=batch_size,
                                 epochs=self.epochs,
                                 validation_data=(self.X_val, self.Y_val),
                                 shuffle=True,
                                 callbacks=self.callbacks
                                 )

    def _reset_param(self):
        self.model = None
        self.dropout = None
        self.dropout_counter = 0

    def _add_dropout(self, X_model, DNN=True):

        if self.dropout is not None and self.dropout_counter < len(self.dropout):
            if self.dropout[self.dropout_counter] != 0:
                if DNN:
                    X_model = kl.Dropout(self.dropout[self.dropout_counter])(X_model)
                else:
                    X_model = kl.SpatialDropout1D(self.dropout[self.dropout_counter])(X_model)

            self.dropout_counter += 1
                    
        return X_model

    def _construct_CNN(self, input_, p_size, params):
        # Input block
        k_size = [params["k_size_input"], params["k_size_b1"], params["k_size_b2"], params["k_size_b3"]]
        model = self._construct_CNN_input(input_, p_size, k_size)

        # Block 3 res
        min_3 = (self.CNN_sizes[p_size[3]] - 1) if (self.CNN_sizes[p_size[3]] - 1) > -1 else 0
        filters_res = [min_3, min_3, self.CNN_sizes[p_size[3]]]
        model = cmp.Conv1D_res_block(model, filters_res, k_size[3])

        # Dropout
        model = self._add_dropout(model, DNN=False)

        # Extra blocks, gotta get deep
        model = cmp.Conv1D_res_block(model, [128, 128, 128], kernel_shape=3)

        # Dropout
        model = self._add_dropout(model, DNN=False)

        model = cmp.Conv1D_res_block(model, [128, 128, self.CNN_sizes[p_size[3]]], kernel_shape=3)

        # Dropout
        model = self._add_dropout(model, DNN=False)

        if params["type_b4"] == 0:
            # Block 4 ID
            filters_id = [self.CNN_sizes[p_size[4]], self.CNN_sizes[p_size[3]]]
            model = cmp.Conv1D_identity_block(model, filters_id, kernel_shape=3)
            # Dropout
            model = self._add_dropout(model, DNN=False)
            # Flatten for dense
            model = kl.Flatten()(model)
        else:
            # Flatten for dense
            model = kl.Flatten()(model)
            # Dense ID Block 4
            model = cmp.dense_res_block(model, self.DNN_sizes[p_size[4]])
            # Dropout
            model = self._add_dropout(model, DNN=True)

        return model

    def _construct_DNN(self,input_, p_size):
        # Input blocks
        model = self._construct_DNN_input(input_, p_size)

        # Block 3 Dense Res
        model = cmp.dense_res_block(model, self.DNN_sizes[p_size[3]])

        model = self._add_dropout(model, DNN=True)

        # Block 4 Dense ID
        model = cmp.dense_identityblock(model, self.DNN_sizes[p_size[4]], self.DNN_sizes[p_size[3]])
        # Dropout
        model = self._add_dropout(model, DNN=True)

        return model

    def _construct_CNN_input(self, input_model, p_size, kernels):

        # CNN input block
        model = cmp.Conv1D_input_layer(input_model, self.CNN_sizes[p_size[0]], kernel_shape=kernels[0])

        # CNN Res Block
        min_1 = (self.CNN_sizes[p_size[1]] - 1) if (self.CNN_sizes[p_size[1]] - 1) > -1 else 0
        filters_res = [min_1, min_1, self.CNN_sizes[p_size[1]]]
        model = cmp.Conv1D_res_block(model, n_filters=filters_res, kernel_shape=kernels[1])
        
        # Dropout
        model = self._add_dropout(model, DNN=False)

        # CNN Identity Block
        filters_id = [self.CNN_sizes[p_size[2]], self.CNN_sizes[p_size[1]]]
        model = cmp.Conv1D_identity_block(model, filters_id, kernel_shape=kernels[2])
        
        # Dropout
        model = self._add_dropout(model, DNN=False)
        
        return model

    def _construct_DNN_input(self, input_model, p_size):

        # Input
        model = cmp.dense_simple_layer(input_model, self.DNN_sizes[p_size[0]])
        
        # Dense Res Block
        model = cmp.dense_res_block(model, self.DNN_sizes[p_size[1]])
        
        # Dropout
        model = self._add_dropout(model)
        
        # Dense Identity Block
        model = cmp.dense_identityblock(model, self.DNN_sizes[p_size[2]], self.DNN_sizes[p_size[1]])
        
        # Dropout
        model = self._add_dropout(model)
        
        return model

    def reshape_to_1D(self):

        self.X_train = self.X_train.reshape((-1, self.input_shape, 1))
        self.X_val = self.X_val.reshape((-1, self.input_shape, 1))
        self.X_tournament = self.X_tournament.reshape((-1, self.input_shape, 1))

    def reshape_to_Dense(self):

        self.X_train = self.X_train.reshape((-1, self.input_shape, ))
        self.X_val = self.X_val.reshape(-1, self.input_shape, )
        self.X_tournament = self.X_tournament.reshape((-1, self.input_shape,) )

