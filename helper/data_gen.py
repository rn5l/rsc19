import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, batch_size=4096, continuous_features=[], cat_features=[], shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.y = y
        self.X = X
        self.shuffle = shuffle
        self.continuous_features = continuous_features
        self.cat_features = cat_features
        
        self.indices = y.index.values

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indices)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_cont = self.X.loc[indices][self.continuous_features].values.astype(np.float32)
        
        X = [X_cont]
        for col in self.cat_features:
            X.append( self.X.loc[indices][col].values.astype(np.float32) )
        
        y = self.y.loc[indices].values.astype(np.float32)

        return X, y

class BalancedDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X, y, batch_size=4096, mult=10, continuous_features=[], cat_features=[], shuffle=True ):
        'Initialization'
        self.batch_size = batch_size
        self.y = y
        self.X = X
        self.shuffle = shuffle
        self.mult=mult
        self.continuous_features = continuous_features
        self.cat_features = cat_features
        
        self.indices_pos = y[y==1].index.values
        self.indices_neg = y[y==0].index.values
        
        self.create_index()
        self.batches = int(np.floor( (len(self.indices) / self.batch_size)) )

        
    def create_index(self):
        self.indices = []
        self.indices += list(self.indices_pos)*self.mult
        for i in range(self.mult):
            np.random.shuffle( self.indices_neg )
            self.indices += list( self.indices_neg[:len(self.indices_pos)] )
        
        self.indices = np.array( self.indices )
        np.random.shuffle( self.indices )
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indices)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            self.create_index()

    def __data_generation(self, indices):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_cont = self.X.loc[indices][self.continuous_features]#.values.astype(np.float32)
        
        X = [X_cont]
        for col in self.cat_features:
            X.append( self.X.loc[indices][col] )#.values.astype(np.float32) )
        
        y = self.y.loc[indices] #.values.astype(np.float32)

        return X, y
    
    