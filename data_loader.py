from math import  ceil
import skimage.io as io
import numpy as np

#==============================================================================
class data_loader_io:
    def __init__( self, 
                  data_path_set, 
                  batch_size   = 1, 
                  is_cut       = False, 
                  auto_shuffle = True 
                ):
        self.data_path_set = data_path_set
        self.batch_size    = batch_size
        #
        cut_n = len( data_path_set) / batch_size 
        self.cycle = int( cut_n if is_cut else ceil( cut_n ) )
        #
        self.start = 0
        self.start_r = 0
        self.auto_shuffle = auto_shuffle
    ##
    def get_data( self, data_path_set ):
        data_set  = []
        label_set = []
        for data_path in data_path_set:
            ## Select the reading mode according to the actual situation
            file = np.load( data_path, allow_pickle=True )
            file_array, file_label = file['image'], file['label']          
            data_set .append( file_array )
            label_set.append( file_label )
        return data_set, label_set
    ##
    def get_batch_path( self ):
        start = self.start * self.batch_size
        data_loc = self.data_path_set[ start : ( start + self.batch_size ) ]
        return data_loc
    ##
    def get_batch_data( self, move_index=True ):
        if self.auto_shuffle and 0 == self.start < self.start_r:
            self.shuffle()
        data_set, label_set = self.get_data( self.get_batch_path() )
        #
        self.start_r = self.start
        if move_index:
            self.start = self.start + 1
            self.start = self.start if self.start < self.cycle else 0
        #
        return data_set, label_set
    ##
    def shuffle( self ):
        np.random.shuffle( self.data_path_set )
    ##
    def __call__( self, move_index=True ):
        return self.get_batch_data( move_index )




# if __name__ == '__main__':
#     train_data_loader = data_loader_io( train_path, 2 )
#     data_set, label_set = train_data_loader()

