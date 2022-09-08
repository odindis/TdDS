import module as MD
from itertools import product

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


class UNet_res2_ds( nn.Module ):
    def __init__( self, in_channel, 
                        out_channel, 
                        base_channels = 32, 
                        max_channels  = 320 
                 ):
        super( UNet_res2_ds, self ).__init__()
        ##
        ic = in_channel
        bc = base_channels
        mc = max_channels
        
        ## Encoder
        self.block_0_0 = MD.res2_block( ic  , bc  , [1,3,3] )
        self.block_1_0 = MD.res2_block( bc  , bc*2, [1,3,3] )
        self.block_2_0 = MD.res2_block( bc*2, bc*4, [1,3,3] )
        self.block_3_0 = MD.res2_block( bc*4, bc*8, [3,3,3] )
        self.block_4_0 = MD.res2_block( bc*8, mc  , [3,3,3] )
        self.block_5_0 = MD.res2_block( mc  , mc  , [3,3,3] )
  
        ## Decoder
        self.block_0_1 = MD.res2_block( bc*2    , bc  , [1,3,3] )
        self.block_1_1 = MD.res2_block( bc*4    , bc*2, [1,3,3] )
        self.block_2_1 = MD.res2_block( bc*8    , bc*4, [1,3,3] )
        self.block_3_1 = MD.res2_block( bc*16   , bc*8, [3,3,3] )
        self.block_4_1 = MD.res2_block( mc+mc   , mc  , [3,3,3] )
        
        ## Up_conv
        self.up_1 = MD.uconv( bc*2, bc  , [1,2,2] )
        self.up_2 = MD.uconv( bc*4, bc*2, [1,2,2] )
        self.up_3 = MD.uconv( bc*8, bc*4, [1,2,2] )
        self.up_4 = MD.uconv( mc  , bc*8, [2,2,2] )           
        self.up_5 = MD.uconv( mc  , mc  , [2,2,2] )           
        
        ## Out_conv
        self.out_0 = MD.out_conv( bc  , 2 )
        self.out_1 = MD.out_conv( bc*2, 2 )
        self.out_2 = MD.out_conv( bc*4, 2 )
        self.out_3 = MD.out_conv( bc*8, 2 )
        self.out_4 = MD.out_conv( mc  , 2 )
               
    def forward( self, x ):
        x_0_0 = self.block_0_0( x ) 
        x_1_0 = self.block_1_0( MD.max_pool3d( x_0_0, [1, 2, 2] ) )
        x_2_0 = self.block_2_0( MD.max_pool3d( x_1_0, [1, 2, 2] ) )
        x_3_0 = self.block_3_0( MD.max_pool3d( x_2_0, [1, 2, 2] ) )
        x_4_0 = self.block_4_0( MD.max_pool3d( x_3_0, [2, 2, 2] ) )
        x_5_0 = self.block_5_0( MD.max_pool3d( x_4_0, [2, 2, 2] ) )
        
        x_4_1 = self.block_4_1( MD.cat( self.up_5( x_5_0 ), x_4_0 ) )
        x_3_1 = self.block_3_1( MD.cat( self.up_4( x_4_1 ), x_3_0 ) )
        x_2_1 = self.block_2_1( MD.cat( self.up_3( x_3_1 ), x_2_0 ) )
        x_1_1 = self.block_1_1( MD.cat( self.up_2( x_2_1 ), x_1_0 ) )
        x_0_1 = self.block_0_1( MD.cat( self.up_1( x_1_1 ), x_0_0 ) )
        
        return self.out_0( x_0_1 ), self.out_1( x_1_1 ), self.out_2( x_2_1 ), \
                self.out_3( x_3_1 ), self.out_4( x_4_1 )


def dice_loss( y_pred, y_true ):
    overlap = 0
    bottom  = 0
    s  = torch.tensor( 0.001 ).to( y_pred.device )
    for index in range( len( y_pred ) ):
        overlap += torch.sum( y_pred[ index ] * y_true[ index ] )
        bottom  += torch.sum( y_pred[ index ] ) + torch.sum( y_true[ index ] )
    return 1 - ( 2 * overlap + s ) / ( bottom + s )



def map_combination( image, model, div_loc, patch_shape, out_stage=0):
    #
    shape = list( image.shape )
    shape[1] = 2
    output = torch.zeros( shape, dtype=image.dtype, device=image.device )
    #
    areas_set = []
    for loc_1, loc_2, loc_3 in product( div_loc[0], div_loc[1], div_loc[2] ):
        areas_set.append( [ loc_1, loc_1 + patch_shape[0],
                            loc_2, loc_2 + patch_shape[1],
                            loc_3, loc_3 + patch_shape[2],
                          ]
                        )
    #print()
    for num, area in enumerate( areas_set ):
        #sys.stdout.write( f'\r> areas: {num}/{areas_num}     ')
        image_part = image[ :,:,
                            area[0] : area[1], 
                            area[2] : area[3], 
                            area[4] : area[5] 
                          ]
        o = MD.scaling_target( model( image_part )[out_stage], image_part )
        
        output[ :,:,
                area[0] : area[1], 
                area[2] : area[3], 
                area[4] : area[5] 
              ]  += o
    #
    return output

               
def calculate_loss( outputs, labels ):
    loss = torch.tensor(0.).to( labels.device )
    for output in outputs: ## Difference stage
        output = F.softmax( output, 1 )
        if output.shape[2:] != labels.shape[2:]:
            output = MD.scaling_target( output, labels )
        num = len( output[:,1] )
        loss2 = torch.tensor(0.).to( labels.device )
        for index in range( num ): ## Difference data
            loss2 += dice_loss( output[index][1], labels[index][0] )
        loss += torch.div( loss2, num )
    return loss
               
def calculate_loss_2( outputs, labels ):
    loss = torch.tensor(0.).to( labels.device )
    for output in outputs: ## Difference stage
        output = F.softmax( output, 1 )
        if output.shape[2:] != labels.shape[2:]:
            kernel = np.array( labels.shape[2:] ) / np.array( output.shape[2:] )
            kernel = list( kernel.astype( np.int32 ) )
            labels = MD.max_pool3d( labels, kernel )
        num = len( output[:,1] )
        loss2 = torch.tensor(0.).to( labels.device )
        for index in range( num ): ## Difference data
            loss2 += dice_loss( output[index][1], labels[index][0] )
        loss += torch.div( loss2, num )
    return loss







