from module import ( double_conv, out_conv, uconv, max_pool3d, cat, 
                     res2_block, res1_block, res1_block_2
                   )
from itertools import product

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from einops import rearrange


## Model ======================================================================
## UNet -----------------------------------------------------------------------
class UNet( nn.Module ):
    def __init__( self, in_channel, 
                        out_channel, 
                        base_channels = 32, 
                        max_channels  = 320 
                 ):
        super( UNet, self ).__init__()
        ##
        ic = in_channel
        bc = base_channels
        mc = max_channels
        
        ## Encoder
        self.block_0_0 = double_conv( ic  , bc  , [1,3,3] )#16,256
        self.block_1_0 = double_conv( bc  , bc*2, [1,3,3] )#16,128
        self.block_2_0 = double_conv( bc*2, bc*4, [1,3,3] )#16,64
        self.block_3_0 = double_conv( bc*4, bc*8, [3,3,3] )#16,32
        self.block_4_0 = double_conv( bc*8, mc  , [3,3,3] )#8,16
        self.block_5_0 = double_conv( mc  , mc  , [3,3,3] )#4,8
  
        ## Decoder
        self.block_0_1 = double_conv( bc*2    , bc  , [1,3,3] )
        self.block_1_1 = double_conv( bc*4    , bc*2, [1,3,3] )
        self.block_2_1 = double_conv( bc*8    , bc*4, [1,3,3] )
        self.block_3_1 = double_conv( bc*16   , bc*8, [3,3,3] )
        self.block_4_1 = double_conv( mc+mc   , mc  , [3,3,3] )
        
        ## Up_conv
        self.up_1 = uconv( bc*2, bc  , [1,2,2] )
        self.up_2 = uconv( bc*4, bc*2, [1,2,2] )
        self.up_3 = uconv( bc*8, bc*4, [1,2,2] )
        self.up_4 = uconv( mc  , bc*8, [2,2,2] )           
        self.up_5 = uconv( mc  , mc  , [2,2,2] )           
        
        ## Out_conv
        self.out_0 = out_conv( bc  , 2 )
        
    def forward( self, x ):
        x_0_0 = self.block_0_0( x ) 
        x_1_0 = self.block_1_0( max_pool3d( x_0_0, [1, 2, 2] ) )
        x_2_0 = self.block_2_0( max_pool3d( x_1_0, [1, 2, 2] ) )
        x_3_0 = self.block_3_0( max_pool3d( x_2_0, [1, 2, 2] ) )
        x_4_0 = self.block_4_0( max_pool3d( x_3_0, [2, 2, 2] ) )
        x_5_0 = self.block_5_0( max_pool3d( x_4_0, [2, 2, 2] ) )
        
        x_4_1 = self.block_4_1( cat( self.up_5( x_5_0 ), x_4_0 ) )
        x_3_1 = self.block_3_1( cat( self.up_4( x_4_1 ), x_3_0 ) )
        x_2_1 = self.block_2_1( cat( self.up_3( x_3_1 ), x_2_0 ) )
        x_1_1 = self.block_1_1( cat( self.up_2( x_2_1 ), x_1_0 ) )
        x_0_1 = self.block_0_1( cat( self.up_1( x_1_1 ), x_0_0 ) )
        
        return ( self.out_0( x_0_1 ), )       

## UNet_res2 ------------------------------------------------------------------
class UNet_res2( nn.Module ):
    def __init__( self, in_channel, 
                        out_channel, 
                        base_channels = 32, 
                        max_channels  = 320 
                 ):
        super( UNet_res2, self ).__init__()
        ##
        ic = in_channel
        bc = base_channels
        mc = max_channels
        
        ## Encoder
        self.block_0_0 = res2_block( ic  , bc  , [1,3,3] )
        self.block_1_0 = res2_block( bc  , bc*2, [1,3,3] )
        self.block_2_0 = res2_block( bc*2, bc*4, [1,3,3] )
        self.block_3_0 = res2_block( bc*4, bc*8, [3,3,3] )
        self.block_4_0 = res2_block( bc*8, mc  , [3,3,3] )
        self.block_5_0 = res2_block( mc  , mc  , [3,3,3] )
  
        ## Decoder
        self.block_0_1 = res2_block( bc*2    , bc  , [1,3,3] )
        self.block_1_1 = res2_block( bc*4    , bc*2, [1,3,3] )
        self.block_2_1 = res2_block( bc*8    , bc*4, [1,3,3] )
        self.block_3_1 = res2_block( bc*16   , bc*8, [3,3,3] )
        self.block_4_1 = res2_block( mc+mc   , mc  , [3,3,3] )
        
        ## Up_conv
        self.up_1 = uconv( bc*2, bc  , [1,2,2] )
        self.up_2 = uconv( bc*4, bc*2, [1,2,2] )
        self.up_3 = uconv( bc*8, bc*4, [1,2,2] )
        self.up_4 = uconv( mc  , bc*8, [2,2,2] )           
        self.up_5 = uconv( mc  , mc  , [2,2,2] )           
        
        ## Out_conv
        self.out_0 = out_conv( bc  , 2 )
        
    def forward( self, x ):
        x_0_0 = self.block_0_0( x ) 
        x_1_0 = self.block_1_0( max_pool3d( x_0_0, [1, 2, 2] ) )
        x_2_0 = self.block_2_0( max_pool3d( x_1_0, [1, 2, 2] ) )
        x_3_0 = self.block_3_0( max_pool3d( x_2_0, [1, 2, 2] ) )
        x_4_0 = self.block_4_0( max_pool3d( x_3_0, [2, 2, 2] ) )
        x_5_0 = self.block_5_0( max_pool3d( x_4_0, [2, 2, 2] ) )
        
        x_4_1 = self.block_4_1( cat( self.up_5( x_5_0 ), x_4_0 ) )
        x_3_1 = self.block_3_1( cat( self.up_4( x_4_1 ), x_3_0 ) )
        x_2_1 = self.block_2_1( cat( self.up_3( x_3_1 ), x_2_0 ) )
        x_1_1 = self.block_1_1( cat( self.up_2( x_2_1 ), x_1_0 ) )
        x_0_1 = self.block_0_1( cat( self.up_1( x_1_1 ), x_0_0 ) )
        
        return ( self.out_0( x_0_1 ), )       
    
## UNet_res2 attention --------------------------------------------------------
class Attention_block( nn.Module ):
    def __init__( self, F_g, F_l, F_int ):
        super( Attention_block, self ).__init__()
        self.W_g = nn.Sequential( 
            nn.Conv3d( F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True ),
            nn.InstanceNorm3d( F_int )
                                )
        self.W_x = nn.Sequential(
            nn.Conv3d( F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True ),
            nn.InstanceNorm3d( F_int )
                                )
        self.psi = nn.Sequential(
            nn.Conv3d( F_int,   1, kernel_size=1, stride=1, padding=0, bias=True ),
            nn.InstanceNorm3d( 1 ),
            nn.Sigmoid()
                                )
        self.relu = nn.ReLU( inplace=True )

    def forward( self, g, x ):
        g1  = self.W_g( g )
        x1  = self.W_x( x )
        psi = self.relu( g1 + x1 )
        psi = self.psi( psi )
        return x * psi, psi
    
class UNet_res2_att( nn.Module ):
    def __init__( self, in_channel, 
                        out_channel, 
                        base_channels = 32, 
                        max_channels  = 320 
                 ):
        super( UNet_res2_att, self ).__init__()
        ##
        ic = in_channel
        bc = base_channels
        mc = max_channels
        
        ## Encoder
        self.block_0_0 = res2_block( ic  , bc  , [1,3,3] )
        self.block_1_0 = res2_block( bc  , bc*2, [1,3,3] )
        self.block_2_0 = res2_block( bc*2, bc*4, [1,3,3] )
        self.block_3_0 = res2_block( bc*4, bc*8, [3,3,3] )
        self.block_4_0 = res2_block( bc*8, mc  , [3,3,3] )
        self.block_5_0 = res2_block( mc  , mc  , [3,3,3] )
  
        ## Decoder
        self.block_0_1 = res2_block( bc*2    , bc  , [1,3,3] )
        self.block_1_1 = res2_block( bc*4    , bc*2, [1,3,3] )
        self.block_2_1 = res2_block( bc*8    , bc*4, [1,3,3] )
        self.block_3_1 = res2_block( bc*16   , bc*8, [3,3,3] )
        self.block_4_1 = res2_block( mc+mc   , mc  , [3,3,3] )
        
        ## Attention
        self.att_0 = Attention_block( bc  , bc  , bc   )
        self.att_1 = Attention_block( bc*2, bc*2, bc*2 )
        self.att_2 = Attention_block( bc*4, bc*4, bc*4 )
        self.att_3 = Attention_block( bc*8, bc*8, bc*8 )
        self.att_4 = Attention_block( mc, mc, mc )
        
        ## Up_conv
        self.up_1 = uconv( bc*2, bc  , [1,2,2] )
        self.up_2 = uconv( bc*4, bc*2, [1,2,2] )
        self.up_3 = uconv( bc*8, bc*4, [1,2,2] )
        self.up_4 = uconv( mc  , bc*8, [2,2,2] )           
        self.up_5 = uconv( mc  , mc  , [2,2,2] )           
        
        ## Out_conv
        self.out_0 = out_conv( bc  , 2 )
     
    # def forward( self, x ):
    #     x_0_0 = self.block_0_0( x ) 
    #     x_1_0 = self.block_1_0( max_pool3d( x_0_0, [1, 2, 2] ) )
    #     x_2_0 = self.block_2_0( max_pool3d( x_1_0, [1, 2, 2] ) )
    #     x_3_0 = self.block_3_0( max_pool3d( x_2_0, [1, 2, 2] ) )
    #     x_4_0 = self.block_4_0( max_pool3d( x_3_0, [2, 2, 2] ) )
    #     x_5_0 = self.block_5_0( max_pool3d( x_4_0, [2, 2, 2] ) )
        
    #     u5 = self.up_5( x_5_0 )
    #     att4, pis4 = self.att_4( u5, x_4_0 )
    #     x_4_1 = self.block_4_1( cat( att4, u5 ) )
        
    #     u4 = self.up_4( x_4_1 )
    #     att3, pis3 = self.att_3( u4, x_3_0 )
    #     x_3_1 = self.block_3_1( cat( att3, u4 ) )
        
    #     u3 = self.up_3( x_3_1 )
    #     att2, pis2 =  self.att_2( u3, x_2_0 )
    #     x_2_1 = self.block_2_1( cat( att2, u3 ) )
        
    #     u2 = self.up_2( x_2_1 )
    #     att1, pis1 = self.att_1( u2, x_1_0 )
    #     x_1_1 = self.block_1_1( cat( att1, u2 ) )
        
    #     u1 = self.up_1( x_1_1 )
    #     att0, pis0 = self.att_0( u1, x_0_0 )
    #     x_0_1 = self.block_0_1( cat( att0, u1 ) )
        
    #     return ( pis0, pis1, pis2, pis3, pis4 )  
    
    def forward( self, x ):
        x_0_0 = self.block_0_0( x ) 
        x_1_0 = self.block_1_0( max_pool3d( x_0_0, [1, 2, 2] ) )
        x_2_0 = self.block_2_0( max_pool3d( x_1_0, [1, 2, 2] ) )
        x_3_0 = self.block_3_0( max_pool3d( x_2_0, [1, 2, 2] ) )
        x_4_0 = self.block_4_0( max_pool3d( x_3_0, [2, 2, 2] ) )
        x_5_0 = self.block_5_0( max_pool3d( x_4_0, [2, 2, 2] ) )
        
        u5 = self.up_5( x_5_0 )
        x_4_1 = self.block_4_1( cat( self.att_4( u5, x_4_0 )[0], u5 ) )
        
        u4 = self.up_4( x_4_1 )
        x_3_1 = self.block_3_1( cat( self.att_3( u4, x_3_0 )[0], u4 ) )
        
        u3 = self.up_3( x_3_1 )
        x_2_1 = self.block_2_1( cat( self.att_2( u3, x_2_0 )[0], u3 ) )
        
        u2 = self.up_2( x_2_1 )
        x_1_1 = self.block_1_1( cat( self.att_1( u2, x_1_0 )[0], u2 ) )
        
        u1 = self.up_1( x_1_1 )
        x_0_1 = self.block_0_1( cat( self.att_0( u1, x_0_0 )[0], u1 ) )
        
        return ( self.out_0( x_0_1 ), )  
    
    
## UNet_res2 ------------------------------------------------------------------
#def dice_loss( y_pred, y_true ):
#    overlap = torch.tensor( 0.0001 ).to( y_pred.device )
#    bottom  = torch.tensor( 0.0001 ).to( y_pred.device )
#    for index in range( len( y_pred ) ):
#        overlap += torch.sum( y_pred[ index ] * y_true[ index ] )
#        bottom  += torch.sum( y_pred[ index ] ) + torch.sum( y_true[ index ] )
#    return 1 - 2 * overlap / bottom

def dice_loss( y_pred, y_true ):
    overlap = torch.tensor( 0.000 ).to( y_pred.device )
    bottom  = torch.tensor( 0.000 ).to( y_pred.device )
    s  = torch.tensor( 0.001 ).to( y_pred.device )
    for index in range( len( y_pred ) ):
        overlap += torch.sum( y_pred[ index ] * y_true[ index ] )
        bottom  += torch.sum( y_pred[ index ] ) + torch.sum( y_true[ index ] )
    return 1 - ( 2 * overlap + s ) / ( bottom + s )

def dice_v( y_pred, y_true ):
    smooth  = torch.tensor( 0.0001 ).to( y_pred.device )
    overlap = torch.sum( y_pred * y_true )
    bottom  = torch.sum( y_pred ) + torch.sum( y_true )
    return ( 2 * overlap + smooth ) / ( bottom + smooth )

def dice_v_2( y_pred, y_true ):
    dice_sum = torch.tensor( 0. ).to( y_pred.device )
    for index in range( len( y_pred ) ):
        dice_sum += dice_v( y_pred[ index ], y_true[ index ] )
    return dice_sum / len( y_pred )

def dice_t( y_pred, y_true ):
    dice_list = []
    p_list = []
    t_list = []
    for index in range( len( y_pred ) ):
        dice_ = float( dice_v( y_pred[ index ], y_true[ index ] ) )
        p = 1 if y_pred[ index ].bool().any() else 0
        t = 1 if y_true[ index ].bool().any() else 0
        dice_list.append( dice_ )
        p_list.append( p )
        t_list.append( t )
    dice_list = np.array( dice_list )
    p_list = np.array( p_list )
    t_list = np.array( t_list )
    # 
    s = ( p_list == t_list )
    #
    return dice_list.mean(), \
           dice_list[ np.where( t_list ) ].mean(), \
           dice_list[ np.where( p_list + t_list ) ].mean(), \
           s.mean()
           
def Region_measure( y_pred, y_true ):
    s = torch.tensor( 0.0001 ).to( y_pred.device ) # smooth
    #    
    intersection  = torch.sum( y_pred * y_true )
    pre = torch.sum( y_pred )
    lab = torch.sum( y_true )
    #
    dice      = ( 2 * intersection + s ) / ( pre + lab + s )
    recall    = (     intersection + s ) / (       lab + s )
    precision = (     intersection + s ) / ( pre       + s )
    #
    return float( dice ), float( recall ), float( precision )


def map_combination( image, model, div_loc, patch_shape, out_stage):
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
        o = scaling_target( model( image_part )[out_stage], image_part )
        
        output[ :,:,
                area[0] : area[1], 
                area[2] : area[3], 
                area[4] : area[5] 
              ]  += o
    #
    return output


def map_combination_tdds( image, model, div_loc, patch_shape, out_stage ):
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
        o0 = scaling_target( model( image_part )[0], image_part )
        o1 = scaling_target( model( image_part )[1], image_part )
        o2 = scaling_target( model( image_part )[2], image_part )
        o3 = scaling_target( model( image_part )[3], image_part )
         
        o0 = torch.softmax( o0, dim=1 )
        o1 = torch.softmax( o1, dim=1 )
        o2 = torch.softmax( o2, dim=1 )
        o3 = torch.softmax( o3, dim=1 )

        output[ :,:,
                area[0] : area[1], 
                area[2] : area[3], 
                area[4] : area[5] 
              ]  += ( o0 * o1 * o2 * o3 )
    #
    return output


def map_combination_2( image, model, div_loc, patch_shape, result_index ):
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
    #
    for num, area in enumerate( areas_set ):
        #sys.stdout.write( f'\r> areas: {num}/{areas_num}     ')
        image_part = image[ :,:,
                            area[0] : area[1], 
                            area[2] : area[3], 
                            area[4] : area[5],
                          ]
        o = model( image_part )[ result_index ]
        o_shape = o.shape
        output[ :,:,
                area[0] : area[1], 
                area[2] : area[3], 
                area[4] : area[5] 
              ] += scaling_target( o, image_part )
    #
    return output, o_shape

def batch_map_combination( images, model, div_loc_list, patch_shape ):
    outputs = torch.zeros( 0, dtype=images.dtype, device=images.device )
    for index_0, image in enumerate( images ):
        outputs[ index_0 ] = map_combination( image, 
                                              model, 
                                              div_loc_list[ index_0 ], 
                                              patch_shape 
                                            )
    return outputs


#model = UNet( 1, 2 ).cuda()
#x = torch.zeros( [ 1,1,32,256,256 ] ).cuda()
#y = model( x )
    
        
## UNet res2 ds ---------------------------------------------------------------
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
        self.block_0_0 = res2_block( ic  , bc  , [1,3,3] )
        self.block_1_0 = res2_block( bc  , bc*2, [1,3,3] )
        self.block_2_0 = res2_block( bc*2, bc*4, [1,3,3] )
        self.block_3_0 = res2_block( bc*4, bc*8, [3,3,3] )
        self.block_4_0 = res2_block( bc*8, mc  , [3,3,3] )
        self.block_5_0 = res2_block( mc  , mc  , [3,3,3] )
  
        ## Decoder
        self.block_0_1 = res2_block( bc*2    , bc  , [1,3,3] )
        self.block_1_1 = res2_block( bc*4    , bc*2, [1,3,3] )
        self.block_2_1 = res2_block( bc*8    , bc*4, [1,3,3] )
        self.block_3_1 = res2_block( bc*16   , bc*8, [3,3,3] )
        self.block_4_1 = res2_block( mc+mc   , mc  , [3,3,3] )
        
        ## Up_conv
        self.up_1 = uconv( bc*2, bc  , [1,2,2] )
        self.up_2 = uconv( bc*4, bc*2, [1,2,2] )
        self.up_3 = uconv( bc*8, bc*4, [1,2,2] )
        self.up_4 = uconv( mc  , bc*8, [2,2,2] )           
        self.up_5 = uconv( mc  , mc  , [2,2,2] )           
        
        ## Out_conv
        self.out_0 = out_conv( bc  , 2 )
        self.out_1 = out_conv( bc*2, 2 )
        self.out_2 = out_conv( bc*4, 2 )
        self.out_3 = out_conv( bc*8, 2 )
        self.out_4 = out_conv( mc  , 2 )
        
    # def forward( self, x ):
    #     x_0_0 = self.block_0_0( x ) 
    #     x_1_0 = self.block_1_0( max_pool3d( x_0_0, [1, 2, 2] ) )
    #     x_2_0 = self.block_2_0( max_pool3d( x_1_0, [1, 2, 2] ) )
    #     x_3_0 = self.block_3_0( max_pool3d( x_2_0, [1, 2, 2] ) )
    #     x_4_0 = self.block_4_0( max_pool3d( x_3_0, [2, 2, 2] ) )
    #     x_5_0 = self.block_5_0( max_pool3d( x_4_0, [2, 2, 2] ) )
        
    #     x_4_1 = self.block_4_1( cat( self.up_5( x_5_0 ), x_4_0 ) )
    #     x_3_1 = self.block_3_1( cat( self.up_4( x_4_1 ), x_3_0 ) )
    #     x_2_1 = self.block_2_1( cat( self.up_3( x_3_1 ), x_2_0 ) )
    #     x_1_1 = self.block_1_1( cat( self.up_2( x_2_1 ), x_1_0 ) )
    #     x_0_1 = self.block_0_1( cat( self.up_1( x_1_1 ), x_0_0 ) )
        
    #     return x_0_1, x_1_1, x_2_1, x_3_1, x_4_1
               
    def forward( self, x ):
        x_0_0 = self.block_0_0( x ) 
        x_1_0 = self.block_1_0( max_pool3d( x_0_0, [1, 2, 2] ) )
        x_2_0 = self.block_2_0( max_pool3d( x_1_0, [1, 2, 2] ) )
        x_3_0 = self.block_3_0( max_pool3d( x_2_0, [1, 2, 2] ) )
        x_4_0 = self.block_4_0( max_pool3d( x_3_0, [2, 2, 2] ) )
        x_5_0 = self.block_5_0( max_pool3d( x_4_0, [2, 2, 2] ) )
        
        x_4_1 = self.block_4_1( cat( self.up_5( x_5_0 ), x_4_0 ) )
        x_3_1 = self.block_3_1( cat( self.up_4( x_4_1 ), x_3_0 ) )
        x_2_1 = self.block_2_1( cat( self.up_3( x_3_1 ), x_2_0 ) )
        x_1_1 = self.block_1_1( cat( self.up_2( x_2_1 ), x_1_0 ) )
        x_0_1 = self.block_0_1( cat( self.up_1( x_1_1 ), x_0_0 ) )
        
        return self.out_0( x_0_1 ), self.out_1( x_1_1 ), self.out_2( x_2_1 ), \
                self.out_3( x_3_1 ), self.out_4( x_4_1 )


def scaling_cat( input_, target ):
    return torch.cat( [ scaling_target( input_, target ), target], dim=1 )

def scaling_target( input_, target, mode='trilinear' ):
    return F.interpolate( input_, 
                          target.shape[2:], 
                          mode = mode, 
                          align_corners=False 
                        )

def scaling_shape( input_, shape, mode='trilinear' ):
    return F.interpolate( input_, 
                          shape, 
                          mode = mode, 
                          align_corners = False, 
                        )
               
def calculate_loss( outputs, labels ):
    loss = torch.tensor(0.).to( labels.device )
    for output in outputs: ## Difference stage
        output = F.softmax( output, 1 )
        if output.shape[2:] != labels.shape[2:]:
            output = scaling_target( output, labels )
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
            labels = max_pool3d( labels, kernel )
        num = len( output[:,1] )
        loss2 = torch.tensor(0.).to( labels.device )
        for index in range( num ): ## Difference data
            loss2 += dice_loss( output[index][1], labels[index][0] )
        loss += torch.div( loss2, num )
    return loss







