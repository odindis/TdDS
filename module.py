import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

#==============================================================================
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
                          align_corners=False 
                        )

def dice_loss( y_pred, y_true ):
    overlap = torch.tensor( 0.0001 ).to( y_pred.device )
    bottom  = torch.tensor( 0.0001 ).to( y_pred.device )
    for index in range( len( y_pred ) ):
        overlap += torch.sum( y_pred[ index ] * y_true[ index ] )
        bottom  += torch.sum( y_pred[ index ] ) + torch.sum( y_true[ index ] )
    return 1 - 2 * overlap / bottom

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

class res2_block( nn.Module ):
    def __init__(self, in_channel, out_channel, kernel_size ):
        super( res2_block, self).__init__()
        kernel_size = np.array( kernel_size ).astype( np.int32 )
        self.is_same = (in_channel == out_channel)
        ## Conv_0
        if not self.is_same:
            self.conv_0 = nn.Conv3d( in_channel, 
                                     out_channel, 
                                     kernel_size = kernel_size, 
                                     padding     = (kernel_size//2).tolist(), 
                                     stride      = 1,
                                     bias        = True,
                                   ) 
            self.norm_0 = nn.InstanceNorm3d( out_channel )
        ## Conv_1
        self.conv_1 = nn.Conv3d( out_channel, 
                                 out_channel, 
                                 kernel_size = kernel_size, 
                                 padding     = (kernel_size//2).tolist(), 
                                 stride      = 1,
                                 bias        = True,
                               )
        self.norm_1 = nn.InstanceNorm3d( out_channel )
        ## Conv_2
        self.conv_2 = nn.Conv3d( out_channel, 
                                 out_channel, 
                                 kernel_size = kernel_size, 
                                 padding     = (kernel_size//2).tolist(), 
                                 stride      = 1,
                                 bias        = True,
                               )
        self.norm_2 = nn.InstanceNorm3d( out_channel )
        ## Non_line
        self.non_line = nn.LeakyReLU( inplace=True )
        
    def forward( self, x ):
        if not self.is_same:
            x = self.non_line( self.norm_0( self.conv_0( x ) ) )
        x_1 = self.non_line( self.norm_1( self.conv_1( x ) ) )
        x_2 = self.conv_2( x_1 )
        return self.non_line( torch.add( self.norm_2( x_2 ), x ) )

class double_conv( nn.Module ):
    def __init__( self, in_channel, out_channel, kernel_size ):
        super( double_conv, self ).__init__()
        kernel_size = np.array( kernel_size ).astype( np.int32 )   
        ## Conv_1
        self.conv_1 = nn.Conv3d( in_channel, 
                                 out_channel, 
                                 kernel_size = kernel_size, 
                                 padding     = (kernel_size//2).tolist(), 
                                 stride      = 1,
                                 bias        = True,
                               )
        self.norm_1 = nn.InstanceNorm3d( out_channel )
        ## Conv_2
        self.conv_2 = nn.Conv3d( out_channel, 
                                 out_channel, 
                                 kernel_size = kernel_size, 
                                 padding     = (kernel_size//2).tolist(), 
                                 stride      = 1,
                                 bias        = True,
                               )
        self.norm_2 = nn.InstanceNorm3d( out_channel )
        ## Non_line
        self.non_line = nn.LeakyReLU( inplace=True )
        
    def forward( self, x ):
        x_1 = self.non_line( self.norm_1( self.conv_1( x   ) ) )
        x_2 = self.non_line( self.norm_2( self.conv_2( x_1 ) ) )
        return x_2
    
class uconv( nn.Module ):
    def __init__( self, in_channel, out_channel, kernel_size ):
        super( uconv, self).__init__()
        
        self.conv_1 = nn.ConvTranspose3d( in_channel,
                                          out_channel, 
                                          kernel_size = kernel_size, 
                                          stride      = kernel_size
                                        )        
                    
    def forward( self, x ):
        return self.conv_1( x )


class trans_conv( nn.Module ):
    def __init__( self, in_channel, out_channel  ):
        super( trans_conv, self).__init__()
        self.conv_1 = nn.Conv3d( in_channel , 
                                 out_channel, 
                                 kernel_size = [1,1,1], 
                                 padding     = [0,0,0], 
                                 stride      = 1,
                                 bias        = True,
                               )
        self.norm_1   = nn.InstanceNorm3d( out_channel )
        self.non_line = nn.LeakyReLU( inplace=True )
        
                    
    def forward( self, x ):
        return self.non_line( self.norm_1( self.conv_1( x ) ) )

    
class out_conv( nn.Module ):
    def __init__( self, in_channel, out_channel,  ):
        super( out_conv, self ).__init__()    
        self.conv_1 = nn.Conv3d( in_channel, 
                                 out_channel, 
                                 kernel_size = 1, 
                                 stride      = 1, 
                                 padding     = 0, 
                                 bias        = False
                               )
    def forward( self, x ):
        return self.conv_1( x )



def max_pool3d( x, kernel ):
    return F.max_pool3d( x, kernel, kernel )


def cat( x, y ):
    return torch.cat( [ x, y ], dim=1 )




class res1_block( nn.Module ):
    def __init__(self, in_channel, out_channel, kernel_size ):
        super( res1_block, self).__init__()
        kernel_size = np.array( kernel_size ).astype( np.int32 )
        self.is_same = (in_channel == out_channel)
        ## Conv_0
        if not self.is_same:
            self.conv_0 = nn.Conv3d( in_channel, 
                                     out_channel, 
                                     kernel_size = kernel_size, 
                                     padding     = (kernel_size//2).tolist(), 
                                     stride      = 1,
                                     bias        = True,
                                   ) 
            self.norm_0 = nn.InstanceNorm3d( out_channel )
        ## Conv_1
        self.conv_1 = nn.Conv3d( out_channel, 
                                 out_channel, 
                                 kernel_size = kernel_size, 
                                 padding     = (kernel_size//2).tolist(), 
                                 stride      = 1,
                                 bias        = True,
                               )
        self.norm_1 = nn.InstanceNorm3d( out_channel )
        ## Conv_2
        self.conv_2 = nn.Conv3d( out_channel, 
                                 out_channel, 
                                 kernel_size = kernel_size, 
                                 padding     = (kernel_size//2).tolist(), 
                                 stride      = 1,
                                 bias        = True,
                               )
        self.norm_2 = nn.InstanceNorm3d( out_channel )
        ## Non_line
        self.non_line = nn.LeakyReLU( inplace=True )
        
    def forward( self, x ):
        if not self.is_same:
            x = self.non_line( self.norm_0( self.conv_0( x ) ) )
        x_1 = self.non_line( torch.add( self.norm_1( self.conv_1( x   ) ), x   ) )
        x_2 = self.non_line( torch.add( self.norm_2( self.conv_2( x_1 ) ), x_1 ) )
        return x_2


class res1_block_2( nn.Module ):
    def __init__(self, in_channel, out_channel, kernel_size ):
        super( res1_block_2, self).__init__()
        kernel_size = np.array( kernel_size ).astype( np.int32 )
        self.is_same = (in_channel == out_channel)
        ## Conv_0
        if not self.is_same:
            self.conv_0 = nn.Conv3d( in_channel, 
                                     out_channel, 
                                     kernel_size = kernel_size, 
                                     padding     = (kernel_size//2).tolist(), 
                                     stride      = 1,
                                     bias        = True,
                                   ) 
            self.norm_0 = nn.InstanceNorm3d( out_channel )
        ## Conv_1
        self.conv_1 = nn.Conv3d( out_channel, 
                                 out_channel, 
                                 kernel_size = kernel_size, 
                                 padding     = (kernel_size//2).tolist(), 
                                 stride      = 1,
                                 bias        = True,
                               )
        self.norm_1 = nn.InstanceNorm3d( out_channel )
        ## Conv_2
        self.conv_2 = nn.Conv3d( out_channel, 
                                 out_channel, 
                                 kernel_size = kernel_size, 
                                 padding     = (kernel_size//2).tolist(), 
                                 stride      = 1,
                                 bias        = True,
                               )
        self.norm_2 = nn.InstanceNorm3d( out_channel )
        ## Non_line
        self.non_line = nn.LeakyReLU( inplace=False )
        
    def forward( self, x ):
        if not self.is_same:
            x = self.non_line( self.norm_0( self.conv_0( x ) ) )
        x_1 = torch.add( x  , self.norm_1( self.conv_1( self.non_line( x   ) ) ) )            
        x_2 = torch.add( x_1, self.norm_2( self.conv_2( self.non_line( x_1 ) ) ) )
        return x_2


