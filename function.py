from email.mime.text import MIMEText
from skimage import io
from skimage import morphology

import metric as mt
import numpy as np
import smtplib
import random
import torch
import os



#==============================================================================
def fill_embedding( image, patch_shape, background=0 ):
    f_image = np.zeros( patch_shape, dtype=image.dtype )
    if background != 0:
        f_image += background
    f_image[ :image.shape[0], :image.shape[1], :image.shape[2] ] = image
    return f_image

from numpy.random import randint as rdt
from numpy.random import rand as rd

def batch_crop_data( images, 
                     labels, 
                     patch_shape, 
                     is_focus       = True, 
                     focus_roi_rate = 0.5, 
                     target_r       = 0.5, 
                     n_target_r     = 0.3, 
                     num_max_find   = 15
                   ):
    images_list = []
    labels_list = []
    for image, label in zip( images, labels ):
        ## Fill: patch may be better than image_ Shape large
        if ( image.shape < patch_shape ).any():
            new_shape = np.max( [ image.shape, patch_shape ], axis=0 )
            image = fill_embedding( image, new_shape )
            label = fill_embedding( label, new_shape )
        ## Focus ---
        if is_focus:
            label_sum = np.sum( label>0 )
            ## Focus on ROI
            if rd() < focus_roi_rate :
                for _ in range( num_max_find ): ## 
                    ## Cut loc
                    shape = label.shape - patch_shape
                    clip_start_loc = ( rdt( -1, shape[0] ) +1, 
                                       rdt( -1, shape[1] ) +1,
                                       rdt( -1, shape[2] ) +1,
                                     )
                    clip_end_loc = clip_start_loc + patch_shape
                    ## Cut        
                    cut_label = label[ clip_start_loc[0]:clip_end_loc[0],
                                       clip_start_loc[1]:clip_end_loc[1],
                                       clip_start_loc[2]:clip_end_loc[2],
                                     ]
                    cut_image = image[ clip_start_loc[0]:clip_end_loc[0],
                                       clip_start_loc[1]:clip_end_loc[1],
                                       clip_start_loc[2]:clip_end_loc[2],
                                     ]
                    cut_label_sum = np.sum( cut_label>0 )
                    #
                    if cut_label_sum / label_sum >= target_r:
                        break
            ## Focus on no_ROI
            else:
                for _ in range( num_max_find ): ## 
                    ## Cut loc
                    shape = label.shape - patch_shape
                    clip_start_loc = ( rdt( -1, shape[0] )+1, 
                                       rdt( -1, shape[1] )+1,
                                       rdt( -1, shape[2] )+1, 
                                     )
                    clip_end_loc = clip_start_loc + patch_shape
                    ## Cut
                    cut_image = image[ clip_start_loc[0]:clip_end_loc[0],
                                       clip_start_loc[1]:clip_end_loc[1],
                                       clip_start_loc[2]:clip_end_loc[2],
                                     ]
            
                    cut_label = label[ clip_start_loc[0]:clip_end_loc[0],
                                       clip_start_loc[1]:clip_end_loc[1],
                                       clip_start_loc[2]:clip_end_loc[2],
                                     ]
                    cut_label_sum = np.sum( cut_label>0 )
                    #
                    if cut_label_sum / label_sum <= n_target_r:
                        break
        # No Focus
        else:
            ## Cut loc
            shape = label.shape - patch_shape
            clip_start_loc = ( rdt( 0, shape[0] ), 
                                rdt( 0, shape[1] ),
                                rdt( 0, shape[2] ) 
                              )
            clip_end_loc = clip_start_loc + patch_shape
            ## Cut        
            cut_label = label[ clip_start_loc[0]:clip_end_loc[0],
                                clip_start_loc[1]:clip_end_loc[1],
                                clip_start_loc[2]:clip_end_loc[2],
                              ]
            cut_image = image[ clip_start_loc[0]:clip_end_loc[0],
                                clip_start_loc[1]:clip_end_loc[1],
                                clip_start_loc[2]:clip_end_loc[2],
                              ]

        ## Load data into list
        images_list.append( cut_image )
        labels_list.append( cut_label )
    #
    return np.array( images_list )[:,np.newaxis], \
           np.array( labels_list )[:,np.newaxis]


def crop_test_data( image, label, patch_shape, no_overlap=1/2 ):
    ## Fill: patch may be better than image Shape large
    if ( image.shape < patch_shape ).any():
        new_shape = np.max( [ image.shape, patch_shape ], axis=0 )
        image = fill_embedding( image, new_shape )
        label = fill_embedding( label, new_shape )
    ## No overlap calculation
    data_shape = image.shape
    block_num = ( data_shape / patch_shape ).astype( np.int32 )
    ban_set = np.array( [ True for _ in patch_shape ] )
    nol = ( data_shape / block_num ) / patch_shape
    while ( ( nol > no_overlap ) * ban_set ).any():
        block_num[ nol > no_overlap ] += 1
        nol = (data_shape / block_num) / patch_shape
        # ban_loc = ( nol <= 0.5 )
        ban_loc = ( nol < no_overlap )
        if ban_loc.any():
            # block_num[ ban_loc ] -= 1
            ban_set[ ban_loc ] = False
    interval = data_shape / block_num
    ## Computational site
    div_loc = []
    for index in range( len( block_num ) ):
        len_ = block_num[index]-1
        ## Calculate split points by interval
        locs = [ int( interval[ index ] * _ ) for _ in range( len_ ) ]
        ## The last point uses a specific algorithm
        if len_ == 0:
            loc = 0
        else:
            loc = int( data_shape[ index ] - patch_shape[ index ] )
        locs.append( loc )
        div_loc.append( locs )
    ## Remove overflow sites
    for index in range( len( div_loc ) ):
        locs = np.array( div_loc[ index ] )
        loc_True = ( locs < locs[-1] )
        loc_True[-1] = True
        locs = locs[ loc_True ]
        div_loc[ index ] = locs.tolist()
    #
    image = image[ np.newaxis, np.newaxis ]
    label = label[ np.newaxis, np.newaxis ]
    return image, label, div_loc



def norm255( images ):
    return ( images - images.min() ) / ( images.max() - images.min() )

def set_pre_and_label( image, pre, label, save_path, value='' ):
    if image.shape != pre.shape != label.shape :
        raise Exception( '"image.shape != pre.shape != label.shape" is True' )
    #
    image = np.repeat( np.expand_dims( image, 3 ), 3, axis=3)
    image_b = np.copy( image )
    # image_1 = np.copy( image )
    # image_2 = np.copy( image )
    #
    pre   = pre  .astype( np.bool_ )
    label = label.astype( np.bool_ )
    #
    # image_1[ pre   ] = [1,0,0]
    # image_2[ label ] = [0,0,1]
    #
    #
    correct = ( pre * label ).astype( np.bool_ )
    leaky   = ( label.astype( np.float32 ) - pre ) > 0
    over    = ( pre.astype( np.float32 ) - label ) > 0
    #
    # image[:,:,:,0][ over    ] += 1.0 # r
    # image[:,:,:,1][ correct ] += 0.5 # g
    # image[:,:,:,:][ leaky   ] += [-0.2,-0.2,1] # b
    image[ over    ] = [1,0,0] # r
    image[ correct ] = [0,1,0] # g
    image[ leaky   ] = [0,0,1] # b
    #
    image = ( np.clip( image, 0, 1 )* 255 ).astype( np.uint8 )
    # image_1 = ( np.clip( image_1, 0, 1 )* 255 ).astype( np.uint8 )
    # image_2 = ( np.clip( image_2, 0, 1 )* 255 ).astype( np.uint8 )

    image_o_l = np.concatenate( [ ( image_b*255 ).astype( np.uint8 ), image ], 
                                axis=2 
                              )
    #
    for index, image_slice in enumerate( image ):
        if not os.path.exists( save_path + '/merge'):
            os.makedirs( save_path + '/merge' )
            
        if not os.path.exists( save_path + '/o_merge'):
            os.makedirs( save_path + '/o_merge' )

        # if not os.path.exists( save_path + '/pre' ):
        #     os.makedirs( save_path + '/pre' )
         
        # if not os.path.exists( save_path + '/label' ):
        #     os.makedirs( save_path + '/label' )

        #io.imsave( save_path +'/merge'  + f'/{index}.png', image_slice        )
        io.imsave( f'{save_path}/o_merge/{index}-{value}.png', image_o_l[ index ] )

        # io.imsave( save_path + '/pre'   + f'/{index}_pre.png'  , image_1[ index ] )
        # io.imsave( save_path + '/label' + f'/{index}_label.png', image_2[ index ] )

def get_boundary( image, num=1 ):
    
    i = np.copy( image ) 
    for _ in range( num ):
        i = morphology.erosion( i ).astype( np.float32 )
    return image - i

def A2B_distance( A, B ):
    pass

        
# image, pre, label, save_path =  norm255( images[0,0].cpu().numpy() ),a,b,dd_path
def set_pre_and_label_2( image, pre, label, save_path, value='' ):
    dsc_list = []
    nsd_list = []
    hd95_list= []
    hd_list = []
    if image.shape != pre.shape != label.shape :
        raise Exception( '"image.shape != pre.shape != label.shape" is True' )
    thr = 4
    spacing = [1.,1.]
    image = np.repeat( np.expand_dims( image, 3 ), 3, axis=3)
    # image_b = np.copy( image )
    #
    pre   = ( pre   ).astype( np.bool_ )
    label = ( label ).astype( np.bool_ )
    #
    _ = zip( image, pre, label )
    for index, [ slice_image, slice_pre, slice_label ] in enumerate( _ ):
        ###
        p_l_any = [ slice_pre.any(), slice_label.any() ]
        if np.sum( p_l_any ) == 2:
            dsc  = mt.DSC( slice_pre, slice_label )
            # if p_l_any[0] == True:
            #     nsd  = mt.NSD( slice_pre, slice_label, thr, spacing, 1 )
            #     hd = mt.hd_p( slice_pre, slice_label, p=100 )
            # else:
            #     nsd  = 0
            #     hd = 0
            nsd  = mt.NSD( slice_pre, slice_label, thr, spacing, 1 )
            hd = mt.hd_p( slice_pre, slice_label, p=95 )
            #
            dsc_list .append( dsc )
            nsd_list .append( nsd )
            hd_list.append( hd )
            
            loc_pre_b = np.where( get_boundary( slice_pre  , 1 ) )
            loc_lab_b = np.where( get_boundary( slice_label, 1 ) )
            
            # a
            slice_image[ slice_label ] += [ 0.35, 0  , 0 ] # rgb
            slice_image[ slice_pre   ] += [ 0  , 0.25, 0 ] # rgb
            # b
            slice_image[ loc_lab_b ] = [1,0,0] # rgb
            slice_image[ loc_pre_b ] = [0,1,0] # rgb
            # c
            slice_image[ slice_image>1 ] = 1
            
            # if not os.path.exists( save_path ):
            #     os.makedirs( save_path )
            dsc  = int( np.round( dsc, 4 ) * 10000 )
            nsd  = int( np.round( nsd, 4 ) * 10000 )
            hd = int( np.round( hd, 2 ) )
            
            if dsc > 8000 and nsd > 8000:
                folder_path = 'tt'
            elif dsc > 8000 and nsd <= 8000 :
                folder_path = 'td'
            elif dsc <= 8000 and nsd > 8000 :
                folder_path = 'dt'
            else:
                folder_path = 'dd'
                
            if hd < 8:
                folder_path = folder_path + '/d'
            else:
                folder_path = folder_path + '/t'
                
            if not os.path.exists( f'{save_path[0]}/{folder_path}' ):
                os.makedirs( f'{save_path[0]}/{folder_path}' )

            io.imsave( f'{save_path[0]}/{folder_path}/{save_path[1]}-{index}-dsc{dsc}-nsd{nsd}-hd{hd}.png', slice_image )
    
    return dsc_list, nsd_list, hd_list















