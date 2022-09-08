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










