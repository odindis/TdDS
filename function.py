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
def clipping( data, cut_range, clipping_loc, background_value=0 ): 
    #
    cut_array = np.zeros( cut_range, dtype=data.dtype )
    if background_value != 0:
        cut_array += background_value
    #
    data_iipoint = np.array([0,0,0])
    data_aapoint= np.array( list( data.shape ) )
    #
    cut_range = np.array( cut_range )
    distance = cut_range // 2
    #
    clipping_loc = np.array( clipping_loc )
    cut_iipoint = clipping_loc - distance
    cut_aapoint = cut_iipoint + cut_range
    #
    assignment_area_iipoint = np.max( [data_iipoint, cut_iipoint], axis=0 )
    assignment_area_aapoint = np.min( [data_aapoint, cut_aapoint], axis=0 )
    #
    assignment_area_data = data[ 
                    assignment_area_iipoint[ 0 ]:assignment_area_aapoint[ 0 ],
                    assignment_area_iipoint[ 1 ]:assignment_area_aapoint[ 1 ],
                    assignment_area_iipoint[ 2 ]:assignment_area_aapoint[ 2 ],
                               ]
    #
    assignment_area_iipoint = assignment_area_iipoint - cut_iipoint
    assignment_area_aapoint = assignment_area_aapoint - cut_iipoint
    #
    cut_array[ assignment_area_iipoint[0]:assignment_area_aapoint[0],
               assignment_area_iipoint[1]:assignment_area_aapoint[1],
               assignment_area_iipoint[2]:assignment_area_aapoint[2],
             ] = assignment_area_data
    #
    return cut_array


#==============================================================================
def get_trianing_data( files, patch_shape ):
    images = []
    labels = []
    for file in files:
        info = np.load( file, allow_pickle=True )
        image = info['array']
        label = info['label']
        #--
        if np.random.randint(0,3): # Get ROI min area 
            cut_label_sum = np.inf
            label_sum     = np.sum( label )
            while cut_label_sum / label_sum > 0.4:
                shape = label.shape
                clipping_loc = [ np.random.randint(0,shape[0]),
                                 np.random.randint(0,shape[1]),
                                 np.random.randint(0,shape[2]),
                               ]
                # print( label.shape, patch_shape, clipping_loc )
                cut_label = clipping( label, patch_shape, clipping_loc )
                cut_label_sum = np.sum( cut_label )
            cut_image = clipping( image, patch_shape, clipping_loc, image.min() )
        else: # Get ROI min area
            cut_label_sum = 0
            label_sum     = np.sum( label )
            while cut_label_sum / label_sum < 0.8:
                shape = label.shape
                clipping_loc = [ np.random.randint(0,shape[0]),
                                 np.random.randint(0,shape[1]),
                                 np.random.randint(0,shape[2]),
                               ]
                # print( label.shape, patch_shape, clipping_loc )
                cut_label = clipping( label, patch_shape, clipping_loc )
                cut_label_sum = np.sum( cut_label )
            cut_image = clipping( image, patch_shape, clipping_loc, image.min() )
        #--
        images.append( cut_image )
        labels.append( cut_label )
    #
    return np.array( images )[:,np.newaxis], np.array( labels )[:,np.newaxis]

def normalvariate_loc( num ):
    return int( np.clip( random.normalvariate(0.5,0.25), 0, 1 ) * num + 0.5 )

def get_unlabel_data( files, patch_shape ):
    images = []
    for file in files:
        info = np.load( file, allow_pickle=True )
        image = info['array']
        #--
        shape = image.shape
        clipping_loc = [ normalvariate_loc( shape[0] ),
                         normalvariate_loc( shape[1] ),
                         normalvariate_loc( shape[2] ),
                       ]
        cut_image = clipping( image, patch_shape, clipping_loc )
        images.append( cut_image )
    #
    return np.array( images )[:,np.newaxis]

#==============================================================================
def shape_supplement( data, shape ):
    temp = np.zeros( shape, dtype=data.dtype )
    temp[ :data.shape[0],:data.shape[1],:data.shape[2] ] = data
    return temp

#==============================================================================
def get_testing_data( file, patch_shape, overlap=0.51 ):
    ##
    info = np.load( file, allow_pickle=True )
    image = info['array']
    label = info['label']
    
    # Patch may be better than image_ Shap large
    image_shape = np.array( image.shape )
    if ( image_shape < patch_shape ).any():
        new_shape = np.max( [ image_shape, patch_shape ], axis=0 )
        image = shape_supplement( image, new_shape )
        label = shape_supplement( label, new_shape )
    #
    data_shape = np.array( image.shape )
    block_num = ( data_shape / patch_shape )
    block_num = np.round( block_num ).astype( np.int32 )
    interval = ( data_shape / block_num )
    ##
    ban_set = np.array( [ True for _ in patch_shape ] )
    ol = interval/patch_shape
    while ( ( ol > overlap) * ( patch_shape<data_shape ) * ban_set ).any() :
        ol = interval / patch_shape
        block_num[ ( ol > overlap) * (patch_shape<data_shape) ] += 1
        interval = ( data_shape / block_num )
        ol = interval / patch_shape
        ban_loc = ol <= 0.5
        if ban_loc.any():
            block_num[ ban_loc ] -= 1
            ban_set[ ban_loc ] = False
    ##
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
    #
    image = image[ np.newaxis, np.newaxis ]
    label = label[ np.newaxis, np.newaxis ]
    
    return image, label, div_loc, image_shape

#==============================================================================
def get_testing_result( image, model, div_loc, patch_shape ):
    #
    shape = list( image.shape )
    shape[1] = 2
    output = torch.zeros( shape, dtype=image.dtype, device=image.device )
    #
    areas_set = []
    for loc_1 in div_loc[0]:
        for loc_2 in div_loc[1]:
            for loc_3 in div_loc[2]:
                areas_set.append( [ loc_1, loc_1 + patch_shape[0],
                                    loc_2, loc_2 + patch_shape[1],
                                    loc_3, loc_3 + patch_shape[2],
                                  ] 
                                )
        
    #
    areas_num = len( areas_set )
    #print()
    for num, area in enumerate( areas_set ):
        #sys.stdout.write( f'\r> areas: {num}/{areas_num}     ')
        image_part = image[ :,:,
                            area[0] : area[1], 
                            area[2] : area[3], 
                            area[4] : area[5] 
                          ]
        output[ :,:,
                area[0] : area[1], 
                area[2] : area[3], 
                area[4] : area[5] 
              ]  += model( image_part )
    #
    return output

#==============================================================================
#from model_2 import scaling_target
from itertools import product

def map_combination( image, model, div_loc, patch_shape ):
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
        o = model( image_part )[0]
        # o = scaling_target( model( image_part )[1], image_part )
        # o = scaling_target( model( image_part )[2], image_part )
        # o = scaling_target( model( image_part )[3], image_part )
        # o = scaling_target( model( image_part )[4], image_part )
        
        output[ :,:,
                area[0] : area[1], 
                area[2] : area[3], 
                area[4] : area[5] 
              ]  += o
    #
    return output

#def get_testing_result_3( image, model, div_loc, patch_shape ):
#    #
#    shape = list( image.shape )
#    shape[1] = 2
#    output = torch.zeros( shape, dtype=image.dtype, device=image.device )
#    output2 = torch.zeros( shape, dtype=image.dtype, device=image.device )
#
#    #
#    areas_set = []
#    for loc_1, loc_2, loc_3 in product( div_loc[0], div_loc[1], div_loc[2] ):
#        areas_set.append( [ loc_1, loc_1 + patch_shape[0],
#                            loc_2, loc_2 + patch_shape[1],
#                            loc_3, loc_3 + patch_shape[2],
#                          ]
#                        )
#    #print()
#    for num, area in enumerate( areas_set ):
#        #sys.stdout.write( f'\r> areas: {num}/{areas_num}     ')
#        image_part = image[ :,:,
#                            area[0] : area[1], 
#                            area[2] : area[3], 
#                            area[4] : area[5] 
#                          ]
#        o0 = scaling_target( model( image_part )[0], image_part )
#        o4 = scaling_target( model( image_part )[4], image_part )
#        
#        output[ :,:,
#                area[0] : area[1], 
#                area[2] : area[3], 
#                area[4] : area[5] 
#              ]  += o0
#        
#        output2[ :,:,
#                area[0] : area[1], 
#                area[2] : area[3], 
#                area[4] : area[5] 
#              ]  += o4
#        
#    #
#    return output, output2

#from einops import rearrange
#
#def get_testing_result_4( image, model, div_loc, patch_shape ):
#    #
#    shape = list( image.shape )
#    shape[1] = 2
#    output = torch.zeros( shape, dtype=image.dtype, device=image.device )
#    #
#    areas_set = []
#    for loc_1 in div_loc[0]:
#        for loc_2 in div_loc[1]:
#            for loc_3 in div_loc[2]:
#                areas_set.append( [ loc_1, loc_1 + patch_shape[0],
#                                    loc_2, loc_2 + patch_shape[1],
#                                    loc_3, loc_3 + patch_shape[2],
#                                  ] 
#                                )
#        
#    #
#    #print()
#    for num, area in enumerate( areas_set ):
#        #sys.stdout.write( f'\r> areas: {num}/{areas_num}     ')
#        image_part = image[ :,:,
#                            area[0] : area[1], 
#                            area[2] : area[3], 
#                            area[4] : area[5] 
#                          ]
#        
#        b, c, n, h, w = image_part.shape
#        image_part = rearrange( image_part, 'b c n h w -> (b n) c h w' )
#        image_part = model( image_part )
#        image_part = rearrange( image_part, 
#                                '(b n) c h w -> b c n h w',
#                                b=b, 
#                                n=n 
#                              )
#        output[ :,:,
#                area[0] : area[1], 
#                area[2] : area[3], 
#                area[4] : area[5] 
#              ]  += image_part
#    #
#    return output


#==============================================================================
def cut_data_base_center( data, cut_range ): 
    shape_max = np.array( [ data.shape, cut_range] ).max(axis=0)
    temp_tensor = np.zeros( shape_max, dtype=data.dtype ) + data.min()
    #
    loc_middle   = ( np.array( temp_tensor.shape ) /2. +0.5 ).astype( np.int16 )
    range_tensor = ( np.array( data.shape        ) /2. +0.5 ).astype( np.int16 )
    range_cut    = ( np.array( cut_range         ) /2. +0.5 ).astype( np.int16 )
    #
    starting_point = loc_middle - range_tensor
    end_point      = np.array( data.shape ) + starting_point
    
    temp_tensor[ starting_point[0] : end_point[0], 
                 starting_point[1] : end_point[1], 
                 starting_point[2] : end_point[2],
               ] = data

    return temp_tensor[ loc_middle[0]-range_cut[0] : loc_middle[0]+range_cut[0], 
                        loc_middle[1]-range_cut[1] : loc_middle[1]+range_cut[1], 
                        loc_middle[2]-range_cut[2] : loc_middle[2]+range_cut[2],
                      ] 

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

## ============================================================================
def get_batch_patch_2d( images, labels, patch_shape ):
    images_list = []
    labels_list = []
    for image, label in zip( images, labels ):
        ## Fill: patch may be better than image_ Shape large
        if ( image.shape < patch_shape ).any():
            new_shape = np.max( [ image.shape, patch_shape ], axis=0 )
            image = fill_embedding( image, new_shape )
            label = fill_embedding( label, new_shape )
        ## get roi slicers
        index_roi_slicers = list( set( np.where( label )[0].tolist() ) )
        image = image[ index_roi_slicers ]
        label = label[ index_roi_slicers ]
        
        ## Cut loc
        shape = label.shape[1:] - patch_shape[1:]
        clip_start_loc = [ rdt( -1, shape[0] )+1,
                           rdt( -1, shape[1] )+1, 
                         ]
        clip_end_loc = clip_start_loc + patch_shape[1:]
        ## Cut        
        cut_label = label[ :, 
                           clip_start_loc[0]:clip_end_loc[0],
                           clip_start_loc[1]:clip_end_loc[1],
                         ]
        cut_image = image[ :,
                           clip_start_loc[0]:clip_end_loc[0],
                           clip_start_loc[1]:clip_end_loc[1],
                         ]

        ## Load data into list
        for image_s, label_s in zip( cut_image, cut_label ):
            images_list.append( image_s )
            labels_list.append( label_s )
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


def batch_crop_test_data( images_list, labels_list, patch_shape, no_overlap=1/2 ):
    crop_images_list = []
    crop_labels_list = []
    div_loc_list = []
    for image, label in zip( images_list, labels_list ):
        image_, label_, div_loc = crop_test_data( image, 
                                                  label, 
                                                  patch_shape, 
                                                  no_overlap=1/2
                                                )
        crop_images_list.append( image_  )
        crop_labels_list.append( label_  )
        div_loc_list    .append( div_loc )
    return crop_images_list, crop_labels_list, div_loc_list


def spilt_train_vali( train_path, rate=0.1 ):
    ID_dic = {}
    for path_ in train_path:
        ID = path_.split('/')[-3]
        if ID not in ID_dic:
            ID_dic[ ID ] = []
        ID_dic[ ID ].append( path_ )
        
    ID_list = list( ID_dic.keys() )
    np.random.shuffle( ID_list )
    
    val_num = int( len( ID_list ) * rate )
    
    vali_ID_list  = ID_list[ :val_num ]
    train_ID_list = ID_list[ val_num: ]
    
    vali_path_list = []
    for vali_ID in vali_ID_list:
        vali_path_list += ID_dic[ vali_ID ]
    
    train_path_list = []
    for train_ID in train_ID_list:
        train_path_list += ID_dic[ train_ID ]

    return train_path_list, vali_path_list

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
# def set_pre_and_label_2( image, pre, label, save_path, value='' ):
#     if image.shape != pre.shape != label.shape :
#         raise Exception( '"image.shape != pre.shape != label.shape" is True' )
#     thr = 1
#     spacing = [1.,1.]
#     image = np.repeat( np.expand_dims( image, 3 ), 3, axis=3)
#     image_b = np.copy( image )
#     #
#     pre   = ( pre   ).astype( np.bool_ )
#     label = ( label ).astype( np.bool_ )
#     #
#     _ = zip( image, pre, label )
#     for index, [ slice_image, slice_pre, slice_label ] in enumerate( _ ):
#         ###
#         p_l_any = [ slice_pre.any(), slice_label.any() ]
#         ###
#         if p_l_any[0]: 
#             slice_pre_b   = np.where( get_boundary( slice_pre   ) )
#         if p_l_any[1]:
#             slice_label_b = np.where( get_boundary( slice_label ) )
#         ###
#         if sum( p_l_any ) == 0: 
#             nsd = 1
#             dsc = 1
#         elif sum( p_l_any ) == 1:
#             nsd = 0
#             dsc = 0
#             if p_l_any[0]:
#                 slice_image[ np.where( slice_pre_b   ) ] = [1,0,0] # rgb
#             elif p_l_any[1]:
#                 slice_image[ np.where( slice_label_b ) ] = [0,0,1] # rgb
#             else:
#                 raise Exception()
            
#         elif sum( p_l_any ) == 2:
            
#             nsd, [p2l, l2p] = mt.NSD( slice_pre, slice_label, thr, spacing, 1, True )
#             dsc = mt.DSC( slice_pre, slice_label )
#             ## draw
#             # pre
#             slice_image[ ( slice_pre_b[0][ p2l> thr ], slice_pre_b[1][ p2l> thr ] ) ] = [1,0,0] # rgb
#             slice_image[ ( slice_pre_b[0][ p2l<=thr ], slice_pre_b[1][ p2l<=thr ] ) ] = [0,1,0] # rgb
#             # label
#             slice_image[ ( slice_label_b[0][ l2p> thr ], slice_label_b[1][ l2p> thr ] ) ] = [0,0,1] # rgb
#             slice_image[ ( slice_label_b[0][ l2p<=thr ], slice_label_b[1][ l2p<=thr ] ) ] = [0,1,0] # rgb
            
        
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















