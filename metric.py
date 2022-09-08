from medpy.metric.binary import __surface_distances
import medpy.metric as mmt

import numpy as np

## Normalized_surface_dice
def NSD( array_a, array_b, thr, spacing=(1,1,1), connectivity=1, return_=False ):
    a2b = __surface_distances( array_a, array_b, spacing, connectivity )
    b2a = __surface_distances( array_b, array_a, spacing, connectivity )
 
    tp_a = np.sum( a2b <= thr )
    tp_b = np.sum( b2a <= thr )
    
    nsd = ( tp_a + tp_b ) / (  len( a2b ) + len( b2a ) + 1e-4 ) 
    if return_:
        return nsd, [a2b,b2a]
    else:
        return nsd

def DSC( A, B ):
    return ( 2 * np.sum( A * B ) ) / ( np.sum(A) + np.sum(B) +1e-4 )

def slice_NSD( pre, label, thr, spacing, connectivity=1 ):
    nsd_list = []
    for index, [slice_pre, slice_label] in enumerate( zip( pre, label ) ):
        p_l_any = [ slice_pre.any(), slice_label.any() ]
        if sum( p_l_any ) == 0:
            # nsd_list.append( 1 )
            pass
        elif sum( p_l_any ) == 1:
            nsd_list.append( 0 )
            # pass
        elif sum( p_l_any ) == 2:
            nsd = NSD( slice_pre, slice_label, thr, spacing[1:], connectivity=1 )
            nsd_list.append( nsd )
    return nsd_list


def slice_dice( pre, label ):
    dice_list = []
    for index, [ slice_pre, slice_label ] in enumerate( zip( pre, label ) ):
        p_l_any = [ slice_pre.any(), slice_label.any() ]
        if sum( p_l_any ) == 0:
            # dice_list.append( 1 )
            pass
        elif sum( p_l_any ) == 1:
            # dice_list.append( 0 )
            pass
        elif sum( p_l_any ) == 2:
            dc = DSC( slice_pre, slice_label )
            dice_list.append( dc )
    return dice_list



def hd_p( array_a, array_b, p=100, spacing=(1,1), connectivity=1  ):
    a2b = __surface_distances( array_a, array_b, spacing, connectivity )
    b2a = __surface_distances( array_b, array_a, spacing, connectivity )
    return np.percentile( a2b.tolist() + b2a.tolist(), p )


def slice_s( pre, label, thr, spacing, connectivity=1 ):
    dsc_list = []
    nsd_list = []
    hd_list  = []
    hd95_list = []
    
    for index, [ slice_pre, slice_label ] in enumerate( zip( pre, label ) ):
        p_l_any = [ slice_pre.any(), slice_label.any() ]
        if sum( p_l_any ) == 0:
            # dice_list.append( 1 )
            pass
        elif sum( p_l_any ) == 1:
            # dice_list.append( 0 )
            pass
        elif sum( p_l_any ) == 2:
            dc   = DSC ( slice_pre, slice_label )
            nsd  = NSD ( slice_pre, slice_label, thr, spacing )
            hd   = hd_p( slice_pre, slice_label, spacing, 90 )
            hd95 = hd_p( slice_pre, slice_label, spacing, 85 )
            
            #
            dsc_list.append( dc  )
            nsd_list.append( nsd )
            hd_list .append( hd  )
            hd95_list.append( hd95 )
            
            
    return [ dsc_list, nsd_list, hd_list, hd95_list ]

# index = 2
# io.imshow( pre[index] + label[index]*2 )

# A = np.zeros( [262,401])
# B = np.zeros( [262,401])

# A[2:6,2:6] = 1
# B[3:8,3:8] = 1

# C = A+B*2

# # io.imshow( C )




# a_to_b = __surface_distances( A, B, (1.,1.) )
# b_to_a = __surface_distances( B, A, (1.,1.) )



# numel_a = len(a_to_b)
# numel_b = len(b_to_a)
 
# tp_a = np.sum(a_to_b <= 1) / numel_a
# tp_b = np.sum(b_to_a <= 1) / numel_b
 
# fp = np.sum(a_to_b > 1) / numel_a
# fn = np.sum(b_to_a > 1) / numel_b
 
# dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + 1e-8)  # 1e-8 just so that we don't get div by 0

# tp_a = np.sum( a_to_b <= 1 )
# tp_b = np.sum( b_to_a <= 1 )
 
# fp = np.sum( a_to_b > 1 )
# fn = np.sum( b_to_a > 1 )


# dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + 1e-8)


