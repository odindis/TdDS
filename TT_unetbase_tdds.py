## Import module ==============================================================
from data_loader import data_loader_io

import function as fct
import metric as mt
import numpy as np
import model as M
import torch
import time
import sys
import os

## Parameters =================================================================
## Two test samples were provided
train_data_path = None
test_data_path  = 'data'

## 
is_train    = False
valid_rate  = 0.1
random_seed = 1
batch_num   = 10
patch_shape = [16,256,256]
iter_num    = 1000000000
epoch_num   = 10
early_stop  = 150

model_save_path = 'unet_res2_ds_tdds.pkl'
pred_model_path = None 



## Data methon ================================================================
train_data = np.load( train_data_path )
valid_num = int( len( train_data ) * valid_rate )

np.random.seed( random_seed )
np.random.shuffle( train_data )
np.random.seed( None )
valid_data = train_data[:valid_num]
train_data = train_data[valid_num:]
test_data = glob.glob( test_data_path + /* )

train_data_loader = data_loader_io( train_data, batch_num )
valid_data_loader = data_loader_io( valid_data, 1 )
test_data_loader  = data_loader_io( test_data , 1 )

if epoch_num == None:
    epoch_num = train_data_loader.cycle -1 
    epoch_num = epoch_num if epoch_num > 0 else 1

for _ in valid_data_loader.data_path_set:
    if _ in train_data_loader.data_path_set:
        raise Exception('valid_data in train_data ')

for _ in test_data_loader.data_path_set:
    if _ in train_data_loader.data_path_set:
        raise Exception('test_data in train_data')

print('--------------------------------')
print( f'train_data_num:{len(train_data_loader.data_path_set)}, batch:{batch_num}, cycle:{train_data_loader.cycle}')
print( f'valid_data_num:{len(valid_data_loader.data_path_set)}, batch:1, cycle:{valid_data_loader.cycle}' )
print( f'test__data_num:{len( test_data_loader.data_path_set)}, batch:1, cycle:{test_data_loader.cycle}'  )
print( f'model_save_path:{model_save_path}' )
print('--------------------------------')

def dice_torch( A, B, s=1e-4 ):
    C = torch.sum( A * B ) + s
    D = torch.sum( A ) + torch.sum( B ) + s
    return 2 * ( C ) / ( D )


## Get gpu ====================================================================
GPU_use = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_use
print( '> Use GPU',GPU_use )

## Model ======================================================================
model = M.UNet_res2_ds( 1, 2 ).cuda()
optimizer = torch.optim.Adam( model.parameters(), lr=1e-4 )

## Traning ====================================================================
patch_shape = np.array( patch_shape )
if is_train:
    if pred_model_path != None:
        print('> Import pred_model:', pred_model_path )
        model.load_state_dict( torch.load( pred_model_path ) )
    train_loss_mean = 1
    focus_roi_rate  = 1
    train_loss_sum  = 0
    r_test_loss     = 99
    epoch = 0
    es = 0
    for iter_ in range( 1, iter_num+1 ):
        ##
        i = iter_ % epoch_num if iter_ % epoch_num else epoch_num
        sys.stdout.write(f'\r> training : {i}/{epoch_num}       ')
        ## Load data
        images, labels = train_data_loader()
        images, labels = fct.batch_crop_data( images, 
                                              labels, 
                                              patch_shape, 
                                              is_focus       = True,
                                              focus_roi_rate = 1/3 ,
                                              target_r       = 0.60,
                                              n_target_r     = 0.40,
                                              num_max_find   = 5,
                                           )
        ## Data Augmentation
        images = torch.tensor( images, dtype=torch.float ).cuda().detach()
        labels = torch.tensor( labels, dtype=torch.float ).cuda().detach()
        ## Into model
        optimizer.zero_grad()
        outputs = model( images )
        ## Get loss
        loss = M.calculate_loss_2( outputs, labels )
        ## Optimizer
        loss.backward()
        optimizer.step()
        # torch.cuda.empty_cache()
        ## Statistical loss
        train_loss_sum += loss.detach().cpu().numpy()
        
        ## Check point ========================================================
        if iter_ % epoch_num == 0 :
            epoch += 1
            ## Validation
            model.eval()
            with torch.no_grad():
                valid_loss_sum = 0
                for _ in range( valid_data_loader.cycle ):
                    sys.stdout.write( f'\r> valid: {_+1}/{valid_data_loader.cycle}'
                                      f'                                          '
                                    )
                    ## Load data
                    images, labels = valid_data_loader()
                    images, labels, div_loc = fct.crop_test_data( images[0], 
                                                              labels[0], 
                                                              patch_shape,
                                                              1, 
                                                            )
                    ## Into model
                    images = torch.tensor( images, dtype=torch.float ).cuda()
                    output = M.map_combination( images, model, div_loc, patch_shape )
                    output[:1] = torch.argmax( output, dim=1 ).unsqueeze(0)
                    # io.imshow( output[0,1].cpu().numpy()[10] )
                    ## Get loss
                    labels = torch.tensor( labels ).cuda()
                    loss = M.dice_loss( output[0,1], labels[0,0] )
                    ## Statistical loss
                    valid_loss_sum += loss.detach().cpu().numpy()
                valid_loss_mean = valid_loss_sum / ( valid_data_loader.cycle )
                
            ## Saveing mode
            if valid_loss_mean < r_test_loss:
                torch.save( model.state_dict(), model_save_path )
                r_test_loss = valid_loss_mean
                es = 0
                
            if es > early_stop:
                print()
                print('> early_stop')
                break
            
            ## Print info
            train_loss_mean = train_loss_sum / epoch_num
            train_loss_sum = 0
            localtime = time.strftime( "%m-%d-%H:%M:%S", time.localtime() )
            sys.stdout.write( f'\r> epoch:{epoch} | loss_train_valid:[{train_loss_mean:.4f},{valid_loss_mean:.4f}] |{es}/{early_stop}| {localtime} |        ')
            print()
    print( f'> best_test_loss : {r_test_loss:.4f}' )

## Testing ====================================================================
model.load_state_dict( torch.load( model_save_path ) )
model.eval()
with torch.no_grad():
    test_loss_sum = 0
    dice_list = []
    for iter_ in range( test_data_loader.cycle ):
        sys.stdout.write( f'\r> testing : {iter_+1}/{test_data_loader.cycle}'
                          f'                                         '
                        )        
        ## Load data
        file_name = os.path.basename( 
                        os.path.splitext( 
                            test_data_loader.get_batch_path()[0] 
                                        )[0] 
                                    )
        images, labels = test_data_loader()
        images, labels, div_loc = fct.crop_test_data( images[0], 
                                                  labels[0], 
                                                  patch_shape, 
                                                  1/2, 
                                                )
        ## Into model
        images = torch.tensor( images, dtype=torch.float ).cuda()
        output = M.map_combination( images, model, div_loc, patch_shape, 0 )
        
        # raise Exception()
        
        output[:1]  = torch.argmax( output , dim=1 ).unsqueeze(0)

        ## Get Measurement 
        labels = torch.tensor( labels ).cuda()
        # Dice
        dice_list.append( float( dice_torch( output[0,1], labels[0,0] ) ) )

        


    # ## Showing
    # print()
    print('---Result--------------------------')
    print( f'> Dice     : {np.mean( dice_list      ):.4f} & {np.std( dice_list      ):.4f}\n' )


