import torch
from torch.utils.data import DataLoader
from tlod.data_loader.dynamic_replica_dataset import DynamicReplicaDataset
from tlod.data_loader.mvaria_dataset import AriaDataset

def test_dynamic_replica_data_loader():
    batch_size = 6

    # create dataset
    dataset = DynamicReplicaDataset(data_root=".\\data\\dynamicreplica")
    # create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # check if data loader works
    for batch in data_loader:
        assert batch is not None, "Batch is None"
        print (batch.keys()) # 'rgb_input', 'rays_t_un_input', 'cameras_input', 'rgb_output', 'rays_t_un_output', 'cameras_output', 'img_name_output', 'ratios_output', 'c2w_avg'
        print (batch['rgb_input'].shape)  # [9, 8, 3, 256, 256]
        print (batch['rays_t_un_input'].shape)  # [9, 8]
        print (batch['cameras_input'].shape)  # [9, 8, 20]
        print (batch['rgb_output'].shape)  # [9, 8, 3, 256, 256]
        print (batch['rays_t_un_output'].shape)  # [9, 8]
        print (batch['cameras_output'].shape) # [9, 8, 20]
        #print (batch['img_name_output']) 
        assert batch['rgb_input'].shape[0] == batch_size, "Batch size should be 8"
        assert batch['rays_t_un_input'].shape[0] == batch_size, "Batch size should be 8"
        assert batch['cameras_input'].shape[0] == batch_size, "Batch size should be 8"
        assert batch['rgb_output'].shape[0] == batch_size, "Batch size should be 8"
        assert batch['rays_t_un_output'].shape[0] == batch_size, "Batch size should be 8"
        assert batch['cameras_output'].shape[0] == batch_size, "Batch size should be 8"

def test_mvaria_data_loader():
    # use real data from C:\Development\AI\CV2\dynamic-novel-view-synthesis\data\mvaria\test
    # create dataset
    batch_size = 8
    #image_num_per_batch = 128 # default is 8 
    #output_image_num = 128 # default is 8 
    #input_image_res = 504  # default is 256
    #output_image_res = 504  # default is 256

    dataset = AriaDataset(data_root=".\\data\\aea", seq_list="loc3_script3_seq1_rec1", 
                        seq_data_roots=["recording/camera-rgb-rectified-600-h1000"],
                        #image_num_per_batch=image_num_per_batch, output_image_num=output_image_num
                        #input_image_res=input_image_res, output_image_res=output_image_res
    )
    # create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # the batch-size doesn't matter here
    # check if data loader works
    for batch in data_loader:
        assert batch is not None, "Batch is None"
        print (batch.keys()) # 'rgb_input', 'rays_t_un_input', 'cameras_input', 'rgb_output', 'rays_t_un_output', 'cameras_output', 'img_name_output', 'ratios_output', 'c2w_avg'
        print (batch['rgb_input'].shape)  # [9, 8, 3, 256, 256]
        print (batch['rays_t_un_input'].shape)  # [9, 8]
        print (batch['cameras_input'].shape)  # [9, 8, 20]
        print (batch['rgb_output'].shape)  # [9, 8, 3, 256, 256]
        print (batch['rays_t_un_output'].shape)  # [9, 8]
        print (batch['cameras_output'].shape) # [9, 8, 20]
        #print (batch['img_name_output']) 
        assert batch['rgb_input'].shape[0] == batch_size, "Batch size should be 8"
        assert batch['rays_t_un_input'].shape[0] == batch_size, "Batch size should be 8"
        assert batch['cameras_input'].shape[0] == batch_size, "Batch size should be 8"
        assert batch['rgb_output'].shape[0] == batch_size, "Batch size should be 8"
        assert batch['rays_t_un_output'].shape[0] == batch_size, "Batch size should be 8"
        assert batch['cameras_output'].shape[0] == batch_size, "Batch size should be 8"


        break  # only test one batch


if __name__ == "__main__":
    test_dynamic_replica_data_loader()
    test_mvaria_data_loader()
    print("Everything passed")