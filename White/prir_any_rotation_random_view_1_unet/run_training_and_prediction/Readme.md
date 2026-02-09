Command to run training:
```bash
run_conditioned_v2_random_views.sbatch
```
Command to run testing of 6 predefined view aggregation (Front, Back, Top, Bottom, Left, Right) with model trained on random views:
```bash
nohup python test_conditioned_model_v2.py     --model_dir /autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/prir_x_y_z_conditional_1_unet/models/xyz_random_views_7794326     --test_data /autofs/space/ballarat_004/users/np341/mindboggle2     --device cuda:0     > test_random_model_6views_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Command to run testing of 20 random view aggregation with model trained on random views (Front, Back, Top, Bottom, Left, Right):
```bash
 nohup python test_conditioned_model_v2.py     --model_dir /autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/prir_x_y_z_conditional_1_unet/models/xyz_random_views_7794326     --test_data /autofs/space/ballarat_004/users/np341/mindboggle2     --num_random_views 20     --device cuda:0     > test_random_20views_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```
