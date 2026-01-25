Command to run training:
```bash
sbatch run_conditioned_v2_1view.sbatch
```
Command to run testing:
```bash
nohup python /autofs/space/ballarat_004/users/np341/run_training_and_prediction/test_conditioned_model_v2.py \
  --model_dir /autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/Models_WM_Conditioned_V2_1view_7711474 \
  --test_data /autofs/space/ballarat_004/users/np341/mindboggle2 \
  --device cuda:0 \
  > test_conditioned_model_v2_7711474.nohup.log 2>&1 &
<img width="468" height="146" alt="image" src="https://github.com/user-attachments/assets/e53b4f33-6695-4424-b390-d9493a4695d4" />
```
