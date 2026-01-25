```bash
nohup python test_6_unet_6_view_with_rasterization.py \
  --checkpoints_path /autofs/space/ballarat_004/users/np341/PRIR_Code/Training/White/Models_WM_SLURM_7625119_worked_5_days \
  --output_suffix _7625119 \
  > test_6_unet_6_view_with_rasterization_7625119_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

