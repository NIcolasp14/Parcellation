Command to run training:
```bash
sbatch run_conditioned_v2_20views.sbatch
```
Command to run testing:
```bash
nohup ./run_test_20views.sh > test_20views_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```
