Run AWQ models using use_llm_class.py to generates python code:
    python3 /mnt/d/Graduation_Thesis/Vebot-pro/use_llm_class.py --use-gpu > /mnt/d/Graduation_Thesis/Vebot-pro/results/iteration_*.txt 

Run tests for python code llm generates:
     python Vebot-pro/scripts/validate_results.py  --write-reports

Run AWQ models using use_rcl_only.py to generates rcl code:
    python Vebot-pro/scripts/use_rcl_only.py --use-gpu > /mnt/d/Graduation_Thesis/Vebot-pro/results/rcl_results/iteration_*.txt 

Run validation for python code:
    python Vebot-pro/scripts/validate_results.py --write-reports

Run validation for rc code:
    python Vebot-pro/scripts/validate_results.py --rcl --write-reports