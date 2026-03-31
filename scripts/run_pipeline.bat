echo Running full pipeline...

python scripts\01_convert_to_csv.py
python scripts\02_clean_parallel.py
python scripts\03_filter_parallel.py
python scripts\04_evaluate_translation.py

echo Done!
pause