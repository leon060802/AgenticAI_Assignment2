python -u run.py \
  --test_file ./data/tasks_test.jsonl \
  --api_key ... \
  --max_iter 20 \
  --max_attached_imgs 3 \
  --temperature 1 \
  --fix_box_color \
  --api_model gpt-4o \
  --seed 42 \
  --error_max_reflection_iter 3 \
  --start_maximized
  # --headless \
