# Manually Re-run Baseline for a Task

When a task shows >100% progress (best TFLOPS exceeds baseline), the baseline measurement may have been affected by GPU contention or other factors. This document explains how to manually re-run the baseline and update task records.

## Prerequisites

1. Ensure no other GPU-intensive processes are running
2. Check GPU state: `nvidia-smi`
3. The monitor backend should be stopped during baseline re-measurement to avoid race conditions

## Step 1: Identify the Task

Check task details in the database:

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('monitor/data/monitor.db')
cur = conn.cursor()
cur.execute('''
SELECT id, task_uid, shape_key, best_tflops, baseline_tflops,
       ROUND(best_tflops / baseline_tflops * 100, 1) as progress_pct
FROM tasks WHERE id = <TASK_ID>
''')
print(cur.fetchone())
"
```

Extract the shape_key components:
- `sm90_NVIDIA_H800_PCIe/croqtile/<shape_key>/<model>`
- Parse shape_key to get `<op>_<dtype>_<M>x<N>x<K>`

## Step 2: Re-run cuBLAS Baseline

### Option A: Using the harness script (recommended)

The `store_baseline.sh` script will **overwrite** the existing baseline if you delete the result file first:

```bash
# 1. Delete existing baseline artifact (forces re-measurement)
rm -f tuning/<gpu>/<dsl>/baseline/<shape_key>/<model>/cublas_result.json

# 2. Delete iter000 from results.tsv (will be re-added)
sed -i '/^iter000/d' tuning/<gpu>/<dsl>/logs/<shape_key>/<model>/results.tsv

# 3. Re-run baseline measurement
bash .claude/skills/croq-tune/tools/store_baseline.sh \
    --dsl <dsl> \
    --shape-key <shape_key> \
    --model <model> \
    --dtype <dtype> \
    --m <M> --n <N> --k <K> \
    --task-uid <task_uid>
```

### Option B: Direct measurement (for debugging)

```bash
bash .claude/skills/croq-tune/tools/cublas_baseline.sh \
    --dtype <dtype> \
    --m <M> --n <N> --k <K> \
    --warmup 10 --iters 50
```

This outputs JSON with the TFLOPS value but doesn't update any files.

## Step 3: Update Database

After re-measuring, update the task's baseline_tflops in the database:

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('monitor/data/monitor.db')
cur = conn.cursor()

NEW_BASELINE = <new_tflops_value>
TASK_ID = <task_id>

cur.execute('UPDATE tasks SET baseline_tflops = ? WHERE id = ?', (NEW_BASELINE, TASK_ID))
conn.commit()

# Verify
cur.execute('SELECT id, baseline_tflops, best_tflops FROM tasks WHERE id = ?', (TASK_ID,))
print('Updated:', cur.fetchone())
"
```

## Step 4: Update Disk Artifacts

Update the baseline result JSON:

```bash
python3 -c "
import json
RESULT_FILE = 'tuning/<gpu>/<dsl>/baseline/<shape_key>/<model>/cublas_result.json'
with open(RESULT_FILE, 'r') as f:
    data = json.load(f)
data['tflops'] = <new_tflops_value>
with open(RESULT_FILE, 'w') as f:
    json.dump(data, f, indent=2)
print('Updated:', RESULT_FILE)
"
```

Update results.tsv iter000 row:

```bash
# Edit tuning/<gpu>/<dsl>/logs/<shape_key>/<model>/results.tsv
# Change the TFLOPS value in the iter000 row
```

## Full Example

Re-running baseline for a task (example: `matmul_fp16fp32_512x16384x16384`):

```bash
# Extract parameters from the task
DSL=croqtile
GPU=sm90_NVIDIA_H800_PCIe
SHAPE_KEY=matmul_fp16fp32_512x16384x16384
MODEL=claude-4-5-opus-high
DTYPE=fp16fp32
M=512
N=16384
K=16384
TASK_UID=<get_from_database>

# Delete existing baseline to force re-measurement
rm -f tuning/${GPU}/${DSL}/baseline/${SHAPE_KEY}/${MODEL}/cublas_result.json

# Re-run baseline
bash .claude/skills/croq-tune/tools/store_baseline.sh \
    --dsl $DSL \
    --shape-key $SHAPE_KEY \
    --model $MODEL \
    --dtype $DTYPE \
    --m $M --n $N --k $K \
    --task-uid $TASK_UID

# The script outputs the new TFLOPS. Use that value to update the database:
python3 -c "
import sqlite3
conn = sqlite3.connect('monitor/data/monitor.db')
cur = conn.cursor()
NEW_BASELINE = <value_from_script_output>
cur.execute('UPDATE tasks SET baseline_tflops = ? WHERE id = 3', (NEW_BASELINE,))
conn.commit()
"
```

## Troubleshooting

### GPU Contention

Before measuring, ensure the GPU is idle:

```bash
bash .claude/skills/croq-tune/tools/gpu_check.sh
# If busy, wait:
bash .claude/skills/croq-tune/tools/gpu_check.sh --wait --timeout 120
```

### Inconsistent Results

Run multiple measurements and take the median:

```bash
for i in {1..5}; do
  bash .claude/skills/croq-tune/tools/cublas_baseline.sh \
      --dtype fp16fp32 --m 512 --n 16384 --k 16384 \
      --warmup 10 --iters 50
done
```

### TF32 Effects

The baseline script disables TF32:
```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

This ensures fair comparison with custom kernels that don't use TF32.
