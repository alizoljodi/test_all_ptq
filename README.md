# MQBench PTQ Script with Compatibility Fixes

This repository contains a fixed version of the MQBench PTQ script that handles the common `LearnableFakeQuantize` compatibility issue.

## The Problem

The original script fails with this error:
```
AttributeError: 'LearnableFakeQuantize' object has no attribute 'init'
```

This typically occurs due to version mismatches between:
- MQBench library
- PyTorch version
- Quantizer implementations

## Solutions Provided

### 1. Automatic Fallback
The script now automatically detects compatibility issues and falls back to basic PTQ when advanced PTQ reconstruction fails.

### 2. Command Line Options
- `--no_advanced`: Force disable advanced PTQ reconstruction
- `--diagnose`: Run MQBench setup diagnostics

### 3. Better Error Messages
Clear error messages explaining the issue and suggesting solutions.

## Usage

### Basic PTQ (Recommended for compatibility issues)
```bash
python mq_bench_ptq.py --model resnet18 --no_advanced
```

### Advanced PTQ (if compatible)
```bash
python mq_bench_ptq.py --model resnet18 --advanced
```

### Run Diagnostics
```bash
python mq_bench_ptq.py --diagnose
```

### Test Compatibility
```bash
python test_compatibility.py
```

## Troubleshooting

### If you still get the 'init' method error:

1. **Use basic PTQ only:**
   ```bash
   python mq_bench_ptq.py --model resnet18 --no_advanced
   ```

2. **Update MQBench:**
   ```bash
   pip install --upgrade mqbench
   ```

3. **Check PyTorch version compatibility:**
   - MQBench may require specific PyTorch versions
   - Check MQBench documentation for version requirements

4. **Run diagnostics:**
   ```bash
   python mq_bench_ptq.py --diagnose
   ```

## What the Fixes Do

1. **Graceful Degradation**: When advanced PTQ fails, the script continues with basic PTQ instead of crashing.

2. **Better Error Handling**: Clear error messages explaining what went wrong and how to fix it.

3. **Compatibility Checking**: Pre-flight checks to detect issues before they cause crashes.

4. **Fallback Options**: Multiple ways to run the script even when advanced features aren't available.

## Expected Behavior

- **With compatible setup**: Advanced PTQ works normally
- **With incompatible setup**: Script falls back to basic PTQ with warnings
- **With --no_advanced flag**: Skips advanced PTQ entirely
- **With --diagnose flag**: Shows detailed setup information

## Files

- `mq_bench_ptq.py`: Main PTQ script with compatibility fixes
- `test_compatibility.py`: Standalone compatibility test script
- `README.md`: This documentation

## Notes

- Basic PTQ will still work even when advanced PTQ fails
- Accuracy may be lower without advanced reconstruction techniques
- The script maintains all original functionality when compatible
- All fixes are backward compatible
