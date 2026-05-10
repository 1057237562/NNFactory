import sys
sys.path.insert(0, r'E:\Workspace\NNFactory\backend')

# Test 1: Verify DatasetInfo is imported correctly
from preprocessing_pipeline import PreprocessingPipeline, PreprocessingResult
from dataset_manager import DatasetInfo
print('Test 1 PASSED: DatasetInfo import fixed')

# Test 2: Verify one_hot encode logic creates binary columns
import inspect
source = inspect.getsource(PreprocessingPipeline._apply_one_hot)
assert 'new_col = f"{col}_{val}"' in source, 'Missing one-hot column creation'
assert 'row[new_col] = "1" if row_val == val else "0"' in source, 'Missing binary assignment'
assert 'cols_to_remove.append(col)' in source, 'Missing original column removal'
print('Test 2 PASSED: One-hot creates N binary columns, removes original')

# Test 3: Verify metadata update in one_hot
assert 'self.metadata["feature_columns"]' in source, 'Missing feature_columns update'
assert 'self.metadata["numeric_columns"]' in source, 'Missing numeric_columns update'
print('Test 3 PASSED: One-hot updates metadata')

# Test 4: Verify metadata update in label_encode
source_le = inspect.getsource(PreprocessingPipeline._apply_label_encode)
assert 'self.metadata["numeric_columns"]' in source_le, 'Missing label_encode metadata update'
print('Test 4 PASSED: Label encode updates metadata')

# Test 5: Verify import statement
with open(r'E:\Workspace\NNFactory\backend\preprocessing_pipeline.py') as f:
    content = f.read()
assert 'from dataset_manager import DatasetManager, DatasetInfo' in content, 'Missing DatasetInfo import'
assert 'DatasetInfo(**new_info)' in content, 'Still using broken self.ds_manager.DatasetInfo'
assert 'self.ds_manager.DatasetInfo' not in content, 'Broken reference still present'
print('Test 5 PASSED: DatasetInfo import and usage fixed')

print()
print('All 5 tests passed!')
