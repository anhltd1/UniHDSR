# Missing Components and Import Issues in UniHDSA

## Problem Summary

The UniHDSA configuration files are trying to import from `projects.unified_layout_analysis_v2`, but this module is missing from the repository. This is a common issue when research code dependencies are not fully included.

## Missing Imports

### 1. Missing Project Structure

The configuration files expect this structure:

```
projects/
└── unified_layout_analysis_v2/
    ├── evaluation/
    │   └── unified_layout_evaluation.py  # UniLayoutEvaluator
    └── modeling/
        ├── backbone/
        │   └── bert.py  # TextTokenizer, Bert classes
        ├── doc_transformer.py
        ├── uni_relation_prediction_head.py
        └── # other components
```

### 2. Current Import Errors

From `hdsa_bert.py`:

```python
# These imports are failing:
from projects.unified_layout_analysis_v2.evaluation.unified_layout_evaluation import UniLayoutEvaluator
from projects.unified_layout_analysis_v2.modeling.backbone.bert import TextTokenizer
```

From model configuration files:

```python
# These imports are also failing:
from projects.unified_layout_analysis_v2.modeling import (
    UniDETRMultiScales,
    DabDeformableDetrTransformer,
    # ... other components
)
```

## Solutions

### Option 1: Use Existing Code (Current Approach)

I've started fixing the imports by using existing code in your repository:

1. **Fixed UniLayoutEvaluator Import**:

   - Changed to use existing: `evaluation.unified_layout_evaluation`
   - ✅ This file exists in your repo

2. **Created Temporary TextTokenizer**:

   - Created: `UniHDSA/utils/text_tokenizer.py`
   - ✅ Temporary implementation using transformers library

3. **Still Need to Fix**:
   - `HRDocDatasetMapper` from detrex
   - Model components (UniDETRMultiScales, etc.)
   - BERT implementation

### Option 2: Install detrex Framework (Recommended)

The missing components are likely part of the detrex framework. You need to:

1. **Install detrex properly**:

```bash
# Clone detrex repository
git clone https://github.com/IDEA-Research/detrex.git
cd detrex

# Install detectron2 first
python -m pip install -e detectron2

# Install detrex
pip install -e .
```

2. **Verify detrex installation**:

```python
import detrex
from detrex.data.dataset_mappers import HRDocDatasetMapper
```

### Option 3: Create Missing Project Structure

If you have access to the original UniHDSA code, create the missing structure:

1. **Create projects directory**:

```bash
mkdir -p projects/unified_layout_analysis_v2/evaluation
mkdir -p projects/unified_layout_analysis_v2/modeling/backbone
```

2. **Add required files**:
   - `projects/unified_layout_analysis_v2/evaluation/unified_layout_evaluation.py`
   - `projects/unified_layout_analysis_v2/modeling/backbone/bert.py`
   - Other model components

## Current Status

### ✅ Fixed

- UniLayoutEvaluator import path
- Created temporary TextTokenizer implementation
- Added proper Python path handling

### ❌ Still Need to Fix

- `HRDocDatasetMapper` - depends on detrex installation
- Model components in `unihdsa_r50_bert.py`:
  - `UniDETRMultiScales`
  - `DabDeformableDetrTransformer`
  - `UniRelationPredictionHead`
  - `DocTransformer`, `DocTransformerEncoder`
  - `Bert` class

## Next Steps

### Immediate Actions:

1. **Install detrex framework** (most important):

```bash
# In your conda environment
pip install torch torchvision transformers
git clone https://github.com/IDEA-Research/detrex.git
cd detrex
git submodule init && git submodule update
python -m pip install -e detectron2
pip install -e .
```

2. **Test detrex installation**:

```python
python -c "import detrex; from detrex.data.dataset_mappers import HRDocDatasetMapper; print('detrex OK')"
```

3. **If detrex installation fails**, you'll need to:
   - Create the missing model components manually
   - Or find the original UniHDSA implementation
   - Or adapt the configuration to use available components

### Alternative Approach:

If you can't install detrex properly, I can help you:

1. Create simplified versions of the missing components
2. Adapt the configuration to use standard detectron2/transformers components
3. Modify the model architecture to work with available libraries

## Files Modified

1. **`UniHDSA/configs/data/hdsa_bert.py`**: Fixed import paths
2. **`UniHDSA/utils/text_tokenizer.py`**: Created temporary TextTokenizer
3. **This documentation**: `MISSING_COMPONENTS.md`

## Recommendations

1. **Priority 1**: Install detrex framework properly
2. **Priority 2**: If detrex fails, we'll need to implement missing components
3. **Priority 3**: Consider using standard detectron2 models as alternatives

The root cause is that this appears to be research code that depends on a specific version of detrex or custom extensions that aren't included in the repository.
