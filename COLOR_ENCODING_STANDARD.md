# COLOR ENCODING STANDARD FOR BUBBLE GAME

## 🎯 UNIFIED STANDARD: 0-BASED INDEXING (0-5)

All bubble-related files now use **consistent 0-based color indexing** from 0 to 5.

## 📁 FILE-BY-FILE IMPLEMENTATION:

### 1. `bubbles.py` (Main Game) ✅
- **BUBBLE_COLORS**: List of 6 RGB tuples
- **Color storage**: `bubble.color` stores RGB tuples
- **Indexing**: `BUBBLE_COLORS.index(bubble.color)` returns 0-5
- **Status**: CORRECT - Already uses 0-based indexing

### 2. `train_target_bubbles_ai.py` (Training Environment) ✅
- **BUBBLE_COLORS**: Constant = 6
- **Color generation**: `random.randint(0, BUBBLE_COLORS-1)` → 0-5
- **Grid storage**: `self.grid_player2[(row, col)] = color` where color is 0-5
- **State encoding**: Direct use of 0-5 indices in color planes
- **Status**: FIXED - Now uses 0-based indexing consistently

### 3. `bubbles_target_dqn.py` (DQN Agent) ✅
- **BUBBLE_COLORS**: Constant = 6
- **State handling**: Expects 0-5 color indices
- **Status**: CORRECT - Already uses 0-based indexing

## 🔄 COLOR MAPPING:

| Index | RGB Color | Description |
|-------|-----------|-------------|
| 0     | (255, 0, 0)   | Red        |
| 1     | (0, 255, 0)   | Green      |
| 2     | (0, 0, 255)   | Blue       |
| 3     | (255, 255, 0) | Yellow     |
| 4     | (255, 0, 255) | Magenta    |
| 5     | (0, 255, 255) | Cyan       |

## ✅ VERIFICATION:

- **Training**: Generates colors 0-5, stores as 0-5, encodes as 0-5
- **Game**: RGB colors map to indices 0-5 via `BUBBLE_COLORS.index()`
- **DQN**: Receives and processes 0-5 color indices
- **State**: 6 color planes with 0-5 indexing

## 🚫 NO MORE CONVERSIONS:

- **No more `color_idx_1based - 1`**
- **No more `max(0, min(BUBBLE_COLORS - 1, color_idx_1based - 1))`**
- **Direct use of 0-5 indices throughout the pipeline**

## 🎮 RESULT:

AI trained in training environment will now work correctly in the main game because:
1. **Same color encoding** (0-5)
2. **Same state representation** (6 color planes)
3. **Same color associations** (Red=0, Green=1, etc.)
4. **No more environment mismatch**

This should resolve the "AI shooting upwards" issue caused by color encoding confusion.
