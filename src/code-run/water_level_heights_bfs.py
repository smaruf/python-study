'''
You are given an integer matrix `isWater` of size `m x n` that represents a map of land and water cells.

- If `isWater[i][j] == 0`, cell `(i, j)` is a land cell.
- If `isWater[i][j] == 1`, cell `(i, j)` is a water cell.

You must assign each cell a height in a way that follows these rules:

1. The height of each cell must be non-negative.
2. If the cell is a water cell, its height must be 0.
3. Any two adjacent cells must have an absolute height difference of at most 1.
   A cell is adjacent to another cell if the former is directly north, east, south, or west of the latter (i.e., their sides are touching).

Your goal is to assign heights to all cells such that:
- The maximum height in the matrix is as large as possible while following the rules.

Return an integer matrix `height` of size `m x n` where `height[i][j]` is the height of cell `(i, j)`. If there are multiple solutions, return any of them.

### Examples:

#### Example 1:
Input: `isWater = [[0,1],[0,0]]`
Output: `[[1,0],[2,1]]`
Explanation: 
- The blue cell is the water cell, and the green cells are the land cells.
- Heights are assigned such that the maximum height is maximized while satisfying the rules.

#### Example 2:
Input: `isWater = [[0,0,1],[1,0,0],[0,0,0]]`
Output: `[[1,1,0],[0,1,1],[1,2,2]]`
Explanation: 
- A height of 2 is the maximum possible height of any assignment.
- Any height assignment that has a maximum height of 2 while still meeting the rules will also be accepted.

### Constraints:
- `m == isWater.length`
- `n == isWater[i].length`
- `1 <= m, n <= 1000`
- `isWater[i][j]` is either `0` (land) or `1` (water).
'''
from collections import deque

def assign_heights(isWater):
    rows, cols = len(isWater), len(isWater[0])
    heights = [[-1] * cols for _ in range(rows)]  # Initialize heights with -1
    queue = deque()

    # Initialize the queue with all water cells and set their height to 0
    for r in range(rows):
        for c in range(cols):
            if isWater[r][c] == 1:
                heights[r][c] = 0
                queue.append((r, c))

    # Directions for moving up, down, left, right
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # Perform BFS to assign heights
    while queue:
        r, c = queue.popleft()
        current_height = heights[r][c]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and heights[nr][nc] == -1:
                heights[nr][nc] = current_height + 1
                queue.append((nr, nc))

    return heights

if __name__ == '__main__':
    isWater = [[0, 0, 1], [1, 0, 0], [0, 0, 0]]
    heights = assign_heights(isWater)
    print(heights)
