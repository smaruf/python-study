# Python3 program to find if there is path
# from top left to right bottom
row = 5
col = 5

# to find the path from
# top left to bottom right
def isPath(arr) :

	# directions
	Dir = [ [0, 1], [0, -1], [1, 0], [-1, 0]]
	
	# queue
	q = []
	
	# insert the top right corner.
	q.append((0, 0))
	
	# until queue is empty
	while(len(q) > 0) :
		p = q[0]
		q.pop(0)
		
		# mark as visited
		arr[p[0]][p[1]] = -1
		
		# destination is reached.
		if(p == (row - 1, col - 1)) :
			return True
			
		# check all four directions
		for i in range(4) :
		
			# using the direction array
			a = p[0] + Dir[i][0]
			b = p[1] + Dir[i][1]
			
			# not blocked and valid
			if(a >= 0 and b >= 0 and a < row and b < col and arr[a][b] != -1) :		
				q.append((a, b))
	return False

# Given array
arr = [[ 0, 0, 0, -1, 0 ],
		[ -1, 0, 0, -1, -1],
		[ 0, 0, 0, -1, 0 ],
		[ -1, 0, 0, 0, 0 ],
		[ 0, 0, -1, 0, 0 ] ]

# path from arr[0][0] to arr[row][col]
if (isPath(arr)) :
	print("Yes")
else :
	print("No")

	# This code is contributed by divyesh072019
