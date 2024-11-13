import kociemba

# Define a scrambled cube
# The cube string format uses a specific order: URFDLB (Up, Right, Front, Down, Left, Back)
scrambled_cube = "DRLUUBFBRBLARLRLUBULBLFRFDFLRLLLBRDDRFFDDBUUUFFFUUUBFBLDRRBD"

def solve_cube(scrambled_state):
    # Solve the cube using kociemba.solve()
    solution = kociemba.solve(scrambled_state)
    return solution

if __name__ == "__main__":
    solution_moves = solve_cube(scrambled_cube)
    print("Solution moves:")
    print(solution_moves)
    # Output will be a series of move notations that will solve the cube
