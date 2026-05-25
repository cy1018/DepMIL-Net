from options_avec import Options
from solver_avec import Solver

def main():
    args = Options().parse()
    solver = Solver(args)
    solver.run()

if __name__ == '__main__':
    main()
