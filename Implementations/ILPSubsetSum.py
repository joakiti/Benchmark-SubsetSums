from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, LpMaximize, LpBinary, LpInteger, PULP_CBC_CMD

from Implementations.Interfaces.IRunnableAlgorithm import IRunnableAlgorithm


class ILPSubsetSum(IRunnableAlgorithm):
    def __init__(self, label):
        self.label = label

    def run(self, values, target):
        # Create the model
        model = LpProblem(name="small-problem", sense=LpMaximize)

        # Initialize the decision variables
        variables = [LpVariable(f"{values[i]}", 0, 1, LpInteger) for i in range(len(values))]
        # for i in range(0, len(variables)):
        #     model += variables[i] * values[i]
                #(variables[x - 1] + variables[edge - 1] >= 1)
        model += sum([variables[val] * values[val] for val in range(len(variables))])
        model += sum([variables[val] * values[val] for val in range(len(variables))]) <= target
        # model += lpSum(variables[v]*values[v] for v in range(len(variables))) <= target
        # Add the objective function to the model
        # print(model)
        status = model.solve(PULP_CBC_CMD(msg=True, maxSeconds=10, mip=True, gapAbs=0), )
        # print(f"status: {model.status}, {LpStatus[model.status]}")
        # print(f"objective: {model.objective.value()}")

        # Solve the problem
        # status = model.solve()
        result = list()
        for var in model.variables():
            if var.value() == 1.0:
                result.append(int(var.name))
        #     print(f"{var.name}: {var.value()}")
        # print(f"objective: {model.objective.value()}")
        return result