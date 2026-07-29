from fastmcp import FastMCP

mcp = FastMCP('calculator')

@mcp.tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    if operation == 'add':
        return {"result": first_num + second_num}
    elif operation == 'subtract':
        return {"result": first_num - second_num}
    elif operation == 'multiply':
        return {"result": first_num * second_num}
    elif operation == 'divide':
        if second_num != 0:
            return {"result": first_num / second_num}
        else:
            return {"error": "Division by zero is not allowed."}
    else:
        return {"error": "Invalid operation. Please use 'add', 'subtract', 'multiply', or 'divide'."}


if __name__ == "__main__":
    mcp.run()