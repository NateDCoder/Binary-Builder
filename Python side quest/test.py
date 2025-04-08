import numpy as np

def tableOperator(A, B):
    # Ensure A and B are numpy arrays and broadcast scalars to arrays if necessary
    A = np.asarray(A)
    B = np.asarray(B)
    
    # Initialize an empty list to store the results of logical operations
    operations = []

    # Perform logical operations
    operations.append(np.zeros(A.shape))        # 0: False
    operations.append(A * B)                    # 1: A ∧ B
    operations.append(A - A * B)                # 2: ¬(A ⇒ B)
    operations.append(A)                        # 3: A
    operations.append(B - A * B)                # 4: ¬(A ⇐ B)
    operations.append(B)                        # 5: B
    operations.append(A + B - 2 * A * B)        # 6: A ⊕ B
    operations.append(A + B - A * B)            # 7: A ∨ B
    operations.append(1 - (A + B - A * B))      # 8: ¬(A ∨ B)
    operations.append(1 - (A + B - 2 * A * B))  # 9: ¬(A ⊕ B)
    operations.append(1 - B)                    # 10: ¬B
    operations.append(1 - B + A * B)            # 11: A ⇐ B
    operations.append(1 - A)                    # 12: ¬A
    operations.append(1 - A + A * B)            # 13: A ⇒ B
    operations.append(1 - A * B)                # 14: ¬(A ∧ B)
    operations.append(np.ones(A.shape))         # 15: True

    # Stack all operations into a 16 x n matrix
    result = np.vstack(operations)
    
    return result

# Example usage:
A = np.array([1, 0])  # Example A values (binary)
B = np.array([0, 1])  # Example B values (binary)

result = tableOperator(A, B)
print(result)

result_scalar = tableOperator([1], [1])  # This works with scalar input
print(result_scalar)
