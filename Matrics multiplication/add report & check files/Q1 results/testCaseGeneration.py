import numpy as np

# Create a 100x100 matrix filled with float numbers
matrix = np.ones((10000, 10000))

matrix2 = np.ones((10000,10000))
# np.fill(matrix, 4)

# Specify the file path
file_path = "matrix_data.txt"

# Write the matrix to the file
with open(file_path, 'w') as file:
    file.write('1\n')
    file.write('100 100\n')

    for row in matrix:
        # Convert each row to a string of space-separated float numbers
        row_str = ' '.join(map(str, row))
        # Write the row to the file
        file.write(row_str + '\n')

    for row in matrix2:
        # Convert each row to a string of space-separated float numbers
        row_str = ' '.join(map(str, row))
        # Write the row to the file
        file.write(row_str + '\n')

print("Matrix has been written to", file_path)
