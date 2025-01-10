import numpy as np
import argparse

def makeMatrixFullRank(A):
    ''' Esta função recebe uma matriz, que pode ser em numpy,
        e retorna dois argumentos:
          - A matriz com linhas eliminadas
          - Uma lista indicando quais linhas foram eliminadas.
    '''
    if np.linalg.matrix_rank(A) == A.shape[0]: return A, []
    row = 1
    rowsEliminated = []
    counter = 0
    while 1:
        counter += 1
        B = A[0:(row+1), :]
        C = np.linalg.qr(B.T)[1]
        C[np.isclose(C, 0)] = 0
        if not np.any(C[row, :]):
            rowsEliminated.append(counter)
            A = np.delete(A, (row), axis=0)
        else:
            row += 1
        # end if
        if row >= A.shape[0]: break
    # end for
    return A, rowsEliminated
# end makeMatrixFullRank

def read_simplex_input(file_path : str):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

    num_vars = int(lines[0])
    num_constraints = int(lines[1])
    non_neg = list(map(int, lines[2].split()))
    c = np.array(list(map(float, lines[3].split()[1:])))

    A = []
    b = []
    signs = []

    for line in lines[4:]:
        parts = line.split()
        coeffs = list(map(float, parts[:-2]))
        sign = parts[-2]
        rhs = float(parts[-1])
        A.append(coeffs)
        b.append(rhs)
        signs.append(sign)
        
    A = np.array(A)
    b = np.array(b)
    return A, b, c, non_neg, signs

def formatValue(value, width, decimals): return f"{value:>{width}.{decimals}f}"

def printTableau(A, b, c, valopt, M, M_c, width=6, decimals=2):
    def formatRow(row): return " ".join(formatValue(x, width, decimals) for x in row)

    print(formatRow(M_c), '||', formatRow(c), '||', formatValue(valopt, width, decimals))
    print('-' * (width * (A.shape[1] + M.shape[1] + 2) + 10))  # Separator line

    for i in range(A.shape[0]):
        print(
            formatRow(M[i]), '||',
            formatRow(A[i]), '||',
            formatValue(b[i], width, decimals)
        )
    print('-' * (width * (A.shape[1] + M.shape[1] + 2) + 10))  

def getFPI():
    ''' Retorna o problema de programação linear: (LOG 09) '''
    
    A = np.array([[-1.00,  1.00,  1.00,  2.00,  1.00,  0.00,  0.00,  0.00,  0.00,  0.00],
                  [0.00,  0.00,  1.00, -2.00,  0.00,  1.00,  0.00,  0.00,  0.00,  0.00],
                  [2.00, -2.00,  1.00,  1.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00],
                  [1.00,  1.00,  0.00, -1.00,  0.00,  0.00,  0.00, -1.00,  0.00,  0.00],
                  [2.00, -1.00, -1.00,  2.00,  0.00,  0.00,  0.00,  0.00,  1.00,  0.00],
                  [-1.00, -2.00,  0.00, -2.00,  0.00,  0.00,  0.00,  0.00,  0.00,  1.00]])
    b = np.array([5.00, 1.00, 9.00, 1.00, 4.00, 9.00])
    c = np.array([-6.00, 10.00, 1.00, 2.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    valopt = 0.0
    M = np.array([[1.00,  0.00,  0.00,  0.00,  0.00,  0.00],
                 [0.00,  1.00,  0.00,  0.00,  0.00,  0.00],
                 [0.00,  0.00,  1.00,  0.00,  0.00,  0.00],
                 [0.00,  0.00,  0.00, -1.00,  0.00,  0.00],
                 [0.00,  0.00,  0.00,  0.00,  1.00,  0.00],
                 [0.00,  0.00,  0.00,  0.00,  0.00,  1.00]])
    M_c = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
    
    return A, b, c, valopt, M, M_c

policy = "bland"

def find_pivot_column(c):
    ''' Retorna a coluna pivô, caso não exista retorna -1
    '''
    negative_indices = [i for i, val in enumerate(c) if val < 0]
    if not negative_indices:
        return -1  # Ótimo
    if policy == "bland":
        return negative_indices[0]
    elif policy == "largest":
        return max(negative_indices, key=lambda i: abs(c[i]))
    elif policy == "smallest":
        return min(negative_indices, key=lambda i: abs(c[i]))
    raise ValueError(f"Política desconhecida: {policy}")


def pivot(A, b, c_aux, val_opt, M, M_c, row, column):
    ''' Faz o pivoteamento de uma linha e coluna '''
    
    el = A[row][column]
    A[row] /= el
    b[row] /= el
    M[row] /= el
    
    
    for i in range(len(A)):
        if i == row: continue
        mult = A[i][column]
        A[i] -= mult * A[row]
        b[i] -= mult * b[row]
        M[i] -= mult * M[row]
        
    mult = -c_aux[column]
    c_aux += mult * A[row]
    val_opt += mult * b[row]
    M_c += mult * M[row]
    
    return val_opt
    
def canonicalForm(basis : list[tuple[int, int]], A, b, c_aux, val_opt, M, M_c):
    ''' Transforma o tableau para a forma canônica '''
    for row, column in enumerate(basis):
        if c_aux[column] != 0:
            
            mult = 0
            mult = -c_aux[column]/A[row][column]
            
            line = A[row] * mult
            c_aux += line
            val_opt += b[row] * mult
            
            line = M[row] * mult
            M_c += line
    
    return val_opt   

def simplexIteration(basis, A_aux, b, c_aux, val_opt, M, M_c):
    
    # Estratégia de escolha de coluna: escolhe a primeira coluna com elemento negativo -> Bland
    column = find_pivot_column(c_aux)
    
    # Escolha de qual coluna sai da base (menor razão positiva)
    row = -1
    min_ratio = np.inf
    for i in range(len(A_aux)):
        if A_aux[i][column] > 0:
            ratio = b[i] / A_aux[i][column]
            if ratio < min_ratio:
                min_ratio = ratio
                row = i

    if row == -1:
        raise Exception('Problema ilimitado')

    basis[row] = column

    return pivot(A_aux, b, c_aux, val_opt, M, M_c, row, column)
    
def first_phase(A, b, c, M, M_c):
    ''' Implementação da primeira fase do simplex '''
    I = np.eye(len(A))
    A_aux = A.copy()
    A_aux = np.concatenate((A_aux, I), axis=1)

    c_aux = np.concatenate((np.zeros(len(c)), np.ones(len(I))), axis=0)
    val_opt = 0
    
    
    printTableau(A_aux, b, c_aux, 0, M, M_c)
    
    # Transform to canonical form
    # elements in -c^t above basis must be 0

    basis = [len(A[0]) + i for i in range(len(A))]

    printTableau(A_aux, b, c_aux, val_opt, M, M_c)
    val_opt = canonicalForm(basis, A_aux, b, c_aux, val_opt, M, M_c)            
    printTableau(A_aux, b, c_aux, val_opt, M, M_c)
    
    curIter = 0
    
    while np.any(c_aux < 0):
        curIter += 1
        val_opt = simplexIteration(basis, A_aux, b, c_aux, val_opt, M, M_c)
        print(f"Iteração {curIter}")
        printTableau(A_aux, b, c_aux, val_opt, M, M_c)
        print("basis:", basis)

    if val_opt < 0: 
        raise Exception('PL inviável!')


    # Get the solution 
    
    sol = [0 for _ in range(len(c_aux))]
    for row, col in enumerate(basis):
        sol[col] = b[row]

    print("Solution to Auxiliary PL: ")

    for i in sol:
        print(f"{i:.3f}", end=' ')
    print()
    
    return basis, A_aux, b, val_opt, M, M_c
    

def findInitialbasis(A, b, c, M, M_c):
    ''' Encontra uma base inicial para o problema '''
    # Tenta encontrar base trivial, I em alguma parte de A, se não encontrar, retorna simplex 1 fase com variaveis auxiliares
    pass  
    


def main():
    
    # manage input args
    parser = argparse.ArgumentParser(description="Simplex algorithm implementation.")
    parser.add_argument("filename", type=str, help="Nome do arquivo lp de entrada.")
    parser.add_argument("--decimals", type=int, default=3, help="N. de casas decimais para imprimir valores numéricos.")
    parser.add_argument("--digits", type=int, default=7, help="N. total de dígitos para imprimir valores numéricos.")
    parser.add_argument("--policy", type=str, choices=["largest", "bland", "smallest"], default="largest",
                        help="Política de seleção de coluna. Valores válidos: largest, bland, smallest.")
    args = parser.parse_args()

    width = args.digits
    decimals = args.decimals
    policy = args.policy

    A, b, c, non_neg, signs = read_simplex_input(args.filename)

    # Transform to FPI    
    
    A, b, c, valopt, M, M_c = getFPI()
    A, rowsEliminated = makeMatrixFullRank(A)
    print('rowsEliminated =', rowsEliminated)

    # Phase 1    
    
    basis, A_aux, b, val_opt, M, M_c = first_phase(A, b, c, M, M_c)
    print(val_opt)
    
    
    # Phase 2 
    
    A = A_aux[:, :(A_aux.shape[1] - len(b))]
    val_opt = canonicalForm(basis, A, b, c, 0, M, M_c)            
    
    print("Canonical form to PL after getting the basis for it: ")
    printTableau(A, b, c, val_opt, M, M_c)
    
    curIter = 0
    
    while np.any(c < 0):
        val_opt = simplexIteration(basis, A, b, c, val_opt, M, M_c)
        print(f"Iteração {curIter}")
        printTableau(A, b, c, val_opt, M, M_c)
        curIter += 1
        print("basis:", basis)

    sol = [0 for _ in range(len(c))]
    for row, col in enumerate(basis):
        sol[col] = b[row]

    print("Solution to FPI PL: ")

    for i in sol:
        print(formatValue(i, width, decimals), end=' ')
    print()
    
        
if __name__ == '__main__': main()