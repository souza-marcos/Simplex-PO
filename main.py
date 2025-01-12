import numpy as np
import argparse

# Exceptions for the simplex algorithm
class Infeasible(Exception):
    def __init__(self):
        super().__init__()

class Unbounded(Exception):
    def __init__(self):
        super().__init__()

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

policy = "bland"
width = 7
decimals = 3

def readInput(file_path : str):
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
    return A, b, c, non_neg, signs, num_vars, num_constraints

def formatValue(value, width, decimals): return f"{value:>{width}.{decimals}f}"

def printTableau(A, b, c, valopt, M, M_c, basis, msg=""):
    def formatRow(row): return " ".join(formatValue(x, width, decimals) for x in row)

    if msg != '': print("\n" + msg, end='\n\n')
    
    print(formatRow(M_c), '||', formatRow(c), '||', formatValue(valopt, width, decimals), "  Base: ")
    print('-' * (width * (A.shape[1] + M.shape[1] + 2) + 10))  # Separator line

    for i in range(A.shape[0]):
        print(
            formatRow(M[i]), '||',
            formatRow(A[i]), '||',
            formatValue(b[i], width, decimals),
            f"      {basis[i]}" if basis[i] != -1 else ""
        )
    print('-' * (width * (A.shape[1] + M.shape[1] + 2) + 10))  

def printResult(status, valopt, sol: list, dual: list):
    
    if status == "otima" or status == "multipla":    
        print(f"Status: otimo {"(multiplos)" if status == 'multipla' else ''}")
        print("Objetivo:", valopt)
        print("Solução: ")
        for i in sol:
            print(formatValue(i, width, decimals), end=' ')
        print()
        
        print("Dual: ")
        for i in dual:
            print(formatValue(i, width, decimals), end=' ')    
        print()
    else:
        print("Status:", status)

threshold = 1e-6
# Tratar mal condicionamento das matrizes
def handleZero(G):
    G[np.abs(G) < threshold] = 0
    return G


def findPivotColumn(c):
    ''' Retorna a coluna pivô, caso não exista retorna -1
    '''
    c = handleZero(c)
    negative_indices = [(val, i) for i, val in enumerate(c) if (val < 0) and (abs(val) > threshold)]
    
    if not negative_indices:
        return -1
    if policy == "bland":
        negative_indices.sort()
        return negative_indices[-1][1]
    elif policy == "largest":
        return np.argmax(c)
    elif policy == "smallest":
        return np.argmin(c)
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
    
    
    A = handleZero(A)
    b = handleZero(b)
    c_aux = handleZero(c_aux)
    M = handleZero(M)
    M_c = handleZero(M_c)
    
    return val_opt

def simplexIteration(basis, A_aux, b, c_aux, val_opt, M, M_c):
    
    column = findPivotColumn(c_aux)
    
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
        raise Unbounded()

    basis[row] = column

    return pivot(A_aux, b, c_aux, val_opt, M, M_c, row, column)
    
def firstPhase(basis, A, b, M, M_c):
    ''' Implementação da primeira fase do simplex '''
   
    count = 0
    c = np.zeros(len(A[0]))
    for i, bi in enumerate(basis):
        if bi == -1:
            A = np.insert(A, A.shape[1], 0, axis=1)
            A[i][-1] = 1
            c = np.insert(c, c.shape[0], 1)
            count += 1
            basis[i] = len(A[0]) - 1
            
    val_opt = 0
    
    
    printTableau(A, b, c, val_opt, M, M_c, basis, "Tableau inicial para a PL auxiliar: ")
    
    for i, bi in enumerate(basis):
        val_opt = pivot(A, b, c, val_opt, M, M_c, i, bi)      
    
    printTableau(A, b, c, val_opt, M, M_c, basis, "Forma canônica para a PL auxiliar: ")
    
    curIter = 0
    
    while np.any(c < 0):
        curIter += 1
        val_opt = simplexIteration(basis, A, b, c, val_opt, M, M_c)
        printTableau(A, b, c, val_opt, M, M_c, basis, f"Iteração {curIter}: ")

    if val_opt < 0: 
        raise Infeasible()


    # Get the solution 
    
    sol = [0 for _ in range(len(c))]
    for row, col in enumerate(basis):
        sol[col] = b[row]

    print("Solution to Auxiliary PL: ")

    for i in sol:
        print(f"{i:.3f}", end=' ')
    print()
    
    A = A[:, :(A.shape[1] - count)]
    return basis, A, b, M, M_c 

def findTrivialbasis(A):
    ''' Encontra colunas triviais para a base'''

    basis = [ -1 for _ in range(len(A))]
    A = handleZero(A)

    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] == 1 and np.count_nonzero(A[:, j]) == 1:
                basis[i] = j
                break

    return basis



def main():
    
    # manage input args
    parser = argparse.ArgumentParser(description="Simplex algorithm implementation.")
    parser.add_argument("filename", type=str, help="Nome do arquivo lp de entrada.")
    parser.add_argument("--decimals", type=int, default=3, help="N. de casas decimais para imprimir valores numéricos.")
    parser.add_argument("--digits", type=int, default=7, help="N. total de dígitos para imprimir valores numéricos.")
    parser.add_argument("--policy", type=str, choices=["largest", "bland", "smallest"], default="largest",
                        help="Política de seleção de coluna. Valores válidos: largest, bland, smallest.")
    args = parser.parse_args()

    global width, decimals, policy 
    
    width = args.digits
    decimals = args.decimals
    policy = args.policy

    A, b, c, non_neg, signs, num_variables, num_restrictions = readInput(args.filename)

    # A, rowsEliminated = makeMatrixFullRank(A)
    # print("Linhas eliminadas: ", rowsEliminated)
    # b = np.delete(b, rowsEliminated)
    
    # Transform to FPI    

    newColumns = 0
    for j in range(len(non_neg)):
        if non_neg[j] == 0: # Free variable
            A = np.insert(A, A.shape[1], 0, axis=1)
            A[:, -1] = -A[:, j]
            c = np.insert(c, c.shape[0], -c[j])
            # newColumns += 1
        elif non_neg[j] == -1: # Non-positive variable
            A[:, j] = -A[:, j]
            c[j] = -c[j]


    for i in range(len(A)):
        if signs[i] == '<=':
            A = np.insert(A, A.shape[1], 0, axis=1)
            A[i][A.shape[1] - 1] = 1
            c = np.insert(c, c.shape[0], 0)
            newColumns += 1
        elif signs[i] == '>=':
            A = np.insert(A, A.shape[1], 0, axis=1)
            A[i][A.shape[1] - 1] = -1
            c = np.insert(c, c.shape[0], 0)
            newColumns += 1
    
    M = np.eye(len(A))
    M_c = np.zeros(len(A))
    val_opt = 0.0
    
    
    for i, val in enumerate(b):
        if val < 0:
            A[i] = -A[i]
            b[i] = -b[i]
            M[i] = -M[i]
    
    
    basis = findTrivialbasis(A)   
    
    c = -c
    printTableau(A, b, c, val_opt, M, M_c, basis, "Tableau para a PL FPI: ")
    
    try:
        if -1 in basis: # Temos que rodar com variaveis artificiais
            basis, A, b, M, M_c = firstPhase(basis, A, b, M, M_c)
            printTableau(A, b, c, val_opt, M, M_c, basis, "Tableau para a PL FPI, usando base encontrada: ")
    
    except Infeasible :
        print("Status: Inviável")
        return
    
    except Unbounded:
        print("Status: Inviável")
        return 
    
    # Phase 2 
    # Canonizar o tableau
    for i, bi in enumerate(basis):
        val_opt = pivot(A, b, c, val_opt, M, M_c, i, bi)
                 
    printTableau(A, b, c, val_opt, M, M_c, basis, "Forma canônica da PL: ")
    
    curIter = 0
    
    try:
        while np.any(c < 0):
            val_opt = simplexIteration(basis, A, b, c, val_opt, M, M_c)
            curIter += 1
            printTableau(A, b, c, val_opt, M, M_c, basis, f"Iteração {curIter}")
            
    except Unbounded:
        print("Status: Ilimitada")
        return

    sol = [0 for _ in range(len(c))]
    for row, col in enumerate(basis):
        sol[col] = b[row]


    sol = sol[:-newColumns]
    
    print("Status: otimo")
    print("Objetivo: ", formatValue(val_opt, width, decimals))
    print("Solução: ")
    
    for i in sol:
        print(formatValue(i, width, decimals), end=' ')
    print()
    
    print("Dual: ")
    for i in M_c:
        print(formatValue(i, width, decimals), end=' ')    
    print()
        
if __name__ == '__main__': main()