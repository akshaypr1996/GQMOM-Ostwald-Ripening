################################################################################
#
#  PyQuadMom - Python tools for Quadrature-Based Moment Methods
#              2023 - 2025 (C) - Alberto Passalacqua
#
################################################################################
#
#  License
#    This file is part of PyQuadMom.
#
#    PyQuadMom is free software: you can redistribute it and/or modify it under 
#    the terms of the GNU General Public License as published by the Free 
#    Software Foundation, either version 3 of the License, or (at your option) 
#    any later version.
#
#    PyQuadMom is distributed in the hope that it will be useful, but WITHOUT 
#    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
#    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for 
#    more details.
#
#    You should have received a copy of the GNU General Public License along 
#    with PyQuadMom.  If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

import numpy as np
import matplotlib.pyplot as plt

def buildJacobiMatrix(aRecurrence, bRecurrence):
    """
    Build the Jacobi matrix from the recurrence coefficients aRecurrence and 
    bRecurrence.

    :param aRecurrence: recurrence coefficients a
    :type aRecurrence: np.ndarray
    :param bRecurrence: recurrence coefficients b
    :type bRecurrence: np.ndarray
    :return: Jacobi matrix
    :rtype: np.ndarray
    """
    
    # Dimension of the n x n Jacobi matrix (bRecurrence has dimension n - 1)
    nJacobi: int  = aRecurrence.size

    # Initialize Jacobi matrix
    JacobiMatrix: np.ndarray = np.zeros((nJacobi, nJacobi))

    # Populate the diagonal of the Jacobi matrix
    for i in range(nJacobi):
        JacobiMatrix[i, i] = aRecurrence[i]

    # Populate the sub- and supra-diagonal of the Jacobi matrix
    for i in range(nJacobi - 1):
        JacobiMatrix[i, i + 1] = np.sqrt(bRecurrence[i + 1])
        JacobiMatrix[i + 1, i] = JacobiMatrix[i, i + 1]
        
    return JacobiMatrix


def calculateQuadrature(JacobiMatrix, m0):

    """
    Calculate the quadrature weights and abscissae from the Jacobi matrix.

    :param JacobiMatrix: Jacobi matrix of the moments
    :type JacobiMatrix: np.ndarray
    :param m0: scaling factor for the weights, defaults to 1
    :type m0: float, optional 
    :return: weights and abscissae of the quadrature
    :rtype: np.ndarray, np.ndarray
    """
    
    # Find eigenvalues and eigenvectors of JacobiMatrix
    eigenValues, eigenVectors = np.linalg.eigh(JacobiMatrix)

    # Dimension of the n x n Jacobi matrix
    nJacobi = JacobiMatrix.shape[0] 

    # Calculate weights and abscissae
    weights = np.zeros(nJacobi)
    abscissae = np.zeros(nJacobi)

    for i in range(nJacobi):
        weights[i] = m0*(eigenVectors[0, i]**2)
        abscissae[i] = eigenValues[i]

    return weights, abscissae  


def WheelerGQMOMRPlus(moments, totalNodes, type="gamma", smallZeta = 1.0e-14, debug=False):
    """
    Perform Wheeler GQMOM R+ inversion to obtain weights and abscissae from
    input moments.
    
    :param moments: input moments
    :type moments: np.ndarray
    :param totalNodes: total number of quadrature nodes
    :type totalNodes: int
    :param type: type of distribution for GQMOM closure, defaults to "gamma"
    :type type: str, optional
    :param smallZeta: minimum value for zeta coefficients, defaults to 1.0e-14
    :type smallZeta: float, optional
    :param debug: enable debug prints, defaults to False
    :type debug: bool, optional
    :return: weights, abscissae, nu parameter, zeta coefficients
    :rtype: np.ndarray, np.ndarray, float, np.ndarray
    """

    nMoments = len(moments)
    generalized = True

    # Theoretical maximum number of base nodes
    baseNodes = nMoments // 2

    # First work on known moments and check realizability to determine the
    # number of base nodes that can be found from the quadrature
    zeta, nMomentsInInteriorOfMomentSpace, aBaseRecurrence, bBaseRecurrence = \
        momentsToZetaWheeler(moments, minZeta=smallZeta)
    
    if (debug):
        print(f"Input moments: {moments}")
        print(f"Number of moments in interior of moment space: {nMomentsInInteriorOfMomentSpace}")
        print(f"Base recurrence coefficients a: {aBaseRecurrence}")
        print(f"Base recurrence coefficients b: {bBaseRecurrence}")
        print(f"Base zeta coefficients: {zeta}")

    if (nMomentsInInteriorOfMomentSpace <= 1):
        weights = np.zeros(totalNodes)
        abscissae = np.zeros(totalNodes)
        nu = 0.0
        print("Error: Insufficient number of realizable moments to calculate quadrature.")
        return weights, abscissae, nu, zeta

    baseNodes = nMomentsInInteriorOfMomentSpace // 2

    # Decide if GQMOM will be used
    if totalNodes <= baseNodes:
        totalNodes = baseNodes
        generalized = False

    if debug:
        print(f"Number of base nodes: {baseNodes}")
        print(f"Using total nodes: {totalNodes}")

    aRecurrence = np.zeros(totalNodes)
    bRecurrence = np.zeros(totalNodes)

    if (debug):
        print(f"Base recurrence coefficients a: {aBaseRecurrence[:baseNodes]}")
        print(f"Base recurrence coefficients b: {bBaseRecurrence[:baseNodes]}")

    nu = 0.0

    if (generalized):
        if debug:
            print("Using GQMOM closure for recurrence coefficients.")
            print(f"Base zeta values: {zeta}")

        # Extend zeta to accommodate totalNodes

        # Copy aBaseRecurrence and bBaseRecurrence to aRecurrence and bRecurrence
        for i in range(baseNodes):
            aRecurrence[i] = aBaseRecurrence[i]

        for i in range(baseNodes + 1):
            bRecurrence[i] = bBaseRecurrence[i]

        additionalNodes = totalNodes - baseNodes

        if (debug):
            print(f"Extending zeta for {additionalNodes} additional nodes.")

        zeta = np.concatenate((zeta, np.zeros(2 * additionalNodes)))

        # Compute nu based on distribution type and close the recurrence with
        # the GQMOM closure
        if type == "gamma":
            sqrM1 = moments[1] ** 2
            nu = (sqrM1 / (moments[2] * moments[0] - sqrM1)) - 1
            for i in range(baseNodes + 1, totalNodes + 1):
                zeta[2 * i - 2] = (i + nu) * zeta[2 * baseNodes - 2] / (baseNodes + nu)
                zeta[2 * i - 1] = i * zeta[2 * baseNodes - 1] / baseNodes
        elif type == "lognormal":
            nu = np.sqrt(moments[2] * moments[0] / (moments[1] ** 2))
            for i in range(baseNodes + 1, totalNodes):
                zeta[2 * i - 2] = zeta[2 * baseNodes - 2] * (nu ** (4 * (i - baseNodes)))
                a = nu ** (2 * i) - 1
                b = nu ** (2 * baseNodes) - 1
                c = nu ** (2 * (i - baseNodes))
                zeta[2 * i - 1] = zeta[2 * baseNodes - 1] * c * a / b
        else:
            raise ValueError('Unknown type: use "gamma" or "lognormal".')
        
        aRecurrence[0] = zeta[0]

        for i in range(baseNodes, totalNodes):
            aRecurrence[i] = zeta[2 * i - 1] + zeta[2 * i]

        for i in range(baseNodes + 1, totalNodes):
            bRecurrence[i] = zeta[2 * i - 2] * zeta[2 * i - 1]

        if debug:
            print(f"Computed nu: {nu}")
            print(f"Zeta values: {zeta}")
            print(f"Recurrence coefficients a: {aRecurrence}")
            print(f"Recurrence coefficients b: {bRecurrence}")
    else:
        aRecurrence = aBaseRecurrence
        bRecurrence = bBaseRecurrence

    J = buildJacobiMatrix(aRecurrence, bRecurrence)
    weights, abscissae = calculateQuadrature(J, moments[0])

    return weights, abscissae, nu, zeta


def momentsToZetaWheeler(moments, minZeta=1e-14):
    """
    Convert a vector of moments to the corresponding zeta coefficients using
    the Wheeler algorithm.
    :param moments: input moments
    :type moments: np.ndarray
    :param minZeta: minimum value for zeta coefficients, defaults to 1e-14
    :type minZeta: float, optional
    :return: zeta coefficients, number of interior moments, coefficients a and b
    :rtype: np.ndarray, int, np.ndarray, np.ndarray"""

    moments = np.asarray(moments, dtype=float)

    # Number of zeta to be computed (zeta0 = 1 is not stored)
    nZeta = len(moments) - 1
    zeta = np.zeros(nZeta, dtype=float)
    nMomentsInInteriorOfMomentSpace = -1

    nMoments = len(moments)

    # Check m0
    if moments[0] < 1e-100:
        nMomentsInInteriorOfMomentSpace = 0
        return zeta, nMomentsInInteriorOfMomentSpace, None, None

    # Special case nn=1
    if nZeta == 1:
        zeta[0] = moments[1] / moments[0]
        return zeta, nZeta, None, None

    nd = nZeta // 2
    nr = nZeta - 2*nd

    # Allocate z
    z = np.zeros((nd+3, nZeta+2))
    z[2, 0:nZeta+1] = moments / moments[0]

    aRecurrence = np.zeros(nd+1)
    bRecurrence = np.zeros(nd+1)

    # First terms found explicitly
    aRecurrence[0] = moments[1] / moments[0]
    bRecurrence[0] = 1.0

    k = 1
    for l in range(k, nZeta-k+1):
        z[k+2, l] = z[k+1, l+1] - aRecurrence[k-1]*z[k+1, l] - bRecurrence[k-1]*z[k, l]

    zeta[0] = aRecurrence[0]

    # Stop is zeta[0] is too small
    if zeta[0] <= minZeta:
        nMomentsInInteriorOfMomentSpace = 0
        zeta[0] = 0.0
        return zeta, nMomentsInInteriorOfMomentSpace, aRecurrence, bRecurrence

    # Find recurrence coefficients and zeta
    for k in range(1, nd):
        bRecurrence[k] = z[k+2, k] / z[k+1, k-1]
        zeta[2*k-1] = bRecurrence[k] / zeta[2*k-2]
        if zeta[2*k-1] <= minZeta:
            nMomentsInInteriorOfMomentSpace = 2*k - 1
            zeta[2*k-1] = 0.0
            return zeta, nMomentsInInteriorOfMomentSpace, aRecurrence, bRecurrence

        aRecurrence[k] = z[k+2, k+1]/z[k+2, k] - z[k+1, k]/z[k+1, k-1]
        zeta[2*k] = aRecurrence[k] - zeta[2*k-1]
        if zeta[2*k] <= minZeta:
            nMomentsInInteriorOfMomentSpace = 2*k
            zeta[2*k] = 0.0
            return zeta, nMomentsInInteriorOfMomentSpace, aRecurrence, bRecurrence

        for l in range(k+1, nZeta-k):
            z[k+3, l] = z[k+2, l+1] - aRecurrence[k]*z[k+2, l] - bRecurrence[k]*z[k+1, l]

    k = nd
    bRecurrence[k] = z[k+2, k] / z[k+1, k-1]
    zeta[2*k-1] = bRecurrence[k] / zeta[2*k-2]

    # Check for realizability
    if zeta[2*k-1] <= minZeta:
        nMomentsInInteriorOfMomentSpace = 2*k - 1
        zeta[2*k-1] = 0.0
        return zeta, nMomentsInInteriorOfMomentSpace, aRecurrence, bRecurrence

    if nr == 1:
        aRecurrence[k] = z[k+2, k+1]/z[k+2, k] - z[k+1, k]/z[k+1, k-1]
        zeta[2*k] = aRecurrence[k] - zeta[2*k-1]
        if zeta[2*k] <= minZeta:
            nMomentsInInteriorOfMomentSpace = 2*k
            zeta[2*k] = 0.0
            return zeta, nMomentsInInteriorOfMomentSpace, aRecurrence, bRecurrence

    nMomentsInInteriorOfMomentSpace = nZeta

    return zeta, nMomentsInInteriorOfMomentSpace, aRecurrence, bRecurrence