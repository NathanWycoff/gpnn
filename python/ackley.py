import numpy as np

#TODO: Figure out how to copyright this correctly.
def ackley(xx, a=20, b=0.2, c=2*np.pi):
    """
    ACKLEY FUNCTION

    Authors: Sonja Surjanovic, Simon Fraser University
            Derek Bingham, Simon Fraser University
            Nathan Wycoff, Virginia Tech (Python translation; mapping to [0,1]^d)
    Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.

    Copyright 2013. Derek Bingham, Simon Fraser University.

    THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
    FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
    derivative works, such modified software should be clearly marked.
    Additionally, this program is free software; you can redistribute it 
    and/or modify it under the terms of the GNU General Public License as 
    published by the Free Software Foundation; version 2.0 of the License. 
    Accordingly, this program is distributed in the hope that it will be 
    useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
    General Public License for more details.

    For function details and reference information, see:
    http://www.sfu.ca/~ssurjano/

    #########################################################################

    INPUTS:

    xx = c(x1, x2, ..., xd) \in [0,1]^d
    a = constant (optional), with default value 20
    b = constant (optional), with default value 0.2
    c = constant (optional), with default value 2*pi

    #########################################################################
    """
    xx = (2*32.768) * xx - 32.768
    d = len(xx)

    sum1 = sum(pow(xx,2))
    sum2 = sum(np.cos(c*xx))

    term1 = -a * np.exp(-b*np.sqrt(sum1/d))
    term2 = -np.exp(sum2/d)

    y = term1 + term2 + a + np.exp(1)
    return(y)
