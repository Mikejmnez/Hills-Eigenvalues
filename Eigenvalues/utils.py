"""
Contains useful information about exceptional points of some shear flows of interest
"""
import numpy as _np


def cosine_shear(n=1):
	"""
	returns the location of EPs associated with a shear flow of multiplicity n.
	"""

	if n==1:
		qs_a = [2.938438, 32.942942, 95.613113, 190.950950, 318.959459, 479.636636,
			  	672.983483, 898.998498, 1157.684684, 1449.039039, 1773.061561, 2129.755255,
			  	2519.69, 2941.22, 3395.896, 3883.9679
		]
		# odd eigenvalues too
		qs_b = [0

		]

	else:
		qs = 0

	return qs
