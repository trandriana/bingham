from __future__ import division
import scipy.sparse as ssparse
import scipy.integrate as sint
import scipy.sparse.linalg as slin
import numpy as np
from matplotlib import pylab as pl

eta = .25
B  = 1.5
R  = .25
C = 8	
y0 = B/C

def diff(k, f, typ):#Difference finie centree
	h  = 2*R/k
	return (f[0:k - 1 + typ] - f[1:k + typ])/h  
    	
def bingham1D(Nh, nbiter , tol, r, plot_option):#Solution approchee par ALG2	
	f = C*np.ones(Nh)
	p = np.ones(Nh + 1)
	lbda = p
	lbda0 = np.zeros(Nh + 1)
	y = np.linspace(-R, R, Nh + 1)
	h = 2*R/(Nh + 1)
	I = np.ones(Nh)
	A =  ssparse.spdiags(np.array([I, -2*I, I]), [-1, 0, 1], Nh, Nh).tocsr()
	
	for n in range(nbiter):
		error= np.sqrt(sint.trapz((lbda0 - lbda)**2, y))#integration 
#		par la methode des trapezes
		if (error <= tol): break
		lbda0 = lbda
	
		#Resolution d'un probleme de Dirichlet
		source = f - r*diff(Nh + 1, p, 0) + diff(Nh + 1, lbda, 0)
		u = [0] + list(slin.spsolve(-r*A/(h*h), source, use_umfpack = True)) + [0]
		u = np.array(u)
		
		#Calcul de p
		du = diff(Nh + 1, u, 1)
		v = lbda + r*du
		indic = (B<=abs(v))
		p = indic*(v -sign(v)*B)/(eta + r)
	
		#Mise a jour de lambda
		lbda = lbda + r*(du - p)
		
	print('Nh =', Nh, 'it = ', n	)
	if (plot_option == 'y'): plotme(u, Nh)
	x = np.linspace(-R, R, Nh + 2)
	h = 2*R/(Nh + 1)
	global_error = np.sqrt(sint.trapz((u - uex(x))**2, x))
	
	return [h, global_error, n]

def uex(y):#Solution analytique 
	Cc = -C/eta 
	
	id1 = (y<-y0)
	id2 = (y>y0)
	id3 = (-y0<y)&(y<y0)

	x1 = Cc*(y*(y/2 + y0) - R*(R/2 - y0))
	x2 = Cc*(y*(y/2 - y0) - R*(R/2 - y0))
	x3 = Cc*(-y0*y0/2 - R*(R/2 - y0))
    
	return x1*id1 + x2*id2 + x3*id3
		
def plotme(f, Nh):#Affichage
	y = np.linspace(-R, R)
	yy = np.linspace(-R, R, Nh+2)
	pl.figure(1)
	pl.plot(yy, f, '^-')
	pl.plot(y, uex(y), 'r')
	pl.grid()
	pl.legend([ 'Solution approchee', 'Solution analytique'], 
					loc = 'lower center')
	pl.xlabel('$y$')
	pl.ylabel('Profil de vitesse.')
	pl.show()
	
def sign(v):#signature d'un vecteur v
	idd = (v!=0)
	return v*idd/abs(v + (v == 0))

N, nbiter, tol, r_penalisant = 15, 10000, 0.0001, 1
bingham1D(N, nbiter, tol, r_penalisant, 'y')
