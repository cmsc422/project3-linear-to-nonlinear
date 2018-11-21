from numpy import *
from pylab import *
from mpl_toolkits.mplot3d import axes3d, Axes3D

#################################
def objective(x):
    return x**2 + 10
    
#################################

def lagrangian(obj, xge):
    # given an optimization problem:
    #    min_x  obj(x)
    #     st    x >= xge
    # or equivalently
    #    min_x  obj(x)
    #     st    x - xge >= 0
    # we convert this into the lagrangian:
    #    max_{alpha >= 0} min_x  obj(x) - alpha(x - xge)
    # if x is big enough then x-xge >= 0, so alpha wants to be zero
    # if x is too small then x-xge <= 0, so alpha --> oo to blow it up
    #
    # we return lagrangian this as a function of x,alpha
    return lambda x,alpha: obj(x) - alpha * (x - xge)


def makePlot(arg, constraint_x_ge=3):
  show1 = show2 = show3 = show4 = False
  if arg == 'all':
    show1 = show2 = show3 = show4 = True
  if arg == 'objective':
    show1 = True
  if arg == 'lagrangian':
    show2 = True
  if arg == 'contour':
    show3 = True
  if arg == 'alpha':
    show4 = True

  xmin = -8
  xmax =  8
  xstep = 0.1
  xvals = arange(xmin, xmax+xstep, xstep)
  
  ymin,ymax = min(0,min(map(objective, xvals))), max(map(objective, xvals))
  if show1:
    plt.figure()
    clf()
    plot(xvals, list(map(objective, xvals)), 'C0-', \
          [constraint_x_ge, constraint_x_ge], [ymin,ymax], 'k:')
    legend(['objective', 'constraint x>=...'])
    axis([xmin,xmax,ymin,ymax])
    #show(False)

  L = lagrangian(objective, constraint_x_ge)

  almin = 0
  almax = 10
  alstep = 1
  alvals = arange(almin, almax+alstep, alstep)
  lx = len(xvals)
  la = len(alvals)
  X  = zeros((len(xvals), len(alvals)))
  Y  = zeros((len(xvals), len(alvals)))
  Z  = zeros((len(xvals), len(alvals)))
  k  = 0
  for i in range(lx):
      for j in range(la):
          X[i,j] = xvals[i]
          Y[i,j] = alvals[j]
          Z[i,j] = L(xvals[i], alvals[j])
  if show2:
    fig = figure(2)
    clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.8, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, 40, zdir='z', offset=Z.min(), cmap=cm.coolwarm)
    #cset = ax.contour(X, Y, Z, zdir='x', offset=xmin, cmap=cm.coolwarm)
    #cset = ax.contour(X, Y, Z, zdir='y', offset=almax, cmap=cm.coolwarm)
  
  A = Z.argmax(axis=1)
  
  xyz = zeros((lx,3))
  for i in range(lx):
      j = Z[i,:].argmax()
      xyz[i,0] = xvals[i]
      xyz[i,1] = alvals[j]
      xyz[i,2] = Z[i,j] + 10
  if show2:
    ax.plot3D(xyz[:,0], xyz[:,1], xyz[:,2], 'k-', linewidth=5)
  
  ayz = zeros((la,3))
  for i in range(la):
      j = Z[:,i].argmin()
      ayz[i,0] = xvals[j]
      ayz[i,1] = alvals[i]
      ayz[i,2] = Z[j,i]
  if show2:
    ax.plot3D(ayz[:,0], ayz[:,1], ayz[:,2], 'C0-', linewidth=5)
    #show(False)

  if show3:
    figure(3)
    clf()
    contour(X,Y,Z,40, cmap=cm.coolwarm)
    plot(xyz[:,0], xyz[:,1], 'k-', linewidth=5)
    plot(ayz[:,0], ayz[:,1], 'C0-', linewidth=5)
    #show(False)


  #################################
  # now take a derivative of the lagrangian with respect to x
  # and we get:
  #    dL/dx = d obj/dx - alpha
  # in the case that objective(x) = x**2 + 10
  # we have dobj/dx = 2x
  # so we get dL/dx = 2x - alpha
  # set this equal to zero and solve for x yields
  #   x = alpha/2
  # because L is convex in x, we know that this is a minimum
  # we can plug this back in to L to get:
  #   L(alpha) = obj(alpha/2) - alpha * ( (alpha/2) - xge )
  # which we can also plot
  
  almin = 0
  almax = 10
  alstep = 0.02
  alvals = arange(almin, almax+alstep, alstep)
  
  if show4:
    figure(4)
    clf()
    plot(alvals, list(map(lambda al: objective(al/2) - al * (al/2 - \
          constraint_x_ge), alvals)), 'r-', label='dual')
    #show(False)
