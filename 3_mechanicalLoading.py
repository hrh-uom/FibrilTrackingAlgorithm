import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers import solve
import sympy as sp
import glob
import os
import customFunctions as md
plt.style.use('./mystyle.mplstyle')

#----------------------------------------------------------------------------
#........................... ELASTIC LOADING...............................
#------------------------------------------------------------------------
start_plane, end_plane=0,695
desired_length=1000 #nm


if ('Dropbox' in os.getcwd()):#MY PC
    dirResults=f'/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-output/results_{start_plane}_{end_plane}/'
else:#ON CSF
    dirResults=f'/mnt/fls01-home01/t97721hr/scratch/nuts-and-bolts/results_{start_plane}_{end_plane}'
md.create_Directory(dirResults+'/mechanical')

DSN=0 #dataset number
directories=[dirResults] #Temp
dir_output=dirResults

elasticmodulus = 350  #MPa VanDerRijt2006
vol_frac=0.706  #from Ben - fibril fraction in the image

def load_FTA_data(DSN):
    """
    Loads data from fibril tracking. If path is set above this should work automatically, no need to adjust file names
    """
    d=directories[DSN] ; err=0
    try:
        e_c=np.load(glob.glob(dirResults+f"scaledlengths_{desired_length}*.npy")[0])-1
    except:
        print("No scaledlengths file found"); err=1
    try:
        MFDs=np.load(glob.glob(dirResults+f"fib_MFDs_{desired_length}*.npy")[0])
    except:
        print("No MFD file found") ; err=1
    try:
        areas=np.load(glob.glob(dirResults+f"area_{desired_length}*.npy")[0])
    except:
        print("No areas file found") ; err=1
    if err==0:
        return  e_c, MFDs, areas
e_c, MFDs, areas=load_FTA_data(0)


#%%
#Criticalstrain
fig, ax1 = plt.subplots( )
n, bins, patches = ax1.hist(100*e_c, 50, density=False, weights=areas, facecolor='g', alpha=0.75)
ax1.set_xlabel('Area weighted critical strain (%)')
ax1.set_ylabel('Number')
plt.show()



#%%----------------------------------------------------------------------------
#........................... CHOOSE STRAIN CONDITIONS.....................
#------------------------------------------------------------------------
max_strain=6 /100 #Percent

def e_t(c1, c2, c3, c4, t):
    """
    Oscillating strain function, t is given in minutes
    """
    return  c1 + c2 * sp.cos(c3 * t + c4)

c1, c2, c3, c4=max_strain/2, -max_strain/2, np.pi/(60*2), 0

T=2*np.pi/c3 #Period


t=sp.symbols('t'); y=e_t(c1, c2, c3, c4, t)
lam_y = sp.lambdify(t, 100*y, modules=['numpy'])
t_vals = np.linspace(0, 15*T, 1000)
y_vals = lam_y(t_vals)

fig, ax2=plt.subplots()
ax2.plot(t_vals/60, y_vals)
ax2.set_xlabel("Time (min)")
ax2.set_ylabel("Strain (%)")
plt.savefig(dir_output+'mechanical/straintime')
plt.show()

#%%----------------------------------------------------------------------------
#...........................FIBRIL STRESS....................
#------------------------------------------------------------------------

e_lin=np.linspace(0.0,max_strain,101)  #global strain values

def calculate_fibril_stress(DSN,e_lin, e_c, MFDs, areas, elasticmodulus):
    nfibs=e_c.size
    def F_fib(i, e_lin): #e=global strain, given in nN
        f=elasticmodulus*areas[i]*(e_lin-e_c[i])/(1+e_c[i])
        if f>0:
            return f
        else:
            return 0
    def stress(e_lin): #in GPa
        F=0
        for i in range (nfibs):
            F+=F_fib(i,e_lin)
        return F/np.sum(areas)

    s=[stress(i) for i in e_lin]
    return np.array(s)

sig_s=calculate_fibril_stress(0,e_lin, e_c, MFDs, areas, elasticmodulus)
plt.plot(100*e_lin,sig_s, '-r')
plt.xlabel('Strain (%)')
plt.ylabel('Stress (MPa)')
plt.savefig(dir_output+'mechanical/elastic-response')
plt.show()




#%%----------------------------------------------------------------------------
#...........................  HYSTERESIS /VISSCOELAASTIC CURVE.................
#------------------------------------------------------------------------
e_lin=np.linspace(0.0,max_strain,101)  #global strain values
#
# nu=0.04*1000
# nu=0.0000139
def nu(p_dot_alpha=15.55020565796672, A_f=38965893.12, mu=1.0016e-3):
    """
    The constant term to nultiply the strain rate function
    by in order to get the viscous stress contribution
    Pulled from fluid model
    used like,
    sig_f = mu * d(epsilon)/dt
    """
    return (p_dot_alpha + 2 * mu *A_f)/A_f

nu=543.974 #MPA OUTPUT FROM BENS ANALYSIS
# # nu=100

def find_timepoints_when_singular(timepoints):
    pad = len(max(timepoints, key=len))
    timepoints=np.array([i + [0]*(pad-len(i)) for i in timepoints])
    meetwhere=np.argwhere(timepoints[:,1]==0)[0,0]
    timepoints[meetwhere,1]=timepoints[meetwhere,0]
    return timepoints

t=sp.symbols('t')
timepoints=[solve(strain - e_t(c1, c2, c3, c4, t), t) for strain in e_lin]

if np.any(np.array([len(xi) for xi in timepoints])==1): #Singular
    timepoints=find_timepoints_when_singular(timepoints)

def deps_dt(c2, c3, c4, tt):
    return sp.diff(e_t(c1, c2, c3, c4, t),t).subs(t, tt)

delt_sig_load=np.array([nu*sp.N(deps_dt(c2, c3, c4, tt) )for tt in timepoints[:,0]])
delt_sig_unload=np.array([nu*sp.N(deps_dt(c2, c3, c4, tt) )for tt in timepoints[:,1]])

totalstress_loading=vol_frac*sig_s+(1-vol_frac)*delt_sig_load
totalstress_unloading=vol_frac*sig_s+(1-vol_frac)*delt_sig_unload


plt.plot(100*e_lin,totalstress_loading, label='Loading')
plt.plot(100*e_lin,totalstress_unloading,label='Unloading')
plt.xlabel('Strain (%)')
plt.ylabel('Stress (MPa)')
plt.legend(loc="upper left")
plt.savefig(dir_output+"mechanical/hysteresis')
plt.show()


#%%----------------------------------------------------------------------------
#..........................BIG FIGURE ...................
#------------------------------------------------------------------------
fig,ax = plt.subplots( 2,2, figsize=(11,8))
n, bins, patches = ax[0,0].hist(100*e_c, 50, density=False, facecolor='g', alpha=0.75)
ax[0,0].set_xlabel('Critical strain (%)')
ax[0,0].set_ylabel('Number')
ax[1,0].plot(t_vals/60, y_vals)
ax[1,0].set_xlabel("Time (min)")
ax[1,0].set_ylabel("Strain (%)")
ax[0,1].plot(100*e_lin,sig_s, '-r')
ax[0,1].set_xlabel('Strain (%)')
ax[0,1].set_ylabel('Stress (MPa)')
ax[1,1].plot(100*e_lin,totalstress_loading, label='Loading')
ax[1,1].plot(100*e_lin,totalstress_unloading,label='Unloading')
ax[1,1].set_xlabel('Strain (%)')
ax[1,1].set_ylabel('Stress (MPa)')

x,y=0,1.05
ax[0,0].annotate('A', xy=(x,y), xycoords="axes fraction", fontsize=30, color='b')
ax[0,1].annotate('B', xy=(x,y), xycoords="axes fraction", fontsize=30, color='b')
ax[1,0].annotate('C', xy=(x,y), xycoords="axes fraction", fontsize=30, color='b')
ax[1,1].annotate('D', xy=(x,y), xycoords="axes fraction", fontsize=30, color='b')

fig.tight_layout()
plt.savefig(dir_output+'mechanical/bigfig')
plt.show()

#%%----------------------------------------------------------------------------
#...........................QUANTIFICATION OF HYSTERESIS....................
#------------------------------------------------------------------------
from scipy.integrate import simps
# Compute the area using the composite Simpson's rule.
load_work=float(simps(totalstress_loading,dx=e_lin[1]))
unload_work=float(simps(totalstress_unloading,dx=e_lin[1]))
delta_work=load_work-unload_work


#%%----------------------------------------------------------------------------
#...........................TEXT FILE RESULTS....................
#------------------------------------------------------------------------
headings=['fluid stress (chosen)', 'elastic modulus (Mpa)', 'c1', 'c2', 'c3', 'c4', 'load work per unit vol (kPa)', 'unload work per unit vol(kPa)', 'delta work /unit vol(kPa)', 'fractional work']
valuessymb=np.array([nu,elasticmodulus,c1, c2, c3, c4, 1000*load_work, 1000*unload_work, 1000*delta_work, delta_work/load_work] )

import csv
with open(dir_output+'mechanical/mechloading_params.csv', mode='w') as params:
    params = csv.writer(params, delimiter=',')
    params.writerow(headings)
    params.writerow(valuessymb)
