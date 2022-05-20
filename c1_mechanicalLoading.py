import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers import solve
import pandas as pd
import sympy as sp
import glob
import os
import csv
import customFunctions as md
from scipy.integrate import simps
plt.style.use('./mystyle.mplstyle')

#----------------------------------------------------------------------------
#........................... ELASTIC LOADING...............................
#------------------------------------------------------------------------
datanames  =       ['9am-1R', '9am-2L', '9am-4R']
directories=    [   f'/Users/user/dbox/2-mechanics-model/csf-output/{datanames[0]}/a1.0_b1.0_c1.0/thresh1/',
                    f'/Users/user/dbox/2-mechanics-model/csf-output/{datanames[1]}/a1.0_b1.0_c1.0_T1.0/',
                    f'/Users/user/dbox/2-mechanics-model/csf-output/{datanames[2]}/a1.0_b1.0_c1.0_T1.0/'  ]
dir_output=         '/Users/user/dbox/2-mechanics-model/csf-output/mechanical-all/'; md.create_Directory(dir_output)
vol_frac=0.706  #from Ben - fibril fraction in the image

def load_FTA_data(DSN):
    """
    Loads data from fibril tracking. If path is set above this should work automatically, no need to adjust file names
    """
    d=directories[DSN] ; err=0
    try:
        e_c=np.load(glob.glob(d+f'stats/scaledlengths*')[0])-1
    except:
        print("No scaledlengths file found"); err=1
    try:
        MFDs=np.load(glob.glob(d+'mfds*')[0])
    except:
        print("No MFD file found") ; err=1
    try:
        areas=np.load(glob.glob(d+f'area*')[0])
    except:
        print("No areas file found") ; err=1
    try:
        VF=pd.read_csv(d+f'stats/VF.csv').VF_raw.to_numpy()
    except:
        print ('No  volume Fraction found'); err=1
    if err==0:
        return  e_c, MFDs, areas,VF

d0=pd.DataFrame(load_FTA_data(0)).T
d1=pd.DataFrame(load_FTA_data(1)).T
d2=pd.DataFrame(load_FTA_data(2)).T

d=pd.concat([d0, d1, d2], axis=1).set_axis([f'ec0', 'mfd0', 'a0','VF0', 'ec1', 'mfd1', 'a1','VF1','ec2', 'mfd2', 'a2' , 'VF2',], axis=1)


d
#%% MFDs

fig, ax = plt.subplots(1,3 , figsize=(20, 4))

n, bins, patches = ax[0].hist(d.mfd0, 50, density=False, facecolor='g', alpha=0.75)
n, bins, patches = ax[1].hist(d.mfd1, 50, density=False, facecolor='g', alpha=0.75)
n, bins, patches = ax[2].hist(d.mfd2, 50, density=False, facecolor='g', alpha=0.75)

[ax[i].set_xlabel('MFD') for i in range(3)];[ax[i].set_ylabel('Number') for i in range (3)]
plt.show()

#%%------------------Criticalstrain
fig, ax = plt.subplots(1,3 , figsize=(20, 6))

n, bins, patches = ax[0].hist(100*d.ec0, 50, density=False, weights=d.a0, facecolor='g', alpha=0.75)
n, bins, patches = ax[1].hist(100*d.ec1, 50, density=False, weights=d.a1, facecolor='g', alpha=0.75)
n, bins, patches = ax[2].hist(100*d.ec2, 50, density=False, weights=d.a2, facecolor='g', alpha=0.75)

[ax[i].set_xlabel('Area weighted critical strain (%)') for i in range(3)];[ax[i].set_ylabel('Number') for i in range (3)]
plt.show()



#%%--------------------------- STRAIN CONDITIONS----------------------
max_strain=6 /100 #Percent
c1, c2, c3, c4=max_strain/2, -max_strain/2, np.pi/(60*2), 0 ; T=2*np.pi/c3 #Period

def e_t(c1, c2, c3, c4, t):
    """
    Oscillating strain function, t is given in minutes
    """
    return  c1 + c2 * sp.cos(c3 * t + c4)
def plot_strain_conditions():
    t=sp.symbols('t'); y=e_t(c1, c2, c3, c4, t)
    lam_y = sp.lambdify(t, 100*y, modules=['numpy'])
    t_vals = np.linspace(0, 15*T, 1000)
    y_vals = lam_y(t_vals)

    fig, ax2=plt.subplots()
    ax2.plot(t_vals/60, y_vals)
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Strain (%)")
    plt.savefig(dir_output+'straintime')
    plt.show()
plot_strain_conditions()

#%%-------------ELASTIC RESPONSE---------------------------------------------------------------
#...........................FIBRIL STRESS....................
#------------------------------------------------------------------------

def plot_elastic_response(E):
    """
    E= Youngs / Elastic modulus
    """
    e_lin=np.linspace(0.0,max_strain,101)  #global strain values

    def calculate_fibril_stress(DSN,e_lin, e_c, MFDs, areas, E):
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

    sig_s0=calculate_fibril_stress(0,e_lin, d.ec0, d.mfd0, d.a0, E)
    sig_s1=calculate_fibril_stress(0,e_lin, d.ec1, d.mfd1, d.a1, E)
    sig_s2=calculate_fibril_stress(0,e_lin, d.ec2, d.mfd2, d.a2, E)
    mean  = np.mean([sig_s0, sig_s1, sig_s2], axis=0)
    sd =np.std([sig_s0, sig_s1, sig_s2], axis=0)
    # plt.plot(100*e_lin,sig_s0, '-r', label=datanames[0])
    # plt.plot(100*e_lin,sig_s1, '-b', label=datanames[1])
    # plt.plot(100*e_lin,sig_s2, '-g', label=datanames[2])
    plt.plot(100*e_lin,mean, '--k', label='mean')
    plt.fill_between(100*e_lin, mean-sd, mean+sd, label = 'standard deviation', color='k', alpha=0.2)

    plt.xlabel('Tensile strain (%)') ; plt.ylabel('Tensile stress (MPa)')
    plt.savefig(dir_output+'elastic-response'); plt.legend() ; plt.show()
E = 350 #MPa VanDerRijt2006

plot_elastic_response(E)

# %%
"""

#%%-------------------------HYSTERESIS /VISSCOELAASTIC CURVE--------------
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

def loadingunloadingstress(sig_s):
    totalstress_loading=vol_frac*sig_s+(1-vol_frac)*delt_sig_load #REMEBER TO CHANGE THIS VF!!!
    totalstress_unloading=vol_frac*sig_s+(1-vol_frac)*delt_sig_unload
    return totalstress_loading, totalstress_unloading

plt.plot(100*e_lin,loadingunloadingstress(sig_s0)[0], '-r', label=datanames[0])
plt.plot(100*e_lin,loadingunloadingstress(sig_s0)[1], '-r')

plt.plot(100*e_lin,loadingunloadingstress(sig_s1)[0], '-b', label=datanames[1])
plt.plot(100*e_lin,loadingunloadingstress(sig_s1)[1], '-b')

plt.plot(100*e_lin,loadingunloadingstress(sig_s2)[0], '-g', label=datanames[2])
plt.plot(100*e_lin,loadingunloadingstress(sig_s2)[1], '-g')

plt.xlabel('Strain (%)')
plt.ylabel('Stress (MPa)')
plt.legend(loc="upper left")
plt.savefig(dir_output+'hysteresis')
plt.show()


#%%----------------------------------------------------------------------------
#..........................BIG FIGURE ...................
#------------------------------------------------------------------------
def bigfig():
    fig,ax = plt.subplots( 2,2, figsize=(11,8))
    n, bins, patches = ax[0,0].hist(100*d.ec0, 50, density=False, facecolor='g', alpha=0.75)
    ax[0,0].set_xlabel('Critical strain (%)')
    ax[0,0].set_ylabel('Number')
    ax[1,0].plot(t_vals/60, y_vals)
    ax[1,0].set_xlabel("Time (min)")
    ax[1,0].set_ylabel("Strain (%)")
    ax[0,1].plot(100*e_lin,sig_s, '-r')
    ax[0,1].set_xlabel('Strain (%)')
    ax[0,1].set_ylabel('Stress (MPa)')
    ax[1,1].plot(100*e_lin,loadingunloadingstress(sig_s0)[0], label='Loading')
    ax[1,1].plot(100*e_lin,loadingunloadingstress(sig_s0)[1],label='Unloading')
    ax[1,1].set_xlabel('Strain (%)')
    ax[1,1].set_ylabel('Stress (MPa)')

    x,y=0,1.05
    ax[0,0].annotate('A', xy=(x,y), xycoords="axes fraction", fontsize=30, color='b')
    ax[0,1].annotate('B', xy=(x,y), xycoords="axes fraction", fontsize=30, color='b')
    ax[1,0].annotate('C', xy=(x,y), xycoords="axes fraction", fontsize=30, color='b')
    ax[1,1].annotate('D', xy=(x,y), xycoords="axes fraction", fontsize=30, color='b')

    fig.tight_layout()
    plt.savefig(dir_output+'bigfig')
    plt.show()

#%%----------------------------------------------------------------------------
#...........................QUANTIFICATION OF HYSTERESIS....................
#------------------------------------------------------------------------

def calculate_work_done(sig_s):
    load_work=float(simps(loadingunloadingstress(sig_s)[0],dx=e_lin[1]))
    unload_work=float(simps(loadingunloadingstress(sig_s)[1],dx=e_lin[1]))
    delta_work=load_work-unload_work
    return load_work, unload_work, delta_work


#%%----------------------------------------------------------------------------
#...........................TEXT FILE RESULTS....................
#------------------------------------------------------------------------
def write_to_file():

    headings=['dataset','fluid stress (chosen)', 'elastic modulus (Mpa)', 'c1', 'c2', 'c3', 'c4', 'load work per unit vol (kPa)', 'unload work per unit vol(kPa)', 'delta work /unit vol(kPa)', 'fractional work']

    valuessymb0=np.array([datanames[0],nu,elasticmodulus,c1, c2, c3, c4, 1000*calculate_work_done(sig_s0)[0], 1000*calculate_work_done(sig_s0)[1], 1000*calculate_work_done(sig_s0)[2], calculate_work_done(sig_s0)[2]/calculate_work_done(sig_s0)[0]] )
    valuessymb1=np.array([datanames[1],nu,elasticmodulus,c1, c2, c3, c4, 1000*calculate_work_done(sig_s1)[0], 1000*calculate_work_done(sig_s1)[1], 1000*calculate_work_done(sig_s1)[2], calculate_work_done(sig_s1)[2]/calculate_work_done(sig_s1)[0]] )
    valuessymb2=np.array([datanames[2],nu,elasticmodulus,c1, c2, c3, c4, 1000*calculate_work_done(sig_s2)[0], 1000*calculate_work_done(sig_s2)[1], 1000*calculate_work_done(sig_s2)[2], calculate_work_done(sig_s2)[2]/calculate_work_done(sig_s2)[0]] )

    with open(dir_output+'mechloading_params.csv', mode='w') as params:
        params = csv.writer(params, delimiter=',')
        params.writerow(headings)
        params.writerow(valuessymb0)
        params.writerow(valuessymb1)
        params.writerow(valuessymb2)
write_to_file()
