from a1_fibtrackmain import *

#%%---------------------5. ABC ROUTINE ------------------------------------------

def ndropped(a, b, c, pID_list):
    """
    figuring out how many dropped in a pair of planes
    """
    lis=[];
    for pID in pID_list:
        nfibs=np.max(MC[pID]);
        fib_rec=np.full((nfibs,nplanes),-1, dtype=int) #-1 means no fibril here, as indices are >= 0
        fib_rec[:,pID]=np.arange(nfibs); #use like fib_rec[fID, pID]
        dz_b, dz_f=increments_back_forward(pID,junk)
        err_table=np.zeros([nfibs,np.max(MC[pID+dz_f])]); #table of errors i,j. Overwritten at each new pID

        #CREATING ERROR TABLES
        for fID in range(nfibs):
            #Isolating the relevant 'patch in morphological components
            if fib_rec[fID,pID]!=-1: # catching nonexistent fibrils, true in pID>0
                cofI=props[pID,fib_rec[fID,pID],0:2]#centroid of fibril in plane
                index=np.ndarray.flatten(md.search_window(cofI, npix/10, npix)).astype('int')
                compare_me=np.delete(np.unique(np.ndarray.flatten(MC[pID+dz_f,index[0]:index[1], index[2]:index[3]]-1) ),0) #find a more neat way to do this. List of indices in next slice to look at.
                for j in compare_me: #going through relevant segments in next slice
                    err_table[fID,j]=err(pID, fib_rec[fID,pID], fib_rec[fID,pID-dz_b], j,dz_b, dz_f, a, b, c)

        #sorted lists of the errors and the pairs of i,j which yield these Errors
        sort_errs=sort_errs=np.sort(err_table, axis=None)
        sort_err_pairs =np.transpose(np.unravel_index(np.argsort(err_table, axis=None), err_table.shape))

        #delete pairs with 0 errors (ie those who are outside the box) and those above the threshold
        delete=np.concatenate((np.where(sort_errs==0)[0],np.where(sort_errs>errorthresh(a,b,c, skip))[0]), axis=0)
        sort_err_pairs=np.delete(sort_err_pairs,delete, axis=0)

        i=0  #Matching up
        while sort_err_pairs.shape[0]>0:
            match=sort_err_pairs[0]  # picks out smallest error match
            fib_rec[match[0], pID+dz_f]=match[1]  # fills in the corresponding fibril recor with this match
            #delete all other occurences of i,j
            deleteme=np.unique(np.ndarray.flatten(np.concatenate((np.argwhere(sort_err_pairs[:,0]==match[0]),np.argwhere(sort_err_pairs[:,1]==match[1]))))).tolist()
            sort_err_pairs=np.delete(sort_err_pairs, deleteme,axis=0)
            i=i+1
        lis.append(nfibs/np.count_nonzero(fib_rec[:,pID+dz_f]<0))
    return lis
def make_abc_map(a, b, c, N, nrepeats=5):
    # filling the heatmap, value by value
    np.savetxt(dirResults+"/abc/values_abc.txt", np.vstack((np.ones(N),b,c)).T) #saves ABC values
    fun_map = np.empty((b.size, c.size))
    for i in range(b.size):
        for j in range(c.size):
            print(f'i,j={i},{j}')
            random_planes=np.random.choice(np.setdiff1d(np.arange(nplanes-1),junk),nrepeats)
            fun_map[i,j] = np.mean(ndropped(a, b[i],c[j], random_planes))
    np.save(dirResults+"/abc/heatmap_abc", fun_map)
def plot_abc_map(a, b, c):
    #PLOTTING THE HEATMAP OF ABC VALUES
    fun_map=np.load(dirResults+"/abc/heatmap_abc.npy")
    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1, xlabel='b', ylabel='c')
    extent = [ b[0], b[-1],  c[0],  c[-1]];
    im = plt.imshow(fun_map.T,extent=extent, origin='lower') #the transpose is because of the row column effect
    fig.colorbar(im);
    plt.title('1 in how many fibrils dropped. a=1')
    plt.savefig(dirResults+"/abc/abc.png")
    plt.show()
    #print ((time_s()-start_time)/60)
def sort_abc(a, b, c):
    fun_map=np.load(dirResults+"/abc/heatmap_abc.npy")
    #SORTING THE VALUES OF B AND C
    sort_pairs=np.vstack(np.unravel_index((-fun_map).argsort(axis=None, kind='mergesort'), fun_map.shape))
    bcSort=np.vstack((b[sort_pairs[0]],c[sort_pairs[1]])).T
    np.savetxt(dirResults+ "/abc/a1_b_c_values_sorted.txt", bcSort)
    return bcSort
def map_through_abc_list(abclist):
    #RUNNING THE BEST VALUES OF ABC AND EXPORTING
    for _ in range (len(abclist)):
        a=1; b,c=abclist[i]
        print(f'Mapping a=1, b={c:.2f}, c={c:.2f}')
        abc_string=f'_a_{a:.2f}_b_{b:.2f}_c_{c:.2f}'
        fibril_mapping(a, b, c, initialise_fibril_record(MC),frfilename='/abc/fibrec'+abc_string)
def main_abc():
    #CREATING HEATMAP OF ABC VALUES
    N=21
    a=1;b=np.linspace(0,3,N);c=b.copy()
    if os.path.isfile(dirResults+'/abc/a1_b_c_values_sorted.txt'):
    # if os.path.isfile('/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-output/abc-dec21/a1_b_c_values_sorted.txt'):
        print ("Importing abc values from previous run")
        bcSort=np.loadtxt(dirResults+ 'abc/a1_b_c_values_sorted.txt')
        # bcSort=np.loadtxt('/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-output/abc-dec21/a1_b_c_values_sorted.txt')
    else:
        print ("ABC Routine")
        print ("Making ABC map")
        make_abc_map(a, b, c, N, nrepeats=10)
        print ("Plotting ABC map")
        plot_abc_map(a, b, c)
        bcSort=sort_abc(a, b, c)
    print ("Running over various ABC values")

    best3Sensible=bcSort[np.all(bcSort<2, axis=1)][0:3]
    worst3=bcSort[-3:]
    abc_ofinterest=np.concatenate((best3Sensible,worst3))
    # map_through_abc_list(best3Sensible)

    start=time.perf_counter()

    li=[]
    for pairs in abc_ofinterest:
        t=threading.Thread(target=fibril_mapping, args=(a, pairs[0], pairs[1],initialise_fibril_record(MC), 1, '/abc/fibrec_a1_bc_'+str(pairs)))
        t.start();li.append(t)
    for t in li:
        t.join()

    finish=time.perf_counter()
    print (f'Finished in {round(finish-start, 5)}s')

# main_abc()
