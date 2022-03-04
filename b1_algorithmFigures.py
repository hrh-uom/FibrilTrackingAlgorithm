from a0_initialise import *
from a1_fibtrackMain import *
import matplotlib.pyplot as plt
#-----------------------------5. ALGORITHM FUNCTION FIG ------------------------

def make_schematic():
    ministack=create_binary_stack(d)
    labels=10*ministack[0]
    pID=0; fID=892

    labels=np.where(MC[pID]==fID+1,fID+1, labels)
    labels.shape
    np.unique(labels)
    cols = ['grey', 'red', 'orange', 'blue', 'yellow', 'purple']
    rgblabel=label2rgb(labels-1, bg_label=-1, colors=cols);
    cofI=props[pID, fID, 0:2]
    xy=md.search_window(cofI, d.npix/10, d.npix)[:,0].tolist();
    recsize=np.ndarray.flatten(np.diff(md.search_window(cofI, d.npix/10, d.npix))).tolist();
    fig1, ax1 = plt.subplots( )
    ax1.imshow(rgblabel, origin='lower', interpolation='nearest')
    # plt.title('fID %i. Plane %i of %i. Size %i' % (fID,pID+1, nplanes, d.npix/10))
    # plt.ylabel('y pix')
    # plt.xlabel('x pix')
    # Create a Rectangle patch
    import matplotlib.patches as patches

    rect=patches.Rectangle((xy[1],xy[0]),recsize[0],recsize[1],linewidth=1,edgecolor='w',facecolor='none')
    rect2=patches.Rectangle((xy[1],xy[0]),recsize[0],recsize[1],linewidth=1,edgecolor='w',facecolor='none')
    # Add the patch to the Axes
    ax1.add_patch(rect)
    ax1.set_title("Plane $p$")
    plt.savefig(d.dirOutputs+'window-schematic1.png');    plt.show()

    fig2, ax2 = plt.subplots( )

    ax2.set_title("Plane $p+1$")

    index=np.ndarray.flatten(md.search_window(cofI, d.npix/10, d.npix)).astype('int')
    compare_me=np.delete(np.unique(np.ndarray.flatten(MC[pID+1,index[0]:index[1], index[2]:index[3]]-1) ),0)
    labels2=10*ministack[1]
    for fID in compare_me:
        labels2=np.where(MC[pID+1]==fID+1,50, labels2)
    rgblabel2=label2rgb(labels2-1, bg_label=-1, colors=cols);
    ax2.imshow(rgblabel2, origin='lower', interpolation='nearest')
    ax2.add_patch(rect2)
    plt.savefig(d.dirOutputs+'window-schematic2.png');    plt.show()
make_schematic()
#%%---------------------------6. ERROR THRESHOLD FIG -------------------------------
def make_errorthresh_fig(skip=1):
    pID=0;prev_i=100; a,b,c=1,1,1 ;dz_b, dz_f=increments_back_forward(pID)
    ni=np.max(MC[pID]);nj=np.max(MC[pID+1])

    fig, ax = plt.subplots(figsize=[8, 6])
    err_grid_all=np.zeros((ni, nj))
    # axins = ax.inset_axes([0.1, 0.56, 0.4, 0.4])
    sizelist=[]
    for i in np.sort(np.random.randint(0,ni,10)):
        cofI=props[pID, i, 0:2]
        size=round(props[pID, i, 5]*d.pxsize)
        index=np.ndarray.flatten(md.search_window(cofI, d.npix/10, d.npix)).astype('int')
        compare_me=np.delete(np.unique(np.ndarray.flatten(MC[pID+dz_f,index[0]:index[1], index[2]:index[3]]-1) ),0) #find a more neat way to do this. List of indices in next slice to look at.
        err_grid_window=np.zeros((1, compare_me.size))

        # for j in range (nj):
        #     error=err(pID, i, 0, j,dz_b, dz_f, a, b, c)
        #     err_grid_all[i,j]=error

        for j in range(len(compare_me)):
            error=err(pID, i, 0, compare_me[j],dz_b, dz_f, a, b, c)
            # print(error)
            err_grid_window[0,j]=error
        # print(cofI, index, compare_me)

        sort_errs=np.sort(err_grid_all, axis=None)
        sort_errs_window=np.sort(err_grid_window, axis=None)
        sort_err_pairs =np.transpose(np.unravel_index(np.argsort(err_grid_all, axis=None), err_grid_all.shape))
        #delete pairs with 0 errors (ie those who are outside the box) and those above the threshold
        delete=np.concatenate((np.where(sort_errs==0)[0],np.where(sort_errs>errorthresh(a,b,c, skip))[0]), axis=0)
        sort_err_pairs=np.delete(sort_err_pairs,delete, axis=0)
        sort_errs_window=sort_errs_window[sort_errs_window>0]
        ax.plot(np.arange(15)+1, sort_errs_window[0:15], label=size)
        # axins.plot(np.arange(20)+1, sort_errs_window[0:20], label=size)

    #Set inset region
    x1, x2, y1, y2 = 1,2,0,1.3
    # axins.set_xlim(x1, x2); axins.set_ylim(y1, y2)
    # axins.set_xticks([1,2]);axins.set_yticks([1, 2])
    # axins.plot([0, 120], [1, 1], '--k')
    # ax.indicate_inset_zoom(axins, edgecolor="black")

    ax.set_xlim(1, 15); ax.set_ylim(0, np.around(ax.yaxis.get_data_interval()[1]*1.2,1))
    ax.plot([0, 120], [1, 1], '--k')
    ax.set_xlabel("Rank of matched pair"); ax.set_ylabel("Error \u03BE")
    ax.set_xticks(np.arange(1, 15, 1))

    #Reorders the labels to be in size order
    handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = zip(*[ (handles[i], labels[i]) for i in sorted(range(len(handles)), key=lambda k: list(map(int,labels))[k])] )
    plt.legend(handles, labels,title='MFD in nm', ncol=2)
    md.create_Directory(d.dirOutputs+'stats')
    plt.savefig(d.dirOutputs+'stats/error_thresh_fig')
    plt.show()
make_errorthresh_fig()
