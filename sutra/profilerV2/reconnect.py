from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
from sutra.logger import message


def get_connected_struct_from_mst(cenx, ceny, filnum, distance_tol=1000):
    # create the 2D array
    sparse_arr = np.zeros((len(cenx), len(cenx)))

    for i in range(len(sparse_arr)):
        for j in range(i+1, len(sparse_arr)):
            dist = np.sqrt((cenx[i]-cenx[j])**2 + (ceny[i]-ceny[j])**2)
            sparse_arr[i,j] = dist

    # get minimum spanning tree
    X = csr_array(np.around(sparse_arr))
    Tcsr = minimum_spanning_tree(X)
    Y = Tcsr.toarray()


    # get branched indices
    branches_list = []
    for m,i in zip(Y, range(len(Y))):
        inds = np.where(m>0.0001)[0]
        if(len(inds)>1):
            branches_list.append([i, *inds])

    # filter branches based on distance
    branch_dist = []
    filt_branch_list = []
    filt_branch_num = []
    for el in branches_list:
        bd = np.sqrt((cenx[el] - cenx[el[0]])**2 + (ceny[el] - ceny[el[0]])**2)
        if(np.max(bd)<distance_tol):
            branch_dist.append(bd[1:])
            filt_branch_list.append(el)
            filt_branch_num.append(filnum[el])

    # print(branch_dist)
    # print(filt_branch_list)
    # print(filt_branch_num)

    # get connected sets from branch data
    connected_set = [np.unique(filt_branch_num[0])]
    for el in filt_branch_num[1:]:
        ue = np.unique(el)
        newflag = 1
        inds = []
        for v in ue:
            tmp = [np.any(sets == v) for sets in connected_set]
            if(np.any(tmp)):
                ind = np.where(tmp)[0][0]
                # connected_set[ind] = np.unique(np.concatenate((connected_set[ind], ue)))
                inds.append(ind)
                newflag = 0
                # break
        
        if(len(np.unique(inds))==1): connected_set[inds[0]] = np.unique(np.concatenate((connected_set[inds[0]], ue)))
        if(len(np.unique(inds))>1):
            # print(np.unique(inds))
            inds = np.array(np.flip(np.sort(inds)), dtype='int')
            mergedlist = np.unique(np.concatenate([connected_set[i] for i in inds]))
            for i in inds:   connected_set.pop(i)
            connected_set.append(mergedlist)
        if(newflag ==1):
            connected_set.append(ue)

    # add isolated filaments to connected sets
    flat_conn = [el for row in connected_set for el in row ]
    for i in np.unique(filnum):
        if(not np.any(flat_conn==i)): connected_set.append([i])

    message(f"connected filaments skeletons : {len(connected_set)}", type=1)

    # renumber filaments
    newfilnum = np.copy(filnum)
    i = 0
    for el in connected_set:
        for v in el:
            newfilnum[np.where(filnum == v)[0]] = i

        i+=1


    return(newfilnum, filt_branch_list , Y)


