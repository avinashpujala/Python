a, a_sub = [],[]
for jj in np.arange(1,11):
    for kk in np.arange(jj):
        a_sub.append(kk)
    a.append(a_sub)
    a_sub = []
    
    