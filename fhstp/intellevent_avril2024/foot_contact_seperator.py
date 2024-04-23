def foot_contact_seperator(tr_grf, tr_traj):  
    #print(len(tr_grf))
    for idx in range(0, len(tr_grf)):
        grf = tr_grf.loc[idx].copy()
        traj = tr_traj.loc[idx].copy()

        #rand_len = grf.shape[0] - np.argwhere(grf > 0)[-1] 
        
        for i in range(0, grf.shape[0]):
            if grf[i] == 2:
                grf[i] = 0
            elif grf[i] == 4:
                grf[i] = 0
            elif grf[i] == 3:
                grf[i] = 1
        #print(tr_grf)+
        try:
            loc = np.argwhere(grf > 0)
            if loc[0][0] >= 140:
                rand_len_fore = random.randint(15, 125)
            else:
                    rand_len_fore = loc[0][0]

            if len(grf) - loc[-1][0] >= 140:
                rand_len_aft = random.randint(15, 125)
            else:
                rand_len_aft = len(grf)-1
                
            grf = grf[(loc[0][0]-rand_len_fore):(loc[-1][0]+rand_len_aft)] 
            tr_traj.loc[idx] = traj[:, (loc[0][0]-rand_len_fore):(loc[-1][0]+rand_len_aft)] 
            #print(grf)
        except:
            print(idx)
        tr_grf.loc[idx] = grf
        
    return tr_grf, tr_traj