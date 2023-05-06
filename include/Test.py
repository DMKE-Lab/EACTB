import numpy as np
import scipy


def get_hits(vec, test_pair, top_k=(1, 10, 50, 100)):
        Lvec = np.array([vec[e1] for e1, e2 in test_pair])
        Rvec = np.array([vec[e2] for e1, e2 in test_pair])
        sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
        MR = {'L_R': 0, 'R_L': 0}
        MRR = {'L_R': 0.0, 'R_L': 0.0}
        top_lr = [0] * len(top_k)
        for i in range(Lvec.shape[0]):
                rank = sim[i, :].argsort()
                rank_index = np.where(rank == i)[0][0]
                MR['L_R'] += rank_index + 1
                MRR['L_R'] += 1.0 / (rank_index + 1)
                for j in range(len(top_k)):
                        if rank_index < top_k[j]:
                                top_lr[j] += 1
        top_rl = [0] * len(top_k)
        MRR_L = 0
        for i in range(Rvec.shape[0]):
                rank = sim[:, i].argsort()
                rank_index = np.where(rank == i)[0][0]
                MR['R_L'] += rank_index + 1
                MRR['R_L'] += 1.0 / (rank_index + 1)
                for j in range(len(top_k)):
                        if rank_index < top_k[j]:
                                top_rl[j] += 1
        print('For each left:')
        for i in range(len(top_lr)):
                print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
        print('MR: %.2f' % (MR['L_R'] / len(test_pair)))
        print('MRR: %.2f' % (MRR['L_R'] / len(test_pair)))
        print('For each right:')
        for i in range(len(top_rl)):
                print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
        print('MR: %.2f' % (MR['R_L'] / len(test_pair)))
        print('MRR: %.2f' % (MRR['R_L'] / len(test_pair)))

