import cplex
import time
import networkx as nx

I = 1000000000 # a very large number
M = 0 # VM
S = 0 # service
V = 0 # VNF
alpha_m = [] # cost to activate one VM
beta_m = [] # cost of VM capacity per unit
c_m = [] # maximum capacity of VM
lambda_s_v = [] # arrival rate of VNF v in service s
g_s_v = [] # whether service s includes VNF v
t_s = [] # maximum admissible delay of service s
d_s_v = [] # maximum admissible number of divisions of VNF v in service s
l_v = [] # an attribute of VNF type

def cal_total_arr(su, v_su, k_su): # selected su, VNF type of selected su, k for each s in selected su
    res = 0
    for index_s in range(len(su)):
        res += lambda_s_v[su[index_s]][v_su] * k_su[index_s]
    return res

def setup_VM_scale(P, used_capacity_m, l_v_m, x_s_v_m, k_s_v_m, M, S, V, beta_m, c_m, t_s):
    P.objective.set_sense(P.objective.sense.minimize)
    #define a_m
    a_m = []
    #objective
    var = []
    lb = []
    ub = []
    types = []
    my_obj = []
    for m in range(M):
        if used_capacity_m[m] > 0:
            a = beta_m[m]
        else:
            a = 0
        var_m = "a_" + str(m)
        var.append(var_m)
        lb.append(0)
        ub.append(c_m[m])
        types.append("C")
        my_obj.append(a)
        a_m.append(var_m)
    P.variables.add(names=var, lb=lb, ub=ub, types=types, obj=my_obj)
    #define t_m
    t_m = []
    var = []
    lb = []
    types = []
    for m in range(M):
        var_m = "t_m_" + str(m)
        var.append(var_m)
        lb.append(0)
        types.append("C")
        t_m.append(var_m)
    P.variables.add(names=var, lb=lb, types=types)

    #define b_m
    b_m = []
    var = []
    lb = []
    types = []
    for m in range(M):
        var_m = "b_m_" + str(m)
        var.append(var_m)
        lb.append(0)
        types.append("C")
        b_m.append(var_m)
    P.variables.add(names=var, lb=lb, types=types)

    #define f_m
    f_m = []
    var = []
    lb = []
    types = []
    for m in range(M):
        var_m = "f_" + str(m)
        var.append(var_m)
        lb.append(0)
        types.append("C")
        f_m.append(var_m)
    P.variables.add(names=var, lb=lb, types=types)

    #constraints
    #(8) a_m >= used_m
    for m in range(M):
        ind_8 = [a_m[m]]
        val_8 = [1]
        expr = cplex.SparsePair(ind=ind_8, val=val_8)
        P.linear_constraints.add(lin_expr=[expr], senses=["G"], rhs=[used_capacity_m[m]])
    '''
    t_m >= 1 / (a_m / l_v) - used_m
    b_m = a_m / l_v - used_m
    t_m >= 1 / b_m, t_m * b_m >= 1
    f_m = t_m + b_m
    f_m^2 >= t_m^2 + b_m^2 + 2
    '''
    #(9) b_m = a_m / l_v - used_m
    for m in range(M):
        if used_capacity_m[m] > 0:
            ind_9 = [a_m[m], b_m[m]]
            val_9 = [1/l_v_m[m], -1]
            expr = cplex.SparsePair(ind=ind_9, val=val_9)
            P.linear_constraints.add(lin_expr=[expr], senses=["E"], rhs=[used_capacity_m[m]])
    #(17b) f_m = t_m + b_m
    for m in range(M):
        if used_capacity_m[m] > 0:
            ind_17b = [t_m[m], b_m[m], f_m[m]]
            val_17b = [1, 1, -1]
            expr = cplex.SparsePair(ind=ind_17b, val=val_17b)
            P.linear_constraints.add(lin_expr=[expr], senses=["E"], rhs=[0])
    #(18b) f_m^2 - t_m^2 - b_m^2 >= 2
    for m in range(M):
        if used_capacity_m[m] > 0:
            q = cplex.SparseTriple(ind1=[t_m[m], b_m[m], f_m[m]], ind2=[t_m[m], b_m[m], f_m[m]], val=[1, 1, -1])
            P.quadratic_constraints.add(quad_expr=q, sense="L", rhs=-2)
    #(10d)
    for s in range(S):
        ind_10d = []
        val_10d = []
        for v in range(V):
            for m in range(M):
                if x_s_v_m[s][v][m] == 1:
                    ind_10d.append(t_m[m])
                    val_10d.append(k_s_v_m[s][v][m])
        expr = cplex.SparsePair(ind=ind_10d, val=val_10d)
        P.linear_constraints.add(lin_expr=[expr], senses=["L"], rhs=[t_s[s]])

def VM_scale(M, S, V, used_capacity_m, beta_m, c_m, l_v_m, x_s_v_m, k_s_v_m, t_s):
    P = cplex.Cplex()
    setup_VM_scale(P, used_capacity_m, l_v_m, x_s_v_m, k_s_v_m, M, S, V, beta_m, c_m, t_s)
    P.set_results_stream(None)
    P.set_warning_stream(None)
    P.write("VM_scale.lp")
    try:
        print("solving..")
        P.solve()
        return P.solution.get_objective_value()
    except cplex.exceptions.errors.CplexSolverError:
        return 0

def flow_partition(M, S, V, alpha_m, beta_m, c_m, g_s_v, lambda_s_v, t_s, d_s_v, l_v):
    # initial settings
    s_v = []
    s_s_v = []
    s_union = []
    v_s_union = []
    k_su_s = []
    m_s_union = []
    deleted_pairs = []
    for v in range(V):
        s_v.append([])
        s_s_v.append([])
    for s in range(S):
        for v in range(V):
            if g_s_v[s][v] > 0:
                s_v[v].append(s)
    for v in range(V):
        s_s_v[v].append(s_v[v][:])
    for v in range(V):
        for ssv in s_s_v[v]:
            s_union.append(ssv[:])
            v_s_union.append(v)
    for index_su in range(len(s_union)):
        m_s_union.append(I)
        k_su_s.append([])
        for s in range(len(s_union[index_su])):
            k_su_s[index_su].append(1)

    len_s_union = len(s_union)
    map_selected_su = {}  # index number of original su : index number of newly created su
    times_su = []  # times that su has been selected
    for index_su in range(len(s_union)):
        times_su.append(0)
    times_s_v = []  # times that service s VNF v has been selected
    for s in range(S):
        times_s_v.append([])
        for v in range(V):
            times_s_v[s].append(1) # number of parts.

    # algorithm start
    step = 1
    while True:
        if step == 1:
            bi_graph = nx.Graph()
            total_cap_su = []
            for index_su in range(len(s_union)):
                total_cap_su.append(cal_total_arr(s_union[index_su], v_s_union[index_su], k_su_s[index_su]))
                for m in range(M):
                    if total_cap_su[index_su] < c_m[m]:
                        weight = alpha_m[m] + beta_m[m] * total_cap_su[index_su]
                        weight_in_hungarian = I - weight
                        bi_graph.add_edge('su_' + str(index_su), 'VM_' + str(m), weight=weight_in_hungarian)
            step = 2
        # step 2
        if step == 2:
            edges = nx.max_weight_matching(bi_graph, maxcardinality=False)
            if len(edges) == len(s_union):
                step = 4
            else:
                step = 3
        # step 3
        if step == 3:
            # decide which su to divide
            min_t_su = []
            for index_su in range(len(s_union)):
                min_t_su.append(I)

            max_appro_cap_su = 0
            for index_su in range(len(s_union)):
                for s in s_union[index_su]:
                    count = 0
                    for v in range(V):
                        if g_s_v[s][v] == 1:
                            count += 1
                    min_t_su[index_su] = min(min_t_su[index_su], t_s[s] / count)
                appro_cap_su = l_v[v_s_union[index_su]] * (1 / min_t_su[index_su] + cal_total_arr(s_union[index_su], v_s_union[index_su], k_su_s[index_su]))
                if appro_cap_su > max_appro_cap_su:
                    max_appro_cap_su = appro_cap_su
                    selected_su_center = index_su
            selected_su = selected_su_center
            times_su[selected_su] += 1
            if times_su[selected_su] > 1000000: # restrict the times that one su can be selected
                return [-1, -1]
            else:
                # decide which s in selected_su to divide
                max_appro_cap_s = -1
                for index_s in range(len(s_union[selected_su])):
                    if (selected_su not in map_selected_su) and times_s_v[s_union[selected_su][index_s]][v_s_union[selected_su]] >= d_s_v[s_union[selected_su][index_s]][v_s_union[selected_su]]: # if times_s_v = d_s_v, selected s cannot in the last su created for VNF v
                        continue
                    count = 0
                    for v in range(V):
                        if g_s_v[s_union[selected_su][index_s]][v] == 1:
                            count += 1
                    appro_cap_s = l_v[v_s_union[selected_su]] * (1 / (t_s[s_union[selected_su][index_s]] / count) + k_su_s[selected_su][index_s] * lambda_s_v[s_union[selected_su][index_s]][v_s_union[selected_su]])
                    if appro_cap_s > max_appro_cap_s:
                        max_appro_cap_s = appro_cap_s
                        selected_s_center = index_s
                if max_appro_cap_s == -1: # no s in selected su can be divided, put the whole s in another su
                    if len(s_union[selected_su]) == 1: # the last s in selected su
                        return [-2, -2]
                    else:
                        max_appro_cap_s_1 = -1
                        for index_s in range(len(s_union[selected_su])):
                            count = 0
                            for v in range(V):
                                if g_s_v[s_union[selected_su][index_s]][v] == 1:
                                    count += 1
                            appro_cap_s = l_v[v_s_union[selected_su]] * (1 / (t_s[s_union[selected_su][index_s]] / count) + k_su_s[selected_su][index_s] * lambda_s_v[s_union[selected_su][index_s]][v_s_union[selected_su]])
                            if appro_cap_s > max_appro_cap_s_1:
                                max_appro_cap_s_1 = appro_cap_s
                                selected_s_center_1 = index_s
                        selected_s = selected_s_center_1
                        if selected_su in map_selected_su: # if there is created su for this v, put the whole selected_s to that su
                            s_union[map_selected_su[selected_su]].append(s_union[selected_su][selected_s])
                            del s_union[selected_su][selected_s]
                            k_su_s[map_selected_su[selected_su]].append(k_su_s[selected_su][selected_s])
                            del k_su_s[selected_su][selected_s]
                        else: # put the whole su to a newly created su
                            if len(s_union) >= M:
                                return [-3, -3]
                            else:
                                s_union.append([s_union[selected_su][selected_s]])
                                del s_union[selected_su][selected_s]
                                v_s_union.append(v_s_union[selected_su])
                                m_s_union.append(I)
                                k_su_s.append([k_su_s[selected_su][selected_s]])
                                times_su.append(0)
                                del k_su_s[selected_su][selected_s]
                                map_selected_su[selected_su] = len_s_union
                                len_s_union = len(s_union)
                else:
                    selected_s = selected_s_center
                    # decide how to divide selected_s
                    # define delta
                    min_delta = I
                    for index_s in range(len(s_union[selected_su])):
                        delta = k_su_s[selected_su][index_s] * lambda_s_v[s_union[selected_su][index_s]][v_s_union[selected_su]] / d_s_v[s_union[selected_su][index_s]][v_s_union[selected_su]]
                        if delta < min_delta:
                            min_delta = delta
                    # update s_union and k_su_s
                    # for selected_su that has been selected previously, move the part to created su
                    if selected_su in map_selected_su:
                        k_su_s[selected_su][selected_s] -= min_delta / lambda_s_v[s_union[selected_su][selected_s]][v_s_union[selected_su]] # revise the proportion of current part
                        flag_same_s = False
                        # check whether there includes selected_s in new su
                        for s in s_union[map_selected_su[selected_su]]:
                            if s == s_union[selected_su][selected_s]:
                                flag_same_s = True
                                break
                        if flag_same_s: # there exists selected_s
                            k_su_s[map_selected_su[selected_su]][s_union[map_selected_su[selected_su]].index(s)] += min_delta / lambda_s_v[s_union[selected_su][selected_s]][v_s_union[selected_su]]
                        else: # there is no selected_s
                            s_union[map_selected_su[selected_su]].append(s_union[selected_su][selected_s])
                            k_su_s[map_selected_su[selected_su]].append(min_delta / lambda_s_v[s_union[selected_su][selected_s]][v_s_union[selected_su]])
                            times_s_v[s_union[selected_su][selected_s]][v_s_union[selected_su]] += 1
                    # for selected_su that has not been selected previously
                    else:
                        if len(s_union) >= M:
                            return [-3, -3]
                        else:
                            times_s_v[s_union[selected_su][selected_s]][v_s_union[selected_su]] += 1
                            s_union.append([s_union[selected_su][selected_s]])
                            v_s_union.append(v_s_union[selected_su])
                            m_s_union.append(I)
                            times_su.append(0)
                            k_su_s[selected_su][selected_s] -= min_delta / lambda_s_v[s_union[selected_su][selected_s]][v_s_union[selected_su]]
                            k_su_s.append([min_delta / lambda_s_v[s_union[selected_su][selected_s]][v_s_union[selected_su]]])
                            map_selected_su[selected_su] = len_s_union
                            len_s_union = len(s_union)
                step = 1
        # step 4
        if step == 4:
            used_capacity_m = []
            l_v_m = []
            x_s_v_m = []
            k_s_v_m = []
            for m in range(M):
                used_capacity_m.append(0)
                l_v_m.append(I)
            for s in range(S):
                x_s_v_m.append([])
                k_s_v_m.append([])
                for v in range(V):
                    x_s_v_m[s].append([])
                    k_s_v_m[s].append([])
                    for m in range(M):
                        x_s_v_m[s][v].append(0)
                        k_s_v_m[s][v].append(0)
            for n1, n2 in edges:
                if n1.startswith('su'):
                    params = n1.split('_')
                    assert len(params) == 2
                    element_in_s_union = int(params[-1])
                    params = n2.split('_')
                    assert len(params) == 2
                    m_element_in_s_union = int(params[-1])
                else:
                    params = n2.split('_')
                    assert len(params) == 2
                    element_in_s_union = int(params[-1])
                    params = n1.split('_')
                    assert len(params) == 2
                    m_element_in_s_union = int(params[-1])

                used_capacity_m[m_element_in_s_union] = cal_total_arr(s_union[element_in_s_union], v_s_union[element_in_s_union], k_su_s[element_in_s_union])
                l_v_m[m_element_in_s_union] = l_v[v_s_union[element_in_s_union]]
                m_s_union[element_in_s_union] = m_element_in_s_union
                for i in range(len(s_union[element_in_s_union])):
                    x_s_v_m[s_union[element_in_s_union][i]][v_s_union[element_in_s_union]][m_element_in_s_union] = 1
                    k_s_v_m[s_union[element_in_s_union][i]][v_s_union[element_in_s_union]][m_element_in_s_union] = k_su_s[element_in_s_union][i]

            total_cost = VM_scale(M, S, V, used_capacity_m, beta_m, c_m, l_v_m, x_s_v_m, k_s_v_m, t_s)
            if total_cost > 0:
                for m in range(M):
                    if used_capacity_m[m] > 0:
                        total_cost += alpha_m[m]
                return [total_cost, len(s_union)]
            else:
                step = 5

        # step 5
        if step == 5:
            min_remain_cap_m = I
            for m in range(M):
                remain_cap_m = c_m[m] - used_capacity_m[m]
                if used_capacity_m[m] > 0 and remain_cap_m < min_remain_cap_m:
                    min_remain_cap_m = remain_cap_m
                    deleted_m = m
            deleted_su = m_s_union.index(deleted_m)

            deleted_pairs.append([deleted_su, deleted_m])
            bi_graph = nx.Graph()
            for index_su in range(len(s_union)):
                for m in range(M):
                    if total_cap_su[index_su] < c_m[m] and [index_su, m] not in deleted_pairs:
                        weight = alpha_m[m] + beta_m[m] * total_cap_su[index_su]
                        weight_in_hungarian = I - weight
                        bi_graph.add_edge('su_' + str(index_su), 'VM_' + str(m), weight=weight_in_hungarian)
            step = 2

if __name__ == "__main__":
    I = 1000000000
    # realistic scenario
    M = 40
    S = 5
    V = 17
    g_s_v = [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]  # whether service s includes VNF v
    t_s = []  # maximum admissible delay of service s
    for s in range(S):
        t_s.append(1)
    d_s_v = []  # maximum admissible number of divisions for service s VNF v
    for s in range(S):
        d_s_v.append([])
        for v in range(V):
            if g_s_v[s][v] == 0:
                d_s_v[s].append(1)
            else:
                d_s_v[s].append(4)
    l_v = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # value of l_v
    multiplier = 3  # regulate the amount of flows
    for m in range(M):
        alpha_m.append(450)
        beta_m.append(1)
        c_m.append(450)
    lambda_s_v = [[117.69, 117.69, 117.69, 11.77, 11.77, 117.69, 0, 0, 0, 0, 0, 0, 0, 0, 117.69, 117.69, 11.77],
                  [179.82, 179.82, 179.82, 17.98, 17.98, 179.82, 179.82, 17.98, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [50, 50, 50, 5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 0, 0, 0, 0],
                  [50, 50, 50, 5, 50, 0, 0, 0, 0, 0, 50, 5, 0, 0, 0, 0, 0],
                  [179.82, 179.82, 179.82, 17.98, 17.98, 0, 0, 0, 0, 0, 0, 0, 17.9, 179.82, 0, 0, 0]]  # arrival rate of service s VNF v
    for s in range(S):
        for v in range(V):
            lambda_s_v[s][v] = lambda_s_v[s][v] * multiplier
    # synthetic scenario
    # M = 10
    # S = 3
    # V = 3
    # g_s_v = [[1, 0, 1],
    #          [0, 1, 1],
    #          [1, 1, 0]]  # whether service s includes VNF v
    # t_s = []  # maximum admissible delay of service s
    # for s in range(S):
    #     t_s.append(1)
    # d_s_v = []  # maximum admissible number of divisions for service s VNF v
    # for s in range(S):
    #     d_s_v.append([])
    #     for v in range(V):
    #         if g_s_v[s][v] == 0:
    #             d_s_v[s].append(1)
    #         else:
    #             d_s_v[s].append(4)
    # l_v = [1, 1, 1]  # value of l_v
    # multiplier = 2.4
    # for m in range(M):
    #     alpha_m.append(450)
    #     beta_m.append(1)
    #     c_m.append(300)
    # lambda_s_v = [[117.69, 0, 117.69],
    #               [0, 50, 50],
    #               [179.82, 179.82, 0]]  # arrival rate of service s VNF v
    # for s in range(S):
    #     for v in range(V):
    #         lambda_s_v[s][v] = lambda_s_v[s][v] * multiplier

    print("multiplier=" + str(multiplier))
    t1 = time.time()
    res_1 = flow_partition(M, S, V, alpha_m, beta_m, c_m, g_s_v, lambda_s_v, t_s, d_s_v, l_v)
    t2 = time.time()
    print('results---flow partition--- =', res_1)
    print("time: " + str(t2 - t1))
