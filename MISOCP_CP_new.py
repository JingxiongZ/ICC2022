import cplex
# import gurobipy as gp

def setup_MISOCP(Q, M, V, S, beta, alpha, c_m, d_s_v, K_s_v_i_j, g_s_v, lambda_s_v, l_v, t_s):
    Q.objective.set_sense(Q.objective.sense.minimize)
    # objective
    # define w_m_v
    w_m_v = []
    for m in range(M):
        var1 = "w_m_" + str(m)
        var_w = []
        w_m_v.append([])
        types_w = []
        my_obj_w = []
        for v in range(V):
            var2 = var1 + "_v_" + str(v)
            var_w.append(var2)
            types_w.append("B")
            my_obj_w.append(alpha)
            w_m_v[m].append(var2)
        Q.variables.add(names=var_w, types=types_w, obj=my_obj_w)
    # define a_m
    a_m = []
    var_a = []
    lb_a = []
    ub_a = []
    types_a = []
    my_obj_a = []
    for m in range(M):
        var = "a_m_" + str(m)
        var_a.append(var)
        lb_a.append(0)
        ub_a.append(c_m)
        types_a.append("C")
        my_obj_a.append(beta)
        a_m.append(var)
    Q.variables.add(names=var_a, lb=lb_a, ub=ub_a, types=types_a, obj=my_obj_a)

    # define b_s_v_j
    b_s_v_j = []
    for s in range(S):
        var1 = "b_s_" + str(s)
        b_s_v_j.append([])
        for v in range(V):
            var2 = var1 + "_v_" + str(v)
            b_s_v_j[s].append([])
            var_b = []
            types_b = []
            for j in range(d_s_v[s][v]):
                var3 = var2 + "_j_" + str(j)
                b_s_v_j[s][v].append(var3)
                var_b.append(var3)
                types_b.append("B")
            Q.variables.add(names=var_b, types=types_b)
    # define y_s_v_i
    y_s_v_i = []
    for s in range(S):
        var1 = "y_s_" + str(s)
        y_s_v_i.append([])
        for v in range(V):
            var2 = var1 + "_v_" + str(v)
            y_s_v_i[s].append([])
            var_y = []
            types_y = []
            for i in range(d_s_v[s][v]):
                var3 = var2 + "_i_" + str(i)
                y_s_v_i[s][v].append(var3)
                var_y.append(var3)
                types_y.append("B")
            Q.variables.add(names=var_y, types=types_y)
    # define alpha_s_v_i_j
    alpha_s_v_i_j = []
    for s in range(S):
        var1 = "alpha_s_" + str(s)
        alpha_s_v_i_j.append([])
        for v in range(V):
            var2 = var1 + "_v_" + str(v)
            alpha_s_v_i_j[s].append([])
            for j in range(d_s_v[s][v]):
                var3 = var2 + "_j_" + str(j)
                alpha_s_v_i_j[s][v].append([])
                var_alpha = []
                types_alpha = []
                for i in range(d_s_v[s][v]):
                    var4 = var3 + "_i_" + str(i)
                    alpha_s_v_i_j[s][v][j].append(var4)
                    var_alpha.append(var4)
                    types_alpha.append("B")
                Q.variables.add(names=var_alpha, types=types_alpha)
    # define beta_s_v_m_i_j
    beta_s_v_m_i_j = []
    for s in range(S):
        var1 = "beta_s_" + str(s)
        beta_s_v_m_i_j.append([])
        for v in range(V):
            var2 = var1 + "_v_" + str(v)
            beta_s_v_m_i_j[s].append([])
            for m in range(M):
                var3 = var2 + "_m_" + str(m)
                beta_s_v_m_i_j[s][v].append([])
                for j in range(d_s_v[s][v]):
                    var4 = var3 + "_j_" + str(j)
                    beta_s_v_m_i_j[s][v][m].append([])
                    var_beta = []
                    types_beta = []
                    for i in range(d_s_v[s][v]):
                        var5 = var4 + "_i_" + str(i)
                        beta_s_v_m_i_j[s][v][m][j].append(var5)
                        var_beta.append(var5)
                        types_beta.append("B")
                    Q.variables.add(names=var_beta, types=types_beta)
    # define z_s_v_m_i
    z_s_v_m_i = []
    for s in range(S):
        var1 = "z_s_" + str(s)
        z_s_v_m_i.append([])
        for v in range(V):
            var2 = var1 + "_v_" + str(v)
            z_s_v_m_i[s].append([])
            for m in range(M):
                var3 = var2 + "_m_" + str(m)
                z_s_v_m_i[s][v].append([])
                var_z = []
                types_z = []
                for i in range(d_s_v[s][v]):
                    var4 = var3 + "_i_" + str(i)
                    z_s_v_m_i[s][v][m].append(var4)
                    var_z.append(var4)
                    types_z.append("B")
                Q.variables.add(names=var_z, types=types_z)
    # define delta_m
    delta_m = []
    var_delta = []
    types_delta = []
    lb_delta = []
    ub_delta = []
    for m in range(M):
        var = "delta_m_" + str(m)
        delta_m.append(var)
        var_delta.append(var)
        types_delta.append("C")
        lb_delta.append(0)
        ub_delta.append(c_m)
    Q.variables.add(names=var_delta, lb=lb_delta, ub=ub_delta, types=types_delta)
    # define e_s_v_m_i_j
    e_s_v_m_i_j = []
    for s in range(S):
        var1 = "e_s_" + str(s)
        e_s_v_m_i_j.append([])
        for v in range(V):
            var2 = var1 + "_v_" + str(v)
            e_s_v_m_i_j[s].append([])
            for m in range(M):
                var3 = var2 + "_m_" + str(m)
                e_s_v_m_i_j[s][v].append([])
                for j in range(d_s_v[s][v]):
                    var4 = var3 + "_j_" + str(j)
                    e_s_v_m_i_j[s][v][m].append([])
                    var_e = []
                    types_e = []
                    lb_e = []
                    for i in range(d_s_v[s][v]):
                        var5 = var4 + "_i_" + str(i)
                        e_s_v_m_i_j[s][v][m][j].append(var5)
                        var_e.append(var5)
                        types_e.append("C")
                        lb_e.append(0)
                    Q.variables.add(names=var_e, lb=lb_e, types=types_e)
    # define t_s_v
    t_s_v = []
    for s in range(S):
        var1 = "t_s_" + str(s)
        t_s_v.append([])
        var_t = []
        types_t = []
        lb_t = []
        for v in range(V):
            var2 = var1 + "_v_" + str(v)
            t_s_v[s].append(var2)
            var_t.append(var2)
            types_t.append("C")
            lb_t.append(0)
        Q.variables.add(names=var_t, lb=lb_t, types=types_t)
    # define nu_m
    nu_m = []
    var_nu = []
    types_nu = []
    lb_nu = []
    for m in range(M):
        var = "nu_m_" + str(m)
        nu_m.append(var)
        var_nu.append(var)
        types_nu.append("C")
        lb_nu.append(0)
    Q.variables.add(names=var_nu, lb=lb_nu, types=types_nu)
    # define p_s_v_m_i_j
    p_s_v_m_i_j = []
    for s in range(S):
        var1 = "p_s_" + str(s)
        p_s_v_m_i_j.append([])
        for v in range(V):
            var2 = var1 + "_v_" + str(v)
            p_s_v_m_i_j[s].append([])
            for m in range(M):
                var3 = var2 + "_m_" + str(m)
                p_s_v_m_i_j[s][v].append([])
                for j in range(d_s_v[s][v]):
                    var4 = var3 + "_j_" + str(j)
                    p_s_v_m_i_j[s][v][m].append([])
                    var_p = []
                    types_p = []
                    lb_p = []
                    for i in range(d_s_v[s][v]):
                        var5 = var4 + "_i_" + str(i)
                        p_s_v_m_i_j[s][v][m][j].append(var5)
                        var_p.append(var5)
                        types_p.append("C")
                        lb_p.append(0)
                    Q.variables.add(names=var_p, lb=lb_p, types=types_p)

    # constraints
    # (11a)
    for s in range(S):
        for v in range(V):
            ind_11a = []
            val_11a = []
            for j in range(d_s_v[s][v]):
                ind_11a.append(b_s_v_j[s][v][j])
                val_11a.append(1)
            expr = cplex.SparsePair(ind=ind_11a, val=val_11a)
            Q.linear_constraints.add(names=['11a'], lin_expr=[expr], senses=["E"], rhs=[g_s_v[s][v]])
    # (11b)
    for s in range(S):
        for v in range(V):
            ind_11b = []
            val_11b = []
            for i in range(d_s_v[s][v]):
                ind_11b.append(y_s_v_i[s][v][i])
                val_11b.append(1)
            for j in range(d_s_v[s][v]):
                ind_11b.append(b_s_v_j[s][v][j])
                val_11b.append(-(j + 1))
            expr = cplex.SparsePair(ind=ind_11b, val=val_11b)
            Q.linear_constraints.add(names=['11b'], lin_expr=[expr], senses=["E"], rhs=[0])
    # (13b)
    for s in range(S):
        for v in range(V):
            ind_13b = []
            val_13b = []
            for j in range(d_s_v[s][v]):
                for i in range(d_s_v[s][v]):
                    ind_13b.append(alpha_s_v_i_j[s][v][j][i])
                    val_13b.append(K_s_v_i_j[s][v][j][i])
            expr = cplex.SparsePair(ind=ind_13b, val=val_13b)
            Q.linear_constraints.add(names=['13b'], lin_expr=[expr], senses=["E"], rhs=[g_s_v[s][v]])
    # (14a)
    for s in range(S):
        for v in range(V):
            for j in range(d_s_v[s][v]):
                for i in range(d_s_v[s][v]):
                    ind_14a = [alpha_s_v_i_j[s][v][j][i], b_s_v_j[s][v][j]]
                    val_14a = [1, -1]
                    expr = cplex.SparsePair(ind=ind_14a, val=val_14a)
                    Q.linear_constraints.add(names=['14a'], lin_expr=[expr], senses=["L"], rhs=[0])
    # (14b)
    for s in range(S):
        for v in range(V):
            for j in range(d_s_v[s][v]):
                for i in range(d_s_v[s][v]):
                    ind_14b = [alpha_s_v_i_j[s][v][j][i], y_s_v_i[s][v][i]]
                    val_14b = [1, -1]
                    expr = cplex.SparsePair(ind=ind_14b, val=val_14b)
                    Q.linear_constraints.add(names=['14b'], lin_expr=[expr], senses=["L"], rhs=[0])
    # (14c)
    for s in range(S):
        for v in range(V):
            for j in range(d_s_v[s][v]):
                for i in range(d_s_v[s][v]):
                    ind_14c = [alpha_s_v_i_j[s][v][j][i], b_s_v_j[s][v][j], y_s_v_i[s][v][i]]
                    val_14c = [-1, 1, 1]
                    expr = cplex.SparsePair(ind=ind_14c, val=val_14c)
                    Q.linear_constraints.add(names=['14c'], lin_expr=[expr], senses=["L"], rhs=[1])
    # (15b)
    # for v in range(V):
    #     for m in range(M):
    #         ind_15b = []
    #         val_15b = []
    #         for s in range(S):
    #             for i in range(d_s_v[s][v]):
    #                 for j in range(d_s_v[s][v]):
    #                     ind_15b.append(beta_s_v_m_i_j[s][v][m][i][j])
    #                     val_15b.append(K_s_v_i_j[s][v][j][i] * lambda_s_v[s][v])
    #         ind_15b.append(a_m[m])
    #         val_15b.append(-1 / l_v[v])
    #         expr = cplex.SparsePair(ind=ind_15b, val=val_15b)
    #         Q.linear_constraints.add(names=['15b'], lin_expr=[expr], senses=["L"], rhs=[0])
    # (15c)
    for s in range(S):
        for v in range(V):
            for m in range(M):
                for j in range(d_s_v[s][v]):
                    for i in range(d_s_v[s][v]):
                        ind_15c = [beta_s_v_m_i_j[s][v][m][j][i], b_s_v_j[s][v][j]]
                        val_15c = [1, -1]
                        expr = cplex.SparsePair(ind=ind_15c, val=val_15c)
                        Q.linear_constraints.add(names=['15c'], lin_expr=[expr], senses=["L"], rhs=[0])
    # (15d)
    for s in range(S):
        for v in range(V):
            for m in range(M):
                for j in range(d_s_v[s][v]):
                    for i in range(d_s_v[s][v]):
                        ind_15d = [beta_s_v_m_i_j[s][v][m][j][i], z_s_v_m_i[s][v][m][i]]
                        val_15d = [1, -1]
                        expr = cplex.SparsePair(ind=ind_15d, val=val_15d)
                        Q.linear_constraints.add(names=['15d'], lin_expr=[expr], senses=["L"], rhs=[0])
    # (15e)
    for s in range(S):
        for v in range(V):
            for m in range(M):
                for j in range(d_s_v[s][v]):
                    for i in range(d_s_v[s][v]):
                        ind_15e = [beta_s_v_m_i_j[s][v][m][j][i], z_s_v_m_i[s][v][m][i], b_s_v_j[s][v][j]]
                        val_15e = [-1, 1, 1]
                        expr = cplex.SparsePair(ind=ind_15e, val=val_15e)
                        Q.linear_constraints.add(names=['15e'], lin_expr=[expr], senses=["L"], rhs=[1])
    # (16a)
    for m in range(M):
        ind_16a = []
        val_16a = []
        for s in range(S):
            for v in range(V):
                for j in range(d_s_v[s][v]):
                    for i in range(d_s_v[s][v]):
                        ind_16a.append(beta_s_v_m_i_j[s][v][m][j][i])
                        val_16a.append(K_s_v_i_j[s][v][j][i] * lambda_s_v[s][v] * l_v[v])
        ind_16a.append(delta_m[m])
        val_16a.append(-1)
        expr = cplex.SparsePair(ind=ind_16a, val=val_16a)
        Q.linear_constraints.add(names=['16a'], lin_expr=[expr], senses=["E"], rhs=[0])
    # (16b)
    for m in range(M):
        ind_16b = [a_m[m], delta_m[m]]
        val_16b = [1, -1]
        expr = cplex.SparsePair(ind=ind_16b, val=val_16b)
        Q.linear_constraints.add(names=['16b'], lin_expr=[expr], senses=["G"], rhs=[0])
    # (17b)
    for s in range(S):
        for v in range(V):
            ind_17b = []
            val_17b = []
            for m in range(M):
                for j in range(d_s_v[s][v]):
                    for i in range(d_s_v[s][v]):
                        ind_17b.append(e_s_v_m_i_j[s][v][m][j][i])
                        val_17b.append(1)
            ind_17b.append(t_s_v[s][v])
            val_17b.append(-1)
            expr = cplex.SparsePair(ind=ind_17b, val=val_17b)
            Q.linear_constraints.add(names=['17b'], lin_expr=[expr], senses=["L"], rhs=[0])
    # (17c)
    for m in range(M):
        ind_17c = [nu_m[m], a_m[m], delta_m[m]]
        val_17c = [-1, 1, -1]
        expr = cplex.SparsePair(ind=ind_17c, val=val_17c)
        Q.linear_constraints.add(names=['17c'], lin_expr=[expr], senses=["E"], rhs=[0])
    # (18a)
    for s in range(S):
        for v in range(V):
            for m in range(M):
                for j in range(d_s_v[s][v]):
                    for i in range(d_s_v[s][v]):
                        ind_18a = [p_s_v_m_i_j[s][v][m][j][i], e_s_v_m_i_j[s][v][m][j][i], nu_m[m]]
                        val_18a = [-1, 1, 1]
                        expr = cplex.SparsePair(ind=ind_18a, val=val_18a)
                        Q.linear_constraints.add(names=['18a'], lin_expr=[expr], senses=["E"], rhs=[0])
    # (18b)
    for s in range(S):
        for v in range(V):
            for m in range(M):
                for j in range(d_s_v[s][v]):
                    for i in range(d_s_v[s][v]):
                        q = cplex.SparseTriple(
                            ind1=[e_s_v_m_i_j[s][v][m][j][i], nu_m[m], p_s_v_m_i_j[s][v][m][j][i], beta_s_v_m_i_j[s][v][m][j][i]],
                            ind2=[e_s_v_m_i_j[s][v][m][j][i], nu_m[m], p_s_v_m_i_j[s][v][m][j][i], beta_s_v_m_i_j[s][v][m][j][i]],
                            val=[1, 1, -1, 2 * K_s_v_i_j[s][v][j][i] * l_v[v]])
                        Q.quadratic_constraints.add(quad_expr=q, sense="L", rhs=0)
    # (3)
    for s in range(S):
        for v in range(V):
            ind_3 = [y_s_v_i[s][v][0]]
            val_3 = [1]
            expr = cplex.SparsePair(ind=ind_3, val=val_3)
            Q.linear_constraints.add(names=['3'], lin_expr=[expr], senses=["E"], rhs=[g_s_v[s][v]])
    # (4)
    for s in range(S):
        for v in range(V):
            for i in range(1, d_s_v[s][v]):
                ind_4 = [y_s_v_i[s][v][i], y_s_v_i[s][v][i - 1]]
                val_4 = [1, -1]
                expr = cplex.SparsePair(ind=ind_4, val=val_4)
                Q.linear_constraints.add(names=['4'], lin_expr=[expr], senses=["L"], rhs=[0])
    # (5)
    for s in range(S):
        for v in range(V):
            for i in range(d_s_v[s][v]):
                ind_5 = []
                val_5 = []
                for m in range(M):
                    ind_5.append(z_s_v_m_i[s][v][m][i])
                    val_5.append(1)
                ind_5.append(y_s_v_i[s][v][i])
                val_5.append(-1)
                expr = cplex.SparsePair(ind=ind_5, val=val_5)
                Q.linear_constraints.add(names=['5'], lin_expr=[expr], senses=["E"], rhs=[0])
    # (6)
    for m in range(M):
        ind_6 = []
        val_6 = []
        for v in range(V):
            ind_6.append(w_m_v[m][v])
            val_6.append(1)
        expr = cplex.SparsePair(ind=ind_6, val=val_6)
        Q.linear_constraints.add(names=['6'], lin_expr=[expr], senses=["L"], rhs=[1])
    # (7)
    for s in range(S):
        for v in range(V):
            for m in range(M):
                ind_7 = []
                val_7 = []
                for i in range(d_s_v[s][v]):
                    ind_7.append(z_s_v_m_i[s][v][m][i])
                    val_7.append(1)
                ind_7.append(w_m_v[m][v])
                val_7.append(-1)
                expr = cplex.SparsePair(ind=ind_7, val=val_7)
                Q.linear_constraints.add(names=['7'], lin_expr=[expr], senses=["L"], rhs=[0])
    # (10c)
    for m in range(M):
        ind_10c = []
        val_10c = []
        for v in range(V):
            ind_10c.append(w_m_v[m][v])
            val_10c.append(c_m)
        ind_10c.append(a_m[m])
        val_10c.append(-1)
        expr = cplex.SparsePair(ind=ind_10c, val=val_10c)
        Q.linear_constraints.add(names=['10c'], lin_expr=[expr], senses=["G"], rhs=[0])
    # (10d)
    for s in range(S):
        ind_10d = []
        val_10d = []
        for v in range(V):
            ind_10d.append(t_s_v[s][v])
            val_10d.append(1)
        expr = cplex.SparsePair(ind=ind_10d, val=val_10d)
        Q.linear_constraints.add(names=['10d'], lin_expr=[expr], senses=["L"], rhs=[t_s[s]])


def MISOCP(M, V, S, beta, alpha, c_m, d_s_v, K_s_v_i_j, g_s_v, lambda_s_v, l_v, t_s):
    Q = cplex.Cplex()
    setup_MISOCP(Q, M, V, S, beta, alpha, c_m, d_s_v, K_s_v_i_j, g_s_v, lambda_s_v, l_v, t_s)
    print("Setting up for the MISOCP problem...")
    # P.set_results_stream(None)
    # P.set_warning_stream(None)
    # P.parameters.timelimit.set(5)
    # Q.parameters.timelimit.set(10000)
    # Q.parameters.mip.strategy.probe.set(3)
    # Q.parameters.mip.strategy.presolvenode.set(3)
    # Q.parameters.emphasis.mip.set(3)
    # Q.parameters.mip.tolerances.absmipgap = 0.01
    # Q.parameters.mip.tolerances.mipgap = 0.01
    # Q.parameters.mip.cuts.mircut.set(2)
    # Q.parameters.mip.cuts.covers.set(3)
    # Q.parameters.mip.cuts.cliques.set(3)
    # Q.parameters.mip.cuts.disjunctive.set(3)
    # Q.parameters.mip.cuts.liftproj.set(3)
    # Q.parameters.mip.cuts.localimplied.set(3)
    # Q.parameters.mip.cuts.bqp.set(2)
    # Q.parameters.mip.cuts.flowcovers.set(2)
    # Q.parameters.mip.cuts.pathcut.set(2)
    # Q.parameters.mip.cuts.zerohalfcut.set(2)
    # Q.parameters.mip.cuts.gomory.set(2)
    # Q.parameters.mip.cuts.gubcovers.set(2)
    # Q.parameters.mip.cuts.mcfcut.set(2)
    # Q.parameters.mip.strategy.variableselect.set(3)
    # Q.parameters.emphasis.memory.set(1)
    # Q.parameters.mip.strategy.file.set(3)
    # Q.parameters.workmem.set()
    # Q.parameters.mip.strategy.nodeselect.set(2)
    Q.solve()
    Q.write("MISOCP_CP_new.lp")
    print(Q.solution.get_status())
    print("Minimum deployment cost: " + str(Q.solution.get_objective_value()))
    vm = Q.solution.get_values()
    number = 0
    for i in range(M * V):
        if vm[i] > 0.99:
            number += 1
        else:
            continue
    print("the number of  activated VMs: " + str(number))
    print(vm[0:M * V + M])


if __name__ == "__main__":
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
    # l_v = [1, 1, 1]  # value of l_v
    # alpha = 450
    # beta = 1
    # c_m = 300
    # lambda_s_v = [[117.69, 0, 117.69],
    #               [0, 50, 50],
    #               [179.82, 179.82, 0]]  # arrival rate of service s VNF v

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
    l_v = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # value of l_v
    alpha = 450
    beta = 1
    c_m = 450
    lambda_s_v = [[117.69, 117.69, 117.69, 11.77, 11.77, 117.69, 0, 0, 0, 0, 0, 0, 0, 0, 117.69, 117.69, 11.77],
                  [179.82, 179.82, 179.82, 17.98, 17.98, 179.82, 179.82, 17.98, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [50, 50, 50, 5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 0, 0, 0, 0],
                  [50, 50, 50, 5, 50, 0, 0, 0, 0, 0, 50, 5, 0, 0, 0, 0, 0],
                  [179.82, 179.82, 179.82, 17.98, 17.98, 0, 0, 0, 0, 0, 0, 0, 17.9, 179.82, 0, 0, 0]]  # arrival rate of service s VNF v

    d_s_v = []  # maximum number of divisions of service s VNF v
    for s in range(S):
        d_s_v.append([])
        for v in range(V):
            if lambda_s_v[s][v] < 1:
                d_s_v[s].append(1)
            else:
                d_s_v[s].append(4)
    multiplier = 3
    for s in range(S):
        for v in range(V):
            lambda_s_v[s][v] = lambda_s_v[s][v] * multiplier
    K_s_v_i_j = []
    for s in range(S):
        K_s_v_i_j.append([])
        for v in range(V):
            K_s_v_i_j[s].append([])
            for j in range(d_s_v[s][v]):
                K_s_v_i_j[s][v].append([])
                for i in range(d_s_v[s][v]):
                    K_s_v_i_j[s][v][j].append(0)
                for i in range(j + 1):
                    K_s_v_i_j[s][v][j][i] = 1 / (j + 1)

    MISOCP(M, V, S, beta, alpha, c_m, d_s_v, K_s_v_i_j, g_s_v, lambda_s_v, l_v, t_s)