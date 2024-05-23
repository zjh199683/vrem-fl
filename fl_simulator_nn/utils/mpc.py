import numpy as np
from scipy import sparse
import casadi as cs
import do_mpc


def build_mat(Q, P, W, N):
    """Build the cal matrices for MPC"""
    P_cal = None if P is None else build_a_cal(sparse.csc_matrix(P), N)
    Q_cal = None if Q is None else build_b_cal(sparse.csc_matrix(Q), P_cal)
    W_cal = None if W is None else build_b_cal(sparse.csc_matrix(W), P_cal)
    return Q_cal, P_cal, W_cal


def build_a_cal(A, N):
    """Build state evolution matrix"""
    return sparse.vstack([A ** k for k in range(1, N + 1)])


def build_b_cal(B, A_cal):
    """Build control and noise evolution matrix"""
    N = int(A_cal.shape[0] / A_cal.shape[1])
    M = A_cal.shape[1]
    A_cal = sparse.lil_matrix(A_cal)
    A_trans = sparse.hstack([sparse.vstack([sparse.vstack([sparse.eye(M) for _ in range(k + 1)], format='lil')]
                                           + [A_cal[k * M: - M]]) for k in range(N)], format='csc')
    B_trans = sparse.hstack([sparse.vstack([np.zeros((k * M, M))] + [B for _ in range(k, N)]) for k in range(N)],
                            format='csc')
    B_cal = A_trans.multiply(B_trans)
    return B_cal


def build_battery_model(n_clients, b_max, cpu_coeff, tx_energy):
    """ Builds mpc model according to do-mpc docs. """
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    # Battery states
    b_lvl = model.set_variable(var_type='_x', var_name='b_lvl', shape=(n_clients, 1))
    # Control variable (# of local steps)
    local_steps = model.set_variable(var_type='_u', var_name='local_steps', shape=(n_clients, 1))
    # energy release
    # e_r = model.set_variable(var_type='_u', var_name='energy_release', shape=(n_clients, 1))
    # Harvested energy
    e_h = model.set_variable(var_type='_tvp', var_name='e_h', shape=(n_clients, 1))
    # State equations
    b_lvl_next = b_lvl - (tx_energy + cpu_coeff * local_steps) + e_h # - e_r
    b_lvl_next = cs.if_else(b_lvl_next >= b_max, b_max, b_lvl_next)
    model.set_rhs('b_lvl', b_lvl_next)
    # Setup model:
    model.setup()

    return model, b_lvl_next


def build_mpc(model, n_clients, weights, window_size, constants, harvested_energy, b_lvl_next, bmax):
    """ Builds the mpc controller. """
    mpc = do_mpc.controller.MPC(model)
    # Set parameters
    setup_mpc = {
        'n_horizon': window_size,
        't_step': 1,
        'state_discretization': 'discrete'
    }
    mpc.set_param(**setup_mpc)
    # Suppressing the output. To be removed for debugging
    ipopt_params = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
    mpc.set_param(nlpsol_opts=ipopt_params)

    # Configure objective function:
    c1 = 2 * cs.sqrt(constants[0] * constants[1]) * cs.sum1(weights * cs.power(model.u['local_steps'] + 1e-3, -1 / 2))
    c2 = constants[2] * cs.sum1(weights * model.u['local_steps'])
    c3 = constants[3] * cs.mmax(cs.power(model.u['local_steps'], 2) - model.u['local_steps'])
    cost = 1 * (c1 + c2 + c3)# + 1e-2 * cs.sum1(model.u['energy_release'])
    s_cost = cs.sum1(model.x['b_lvl']) * 0
    mpc.set_objective(mterm=s_cost, lterm=cost)

    # State and input bounds:
    mpc.bounds['lower', '_u', 'local_steps'] = np.zeros((n_clients,))
    mpc.bounds['lower', '_x', 'b_lvl'] = np.zeros((n_clients,))
    # mpc.bounds['upper', '_x', 'b_lvl'] = np.array(bmax)
    # mpc.bounds['lower', '_u', 'energy_release'] = np.zeros((n_clients,))
    # TODO: the following constraint is wrong?
    # er_bound = cs.if_else(model.x['b_lvl'] >= bmax, model.x['b_lvl'] - bmax, 0)
    # mpc.set_nl_cons('er_bound_1', model.u['energy_release'] - er_bound, 0)
    # mpc.set_nl_cons('er_bound_2', - model.u['energy_release'] + er_bound, 0)
    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        for n in range(window_size):
            tvp_template['_tvp', n, 'e_h'] = harvested_energy[:, n]
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    return mpc
