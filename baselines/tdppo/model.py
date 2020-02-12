import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD
  
        # the train model
        with tf.variable_scope('tdppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)
        # the shadow model
        with tf.variable_scope('shadow_tdppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            sact_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                strain_model = policy(nbatch_train, nsteps, sess)
            else:
                strain_model = policy(microbatch_size, nsteps, sess)

        # deliver the parameters of the model to the parameters of shadow model shadow_tdppo2_model
        e_params = tf.global_variables('tdppo2_model')
        s_model_params = tf.global_variables('shadow_tdppo2_model')
        e_model_params = e_params[:len(s_model_params)] 
        # e_params = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='tdppo2_model')
        # s_params = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='shadow_tdppo2_model')

        self.replace_op = [tf.assign(s,e) for s, e in zip(s_model_params, e_model_params)] 
        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        self.ENTNOW = ENTNOW = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
        # Entropy
        self.OLDENTROPY = OLDENTROPY = tf.placeholder(tf.float32, [None])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        # entropy = tf.reduce_mean(train_model.pd.entropy())
        entropy = train_model.pd.entropy()
        e_adv = entropy - OLDENTROPY
        e_advs = tf.concat([e_adv[1:], e_adv[-1:]], axis=0)
        # ADVS = ADV +  ENTNOW * 1e-3 * e_advs
        ADVS = ADV +  ENTNOW * 1e-3 * e_advs
        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADVS * ratio

        # pg_losses2 = -ADVS * ratio
        pg_losses2 = -ADVS * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        # loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        loss = pg_loss + vf_loss * vf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('tdppo2_model')
        # 2. Build our trainer
        if comm is not None and comm.Get_size() > 1:
            self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        # the function of model which is used in runner
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        # the function of shadow model which is used in runner
        self.strain_model = strain_model
        self.sact_model = sact_model
        self.sstep = sact_model.step
        self.svalue = sact_model.value
        self.sinitial_state = sact_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

    def update_shadow_model(self):
        # update the shadow model by the current model before training
        self.sess.run(self.replace_op)


    def train(self, lr, ent_now, cliprange, obs, returns, masks, actions, values, advs, neglogpacs, entropy, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        # advs = returns - values
        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.ENTNOW: ent_now,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values,
            self.OLDENTROPY: entropy
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks
            
        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]