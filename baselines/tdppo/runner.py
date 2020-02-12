import numpy as np
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, update_type, alpha, weight):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.update_type = update_type

    def run(self, ent_now):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_entropy = [],[],[],[],[],[],[]
        mb_svalues, mb_sentropy = [],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs, entropy = self.model.step(self.obs, S=self.states, M=self.dones)
            _, svalues, _, _, sentropy = self.model.sstep(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_entropy.append(entropy)
            mb_dones.append(self.dones)
            mb_svalues.append(svalues)
            mb_sentropy.append(sentropy)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_entropy = np.asarray(mb_entropy, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_svalues = np.asarray(mb_svalues, dtype=np.float32)
        mb_sentropy = np.asarray(mb_sentropy, dtype=np.float32)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)
        last_svalues = self.model.svalue(self.obs, S=self.states, M=self.dones)

        # add the reward and the entropy as the new reward
        mb_entropy = np.concatenate([mb_entropy[1:], mb_entropy[-1:]], axis=0)
        mb_sentropy = np.concatenate([mb_sentropy[1:], mb_sentropy[-1:]], axis=0)
        mb_rewards = mb_rewards + ent_now * mb_entropy
        mb_srewards = mb_rewards + ent_now * mb_sentropy

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        mb_tdadvs = np.zeros_like(mb_srewards)
        lastgaelam = 0
        td_lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
                nextsvalues = last_svalues
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
                nextsvalues = mb_svalues[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            td_delta = mb_srewards[t] + self.gamma * nextsvalues * nextnonterminal - mb_svalues[t]
            mb_tdadvs[t] = (1 - self.alpha) * td_delta + self.alpha * (1./self.lam - 1) * \
                self.gamma * self.lam * nextnonterminal * td_lastgaelam
            td_lastgaelam = td_delta + self.gamma * self.lam * nextnonterminal * td_lastgaelam

        mb_returns = mb_advs + mb_values

        if self.update_type == 'min':
            # the minimal value of mb_advs and mb_tdadvs
            mb_fadvs = np.amin([mb_advs, mb_tdadvs], 0)
        else:
            if self.update_type == 'max':
                # the maximan value of mb_advs and mb_tdadvs
                mb_fadvs = np.amax([mb_advs, mb_tdadvs], 0)
            else:
                # the different weights of mb_advs and mb_tdadvs
                mb_fadvs = self.weight * mb_advs + (1 - self.weight) * mb_tdadvs
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_fadvs, mb_neglogpacs, mb_entropy)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


