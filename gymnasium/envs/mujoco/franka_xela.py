import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_py_env import MuJocoPyEnv
from gymnasium.spaces import Box


class FrankaXelaEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, **kwargs):
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(111,), dtype=np.float64
        )
        MuJocoPyEnv.__init__(
            self, "franka_xela.xml", 5, observation_space=observation_space, **kwargs
        ) # frame_skipL: 5
        utils.EzPickle.__init__(self, **kwargs)

        self.get_geom_idx_tactile()

    def get_geom_idx_tactile(self):
        ## list_of_geom_names is in the same order as the sensor is defined in the xml
        list_of_geom_names = ['hand_palm_top_left', 'hand_palm_top_right', 'hand_palm_bottom_left', \
            'hand_index_prox_dis', 'hand_index_prox_prox', 'hand_index_dist', 'hand_index_fingertip', \
            'hand_middle_prox_dis', 'hand_middle_prox_prox', 'hand_middle_dist', 'hand_middle_fingertip', \
            'hand_ring_prox_dis', 'hand_ring_prox_prox', 'hand_ring_dist', 'hand_ring_fingertip', \
            'hand_thumb_prox', 'hand_thumb_dist', 'hand_thumb_fingertip']

        self.list_of_sensor_site_names = ['hand_sensor'+s[4::] for s in list_of_geom_names]


        self.list_of_geom_names = list_of_geom_names
        self.geom_idx_tactile = {}
        self.list_of_geom_ids = []
        for geom_name in list_of_geom_names:
            geom_id = self.robot_model.geom(geom_name).id
            self.geom_idx_tactile[geom_id] = geom_name
            self.list_of_geom_ids.append(geom_id)

        geom_name = "object"
        object_geom_id = self.robot_model.geom(geom_name).id
        self.object_geom_id = object_geom_id

        ## add visualization for minkowski
        self.epsilon_force_geom_id = self.robot_model.geom("epsilon_force").id
        self.epsilon_torque_geom_id = self.robot_model.geom("epsilon_torque").id
        self.epsilon1_force_geom_id = self.robot_model.geom("epsilon1_force").id
        self.epsilon1_torque_geom_id = self.robot_model.geom("epsilon1_torque").id
        self.hull_force_geom_id = self.robot_model.geom("hull_force").id
        self.hull_torque_geom_id = self.robot_model.geom("hull_torque").id

        # print(f"list_of_geom_ids: {self.list_of_geom_ids}")

    def reward(self, xposbefore, xposafter, a):
        # forward_reward = (xposafter - xposbefore) / self.dt
        # ctrl_cost = 0.5 * np.square(a).sum()
        # contact_cost = (
        #     0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # )
        # survive_reward = 1.0
        # reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        ## TODO
        reward = 0.0
        return reward

    def check_if_terminated(self):
        #TODO update not_terminated for xela termination condition (cup in hand)
        not_terminated = (
            np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        )
        terminated = not not_terminated
        return terminated

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        reward = self.reward()
        state = self.state_vector()
        terminated = self.check_if_terminated()
        ob = self._get_obs()

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return (
            ob,
            reward,
            terminated,
            False,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        #TODO: update observation space for xela
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 0.5

from stable_baselines.common.env_checker import check_env
env = FrankaXelaEnv()
check_env(env, warn=True)
