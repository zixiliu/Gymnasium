import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box


class FrankaXelaEnv(MujocoEnv, utils.EzPickle):
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
            low=-np.inf, high=np.inf, shape=(224,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self, "franka_xela.xml", 25, observation_space=observation_space, **kwargs
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
            geom_id = self.model.geom(geom_name).id
            self.geom_idx_tactile[geom_id] = geom_name
            self.list_of_geom_ids.append(geom_id)

        geom_name = "object"
        object_geom_id = self.model.geom(geom_name).id
        self.object_geom_id = object_geom_id

        ## add visualization for minkowski
        self.epsilon_force_geom_id = self.model.geom("epsilon_force").id
        self.epsilon_torque_geom_id = self.model.geom("epsilon_torque").id
        self.epsilon1_force_geom_id = self.model.geom("epsilon1_force").id
        self.epsilon1_torque_geom_id = self.model.geom("epsilon1_torque").id
        self.hull_force_geom_id = self.model.geom("hull_force").id
        self.hull_torque_geom_id = self.model.geom("hull_torque").id

        # print(f"list_of_geom_ids: {self.list_of_geom_ids}")

    def get_reward(self):
        d = self.data
        object_center_of_mass_world = d.geom_xpos[self.object_geom_id]
        z_object = object_center_of_mass_world[2]
        # print(f"z_object: {z_object}")
        # bottle_at_target_reward = 1.0 - abs(z_object - 1) if (z_object > 0.61) else 0
        bottle_at_target_reward = 1.0 - abs(z_object - 1)

        # Calculate contact reward
        _, _, _, contact_forces = self.get_contact_sensor_readings()
        indices_of_contact_sensors = np.where(np.linalg.norm(contact_forces, axis=1) > 0.2)[0]
        num_contacts = len(indices_of_contact_sensors)
        contact_reward = 1.0 if num_contacts >= 3 else 0.0

        reward = bottle_at_target_reward + contact_reward


        return reward, bottle_at_target_reward, contact_reward

    def check_if_terminated(self):

        """ Terminate if the bottle's COM z value is too low. """
        d = self.data
        object_center_of_mass_world = d.geom_xpos[self.object_geom_id]
        z_object = object_center_of_mass_world[2]
        bottole_down = z_object < 0.4
        terminated = bottole_down

        return False

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        # print(f"action: {a}")
        # import pdb; pdb.set_trace()

        reward, bottle_at_target_reward, contact_reward = self.get_reward()
        # print(f"reward: {reward}")
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
                bottle_at_target_reward=bottle_at_target_reward,
                contact_reward=contact_reward
            ),
        )

    def get_force_world(self):
        d = self.data
        contact_forces_site = d.sensordata.copy().reshape((-1, 3)) # n x 3 array for contact forces in the orders that sensors are defined, in the site frame
        contact_forces_world = np.zeros(contact_forces_site.shape)

        num_sensors = len(self.list_of_sensor_site_names)
        for i, site_name in enumerate(self.list_of_sensor_site_names):
            site_xmat = d.site(site_name).xmat.reshape((3,3))
            contact_forces_world[i,:] = np.matmul(site_xmat, contact_forces_site[i,:].reshape((3,1))).reshape((3,))

        return -1 * contact_forces_world

    def get_contact_sensor_readings(self):
        d = self.data

        ## let's get contact forces from these contact locations.
        contact_forces = self.get_force_world()

        contact_locations = d.contact.pos
        surface_normals = -1 * d.contact.frame[:, 0:3] #// normal is in [0-2], points from geom[0] to geom[1]
        contact_geoms = d.contact.geom
        contact_locations_sensor = np.zeros(contact_forces.shape)
        surface_normals_sensor = np.zeros(contact_forces.shape)
        # mu = d.contact.mu
        # contact_friction = d.contact.friction # n x 5 array for contact friction parameters

        for i, contact_geom_pair in enumerate(contact_geoms):
            id0 = contact_geom_pair[0]
            id1 = contact_geom_pair[1]

            if id0 in self.list_of_geom_ids:
                index_id0 = self.list_of_geom_ids.index(id0)
                tactile_id = id0
                contact_locations_sensor[index_id0, :] = contact_locations[i, :]
                surface_normals_sensor[index_id0, :] = surface_normals[i, :]

        normal_forces = contact_forces * surface_normals_sensor

        return contact_locations_sensor, surface_normals_sensor, normal_forces, contact_forces


    def _get_obs(self):
        contact_locations_sensor, surface_normals_sensor, normal_forces, contact_forces = self.get_contact_sensor_readings()
        object_center_of_mass_world = self.data.geom_xpos[self.object_geom_id]
        return np.concatenate(
            [
                self.data.qpos.flatten(),
                self.data.qvel.flatten(),
                contact_locations_sensor.flatten(),
                surface_normals_sensor.flatten(),
                # normal_forces.flatten(),
                contact_forces.flatten(),
                object_center_of_mass_world.flatten(),
                # self.sim.data.qpos.flat[2:],
                # self.sim.data.qvel.flat,
                # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
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

# from stable_baselines.common.env_checker import check_env
# env = FrankaXelaEnv()
# check_env(env, warn=True)
