from sys import argv

import stable_baselines3
import sb3_contrib
from stable_baselines3.common import vec_env

import supersuit as ss
import torch

import cv2
import numpy as np
from tqdm import tqdm

from examples.pettingzoo import utils
from meltingpot.python import substrate

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")


def main():
    recurrent = len(argv) == 3 and argv[2] == "recurrent"
    substrate_name = "commons_harvest__open"
    player_roles = substrate.get_config(substrate_name).default_player_roles[:4]
    env_config = {"substrate": substrate_name, "roles": player_roles}
    rollout_len = 1000
    num_envs = 1  # number of parallel multi-agent environments
    num_frames = 1 if recurrent else 50

    env = utils.parallel_env(
        max_cycles=rollout_len,
        env_config=env_config,
        render_mode="human"
    )
    env = ss.observation_lambda_v0(env, lambda x, _: x["RGB"],
                                        lambda s: s["RGB"])
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class="stable_baselines3"
    )
    env = vec_env.VecMonitor(env)
    env = vec_env.VecTransposeImage(env, True)
    env = vec_env.VecFrameStack(env, num_frames)
    # edit manually
    model_num = 1 if len(argv) == 1 else int(argv[1])
    logdir = f"./results/sb3/harvest_open_ppo_paramsharing/{'Recurrent'*recurrent}PPO_{model_num}"
    if recurrent:
        model = sb3_contrib.RecurrentPPO.load(logdir + "/model")  # noqa: F841
    else:
        model = stable_baselines3.PPO.load(logdir + "/model")  # noqa: F841
    obs = env.reset()
    out_shape = np.array((160, 160)) * 3
    out = cv2.VideoWriter(f"output{model_num}_{'recurrent'*recurrent}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 10, out_shape)
    frames = 1
    run_frames = 500
    for _ in tqdm(range(run_frames)):
        # print(f"Frame {frames}")
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if 'terminal_observation' in info[0].keys():
            break
        # save frame to video file
        frame = env.venv.venv.venv.venv.vec_envs[0].par_env.aec_env.env.env.env.render(mode='rgb_array')
        # swap r and b channels
        # input(frame.shape)
        frame = frame[:, :, [2, 1, 0]].repeat(3, axis=0).repeat(3, axis=1)
        frames += 1
        out.write(frame)
    out.release()


if __name__ == "__main__":
    main()
