from typing import Union

import attr

# NOTE maybe write my own create_environment_fns


# c = {"runner": {"base_dir": "pippo", "mimmo": {"cip": 2}}, "a": 2}


# class Runner(dict):
#     # base_dir: str

#     def __init__(self, **kwargs):
#         super().__init__(self, **kwargs)
#         # self.base_dir = self["runner", "base_dir"]

#     def __getitem__(self, keys: Union[tuple, str]):
#         print(f"start: {self}")
#         if isinstance(keys, str):
#             print("here")
#             try:
#                 # print("found")
#                 return super().__getitem__(keys)
#             except KeyError as k:
#                 raise AttributeError(k)
#         k, *rest = keys
#         print(k, rest)
#         inner = self[k]
#         print(type(inner))
#         return inner[tuple(rest)]
#         # return self[k][tuple(rest)]
#         # # inner = self[k]
#         # inner = self.__dict__[k]
#         # print(f"got inner: {inner}")
#         # res = inner.__getitem__(tuple(rest))
#         # # print("return")
#         # return res


@attr.s(auto_attribs=True)
class Runner:
    conf: dict
    base_dir: attr.ib(init=False)

    def __post_init__(self):
        self.base_dir = self.conf
        ...

    @property
    def hparams(self):
        ...

    # def __init__(
    #     self,
    #     base_dir,
    #     create_agent_fn,
    #     create_environment_fn=atari_lib.create_atari_environment,
    #     checkpoint_file_prefix="ckpt",
    #     logging_file_prefix="log",
    #     log_every_n=1,
    #     num_iterations=200,
    #     training_steps=250000,
    #     evaluation_steps=125000,
    #     max_steps_per_episode=27000,
    #     clip_rewards=True,
    # ):
    #     assert base_dir is not None
    #     tf.compat.v1.disable_v2_behavior()

    #     self._logging_file_prefix = logging_file_prefix
    #     self._log_every_n = log_every_n
    #     self._num_iterations = num_iterations
    #     self._training_steps = training_steps
    #     self._evaluation_steps = evaluation_steps
    #     self._max_steps_per_episode = max_steps_per_episode
    #     self._base_dir = base_dir
    #     self._clip_rewards = clip_rewards
    #     self._create_directories()
    #     self._summary_writer = tf.compat.v1.summary.FileWriter(self._base_dir)

    #     self._environment = create_environment_fn()
    #     config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    #     # Allocate only subset of the GPU memory as needed which allows for running
    #     # multiple agents/workers on the same GPU.
    #     config.gpu_options.allow_growth = True
    #     # Set up a session and initialize variables.
    #     self._sess = tf.compat.v1.Session("", config=config)
    #     self._agent = create_agent_fn(
    #         self._sess, self._environment, summary_writer=self._summary_writer
    #     )
    #     self._summary_writer.add_graph(graph=tf.compat.v1.get_default_graph())
    #     self._sess.run(tf.compat.v1.global_variables_initializer())

    #     self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

    # def _create_directories(self):
    #     self._checkpoint_dir = os.path.join(self._base_dir, "checkpoints")
    #     self._logger = logger.Logger(os.path.join(self._base_dir, "logs"))

    # def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    #     self._checkpointer = checkpointer.Checkpointer(
    #         self._checkpoint_dir, checkpoint_file_prefix
    #     )
    #     self._start_iteration = 0
    #     # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    #     # that we have finished iteration 0 (so we will start from iteration 1).
    #     latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
    #         self._checkpoint_dir
    #     )
    #     if latest_checkpoint_version >= 0:
    #         experiment_data = self._checkpointer.load_checkpoint(
    #             latest_checkpoint_version
    #         )
    #         if self._agent.unbundle(
    #             self._checkpoint_dir, latest_checkpoint_version, experiment_data
    #         ):
    #             if experiment_data is not None:
    #                 assert "logs" in experiment_data
    #                 assert "current_iteration" in experiment_data
    #                 self._logger.data = experiment_data["logs"]
    #                 self._start_iteration = experiment_data["current_iteration"] + 1
    #             logging.info(
    #                 "Reloaded checkpoint and will start from iteration %d",
    #                 self._start_iteration,
    #             )

    # def _initialize_episode(self):
    #     initial_observation = self._environment.reset()
    #     return self._agent.begin_episode(initial_observation)

    # def _run_one_step(self, action):
    #     observation, reward, is_terminal, _ = self._environment.step(action)
    #     return observation, reward, is_terminal

    # def _end_episode(self, reward, terminal=True):
    #     if isinstance(self._agent, jax_dqn_agent.JaxDQNAgent):
    #         self._agent.end_episode(reward, terminal)
    #     else:
    #         # TODO(joshgreaves): Add terminal signal to TF dopamine agents
    #         self._agent.end_episode(reward)

    # def _run_one_episode(self):
    #     step_number = 0
    #     total_reward = 0.0

    #     action = self._initialize_episode()
    #     is_terminal = False

    #     # Keep interacting until we reach a terminal state.
    #     while True:
    #         observation, reward, is_terminal = self._run_one_step(action)

    #         total_reward += reward
    #         step_number += 1

    #         if self._clip_rewards:
    #             # Perform reward clipping.
    #             reward = np.clip(reward, -1, 1)

    #         if (
    #             self._environment.game_over
    #             or step_number == self._max_steps_per_episode
    #         ):
    #             # Stop the run loop once we reach the true end of episode.
    #             break
    #         elif is_terminal:
    #             # If we lose a life but the episode is not over, signal an artificial
    #             # end of episode to the agent.
    #             self._end_episode(reward, is_terminal)
    #             action = self._agent.begin_episode(observation)
    #         else:
    #             action = self._agent.step(reward, observation)

    #     self._end_episode(reward, is_terminal)

    #     return step_number, total_reward

    # def _run_one_phase(self, min_steps, statistics, run_mode_str):
    #     step_count = 0
    #     num_episodes = 0
    #     sum_returns = 0.0

    #     while step_count < min_steps:
    #         episode_length, episode_return = self._run_one_episode()
    #         statistics.append(
    #             {
    #                 "{}_episode_lengths".format(run_mode_str): episode_length,
    #                 "{}_episode_returns".format(run_mode_str): episode_return,
    #             }
    #         )
    #         step_count += episode_length
    #         sum_returns += episode_return
    #         num_episodes += 1
    #         # We use sys.stdout.write instead of logging so as to flush frequently
    #         # without generating a line break.
    #         sys.stdout.write(
    #             "Steps executed: {} ".format(step_count)
    #             + "Episode length: {} ".format(episode_length)
    #             + "Return: {}\r".format(episode_return)
    #         )
    #         sys.stdout.flush()
    #     return step_count, sum_returns, num_episodes

    # def _run_train_phase(self, statistics):
    #     # Perform the training phase, during which the agent learns.
    #     self._agent.eval_mode = False
    #     start_time = time.time()
    #     number_steps, sum_returns, num_episodes = self._run_one_phase(
    #         self._training_steps, statistics, "train"
    #     )
    #     average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    #     statistics.append({"train_average_return": average_return})
    #     time_delta = time.time() - start_time
    #     average_steps_per_second = number_steps / time_delta
    #     statistics.append({"train_average_steps_per_second": average_steps_per_second})
    #     logging.info(
    #         "Average undiscounted return per training episode: %.2f", average_return
    #     )
    #     logging.info(
    #         "Average training steps per second: %.2f", average_steps_per_second
    #     )
    #     return num_episodes, average_return, average_steps_per_second

    # def _run_eval_phase(self, statistics):
    #     # Perform the evaluation phase -- no learning.
    #     self._agent.eval_mode = True
    #     _, sum_returns, num_episodes = self._run_one_phase(
    #         self._evaluation_steps, statistics, "eval"
    #     )
    #     average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
    #     logging.info(
    #         "Average undiscounted return per evaluation episode: %.2f", average_return
    #     )
    #     statistics.append({"eval_average_return": average_return})
    #     return num_episodes, average_return

    # def _run_one_iteration(self, iteration):
    #     statistics = iteration_statistics.IterationStatistics()
    #     logging.info("Starting iteration %d", iteration)
    #     (
    #         num_episodes_train,
    #         average_reward_train,
    #         average_steps_per_second,
    #     ) = self._run_train_phase(statistics)
    #     num_episodes_eval, average_reward_eval = self._run_eval_phase(statistics)

    #     self._save_tensorboard_summaries(
    #         iteration,
    #         num_episodes_train,
    #         average_reward_train,
    #         num_episodes_eval,
    #         average_reward_eval,
    #         average_steps_per_second,
    #     )
    #     return statistics.data_lists

    # def _save_tensorboard_summaries(
    #     self,
    #     iteration,
    #     num_episodes_train,
    #     average_reward_train,
    #     num_episodes_eval,
    #     average_reward_eval,
    #     average_steps_per_second,
    # ):
    #     summary = tf.compat.v1.Summary(
    #         value=[
    #             tf.compat.v1.Summary.Value(
    #                 tag="Train/NumEpisodes", simple_value=num_episodes_train
    #             ),
    #             tf.compat.v1.Summary.Value(
    #                 tag="Train/AverageReturns", simple_value=average_reward_train
    #             ),
    #             tf.compat.v1.Summary.Value(
    #                 tag="Train/AverageStepsPerSecond",
    #                 simple_value=average_steps_per_second,
    #             ),
    #             tf.compat.v1.Summary.Value(
    #                 tag="Eval/NumEpisodes", simple_value=num_episodes_eval
    #             ),
    #             tf.compat.v1.Summary.Value(
    #                 tag="Eval/AverageReturns", simple_value=average_reward_eval
    #             ),
    #         ]
    #     )
    #     self._summary_writer.add_summary(summary, iteration)

    # def _log_experiment(self, iteration, statistics):
    #     self._logger["iteration_{:d}".format(iteration)] = statistics
    #     if iteration % self._log_every_n == 0:
    #         self._logger.log_to_file(self._logging_file_prefix, iteration)

    # def _checkpoint_experiment(self, iteration):
    #     experiment_data = self._agent.bundle_and_checkpoint(
    #         self._checkpoint_dir, iteration
    #     )
    #     if experiment_data:
    #         experiment_data["current_iteration"] = iteration
    #         experiment_data["logs"] = self._logger.data
    #         self._checkpointer.save_checkpoint(iteration, experiment_data)

    # def run_experiment(self):
    #     logging.info("Beginning training...")
    #     if self._num_iterations <= self._start_iteration:
    #         logging.warning(
    #             "num_iterations (%d) < start_iteration(%d)",
    #             self._num_iterations,
    #             self._start_iteration,
    #         )
    #         return

    #     for iteration in range(self._start_iteration, self._num_iterations):
    #         statistics = self._run_one_iteration(iteration)
    #         self._log_experiment(iteration, statistics)
    #         self._checkpoint_experiment(iteration)
    #     self._summary_writer.flush()
