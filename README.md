# Measure-Valued Derivatives in Reinforcement Learning

Accompanying code for the paper [An Empirical Analysis of Measure-Valued Derivatives for Policy Gradients](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/JoaoCarvalho/2021_ijcnn-mvd_rl.pdf), submitted to IJCNN 2021. 

---

### Installation

Install MuJoCo as in 
[https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco](https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco)

Add to .bashrc
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
```

Install everything with
```bash
bash setup.sh
```

---

### Run the experiments

#### Test functions
```bash
python scripts/episodic/launch_episodic_test_functions.py 
```

#### LQR
```bash
python scripts/lqr_pg/launch_exp_oracle_pg_lqr.py
python scripts/lqr_pg/launch_exp_oracle_pg_lqr_error.py 
python scripts/lqr_pg/launch_exp_oracle_pg_lqr_error_training.py  
```

#### Off-policy
```bash
python scripts/off_policy/launch_exp_ddpg.py
python scripts/off_policy/launch_exp_sac.py
python scripts/off_policy/launch_exp_sac_extra_samples.py
python scripts/off_policy/launch_exp_sac_mvd.py
python scripts/off_policy/launch_exp_sac_sf.py
python scripts/off_policy/launch_exp_sac_sf_extra_samples.py
python scripts/off_policy/launch_exp_td3.py
```

#### On-policy
```bash
python scripts/on_policy/launch_exp_tree_mvd_lunarlander.py
python scripts/on_policy/launch_exp_tree_mvd_pendulum.py
python scripts/on_policy/launch_exp_tree_mvd_room.py
python scripts/on_policy/launch_exp_trustregion_lunarlander.py
python scripts/on_policy/launch_exp_trustregion_pendulum.py
python scripts/on_policy/launch_exp_trustregion_room.py
```


---

### Plot the results


```bash
mkdir out
python plots/plot_test_functions.py
python plots/plot_lqr.py
python plots/plot_lqr_error.py
python plots/plot_lqr_error_training.py
python plots/plot_off_policy.py
python plots/plot_on_policy.py
```

Check the plots in the `out` directory.
