import sys

sys.path.append(".")
import pytest
import numpy as np
from env.psp_state import PSPState
from problem.psp_description import PSPDescription
from models.agent_specification import AgentSpecification
from models.psp_agent import PSPAgent
from models.training_specification import TrainingSpecification

from env.psp_env_specification import PSPEnvSpecification
from utils.loaders import PSPLoader
from utils.psp_env_observation import PSPEnvObservation as EnvObservation
from utils.psp_agent_observation import PSPAgentObservation as AgentObservation


@pytest.fixture
def small_pb():
    loader = PSPLoader()
    return loader.load_single("instances/psp/small/small.sm")


@pytest.fixture
def large_pb():
    loader = PSPLoader()
    return loader.load_single("instances/psp/sm/j3010_1.sm")


@pytest.fixture
def problem_description_small(small_pb):
    return PSPDescription(
        transition_model_config="simple",
        reward_model_config="Sparse",
        deterministic=True,
        train_psps=[small_pb],
        test_psps=[small_pb],
    )


@pytest.fixture
def problem_description_large(large_pb):
    return PSPDescription(
        transition_model_config="simple",
        reward_model_config="Sparse",
        deterministic=True,
        train_psps=[large_pb],
        test_psps=[large_pb],
    )


@pytest.fixture
def env_specification_small(problem_description_small):
    return PSPEnvSpecification(
        problems=problem_description_small,
        normalize_input=True,
        input_list=["duration"],
        max_edges_factor=2,
        sample_n_jobs=-1,
        chunk_n_jobs=-1,
        observe_conflicts_as_cliques=True,
        observe_real_duration_when_affect=False,
        do_not_observe_updated_bounds=False,
    )


@pytest.fixture
def env_specification_large(problem_description_large):
    return PSPEnvSpecification(
        problems=problem_description_large,
        normalize_input=True,
        input_list=["duration"],
        max_edges_factor=2,
        sample_n_jobs=-1,
        chunk_n_jobs=-1,
        observe_conflicts_as_cliques=True,
        observe_real_duration_when_affect=False,
        do_not_observe_updated_bounds=False,
    )


@pytest.fixture
def agent_specification(env_specification_small):
    return AgentSpecification(
        lr=2e-4,
        fe_lr=None,
        n_steps_episode=1024,
        batch_size=128,
        n_epochs=10,
        gamma=1,
        clip_range=0.25,
        target_kl=0.04,
        ent_coef=0.005,
        vf_coef=0.5,
        normalize_advantage=True,
        optimizer="adam",
        freeze_graph=False,
        n_features=env_specification_small.n_features,
        gconv_type="gatv2",
        graph_has_relu=False,
        graph_pooling="learn",
        layer_pooling="all",
        mlp_act="tanh",
        mlp_act_graph="gelu",
        n_workers=10,
        device="cuda",
        n_mlp_layers_features_extractor=3,
        n_layers_features_extractor=6,
        hidden_dim_features_extractor=64,
        n_attention_heads=4,
        reverse_adj=False,
        residual_gnn=False,
        normalize_gnn=False,
        conflicts="clique",
        n_mlp_layers_actor=1,
        hidden_dim_actor=64,
        n_mlp_layers_critic=1,
        hidden_dim_critic=64,
        fe_type="dgl",
        transformer_flavor="linear",
        dropout=0.0,
        cache_lap_node_id=False,
        lap_node_id_k=10,
        rpo=False,
        rpo_smoothing_param=1.0,
    )


@pytest.fixture
def state_small(problem_description_small, env_specification_small):
    return PSPState(
        env_specification_small,
        problem_description_small,
        problem_description_small.train_psps[0],
        deterministic=True,
        observe_conflicts_as_cliques=False,
    )


@pytest.fixture
def state_large(problem_description_large, env_specification_large):
    return PSPState(
        env_specification_large,
        problem_description_large,
        problem_description_large.train_psps[0],
        deterministic=True,
        observe_conflicts_as_cliques=False,
    )


@pytest.fixture
def state_small_preclique(problem_description_small, env_specification_small):
    return PSPState(
        env_specification_small,
        problem_description_small,
        problem_description_small.train_psps[0],
        deterministic=True,
        observe_conflicts_as_cliques=True,
    )


@pytest.fixture
def psp_agent(env_specification_small, agent_specification):
    return PSPAgent(
        env_specification=env_specification_small,
        agent_specification=agent_specification,
    )


@pytest.fixture
def training_specification():
    return TrainingSpecification(
        total_timesteps=10,
        n_validation_env=2,
        fixed_validation=True,
        fixed_random_validation=True,
        validation_batch_size=2,
        validation_freq=1,
        display_env="test",
        path="saved_networks/test/",
        custom_heuristic_name="None",
        ortools_strategy="optimistic",
        max_time_ortools=10,
        scaling_constant_ortools=1,
        vecenv_type="subproc",
        validate_on_total_data=False,
    )
