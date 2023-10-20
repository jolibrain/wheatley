from psp.models.gnn_dgl import GnnDGL
from psp.env.observation import EnvObservation
from psp.models.agent_observation import AgentObservation


def test_gnn_dgl(env_specification_small, agent_specification, state_small):
    s = state_small
    env_specification = env_specification_small
    gnn = PSPGnnDGL(
        input_dim_features_extractor=env_specification.n_features,
        graph_pooling=agent_specification.graph_pooling,
        max_n_nodes=env_specification.max_n_nodes,
        max_n_resources=env_specification.max_n_resources,
        n_mlp_layers_features_extractor=agent_specification.n_mlp_layers_features_extractor,
        n_layers_features_extractor=agent_specification.n_layers_features_extractor,
        hidden_dim_features_extractor=agent_specification.hidden_dim_features_extractor,
        activation_features_extractor=agent_specification.activation_fn_graph,
        n_attention_heads=agent_specification.n_attention_heads,
        residual=agent_specification.residual_gnn,
        normalize=agent_specification.normalize_gnn,
        conflicts=agent_specification.conflicts,
    )

    (
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ) = s.to_features_and_edge_index(False)

    eog = EnvObservation(
        s.env_specification,
        s.problem["n_jobs"],
        s.n_nodes,
        s.problem["n_resources"],
        s.problem_description.max_n_jobs,
        s.problem_description.max_n_modes,
        s.env_specification.max_n_resources,
        s.env_specification.max_edges_factor,
        features,
        problem_edge_index,
        resource_conf_edges,
        resource_conf_att,
        resource_prec_edges,
        resource_prec_att,
    ).to_gym_observation()

    eogt = AgentObservation.np_to_torch(eog)

    r = gnn(eogt)
    assert r.shape[0] == 1
    assert r.shape[1] == 8
    assert (
        r.shape[2]
        == (
            env_specification.n_features
            + agent_specification.hidden_dim_features_extractor
            * (agent_specification.n_layers_features_extractor + 1)
        )
        * 2
    )
