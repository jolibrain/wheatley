import torch


class AgentSpecification:
    def __init__(
        self,
        lr,
        fe_lr,
        n_steps_episode,
        batch_size,
        n_epochs,
        gamma,
        clip_range,
        target_kl,
        ent_coef,
        vf_coef,
        normalize_advantage,
        optimizer,
        freeze_graph,
        n_features,
        gconv_type,
        graph_has_relu,
        graph_pooling,
        layer_pooling,
        mlp_act,
        mlp_act_graph,
        n_workers,
        device,
        n_mlp_layers_features_extractor,
        n_layers_features_extractor,
        hidden_dim_features_extractor,
        n_attention_heads,
        reverse_adj,
        residual_gnn,
        normalize_gnn,
        conflicts,
        n_mlp_layers_shared,
        hidden_dim_shared,
        n_mlp_layers_actor,
        hidden_dim_actor,
        n_mlp_layers_critic,
        hidden_dim_critic,
        fe_type,
        transformer_flavor,
        dropout,
        cache_lap_node_id,
        lap_node_id_k,
        rpo,
        rpo_smoothing_param,
    ):
        self.lr = lr
        self.fe_lr = fe_lr
        self.n_steps_episode = n_steps_episode
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.clip_range = clip_range
        self.target_kl = target_kl
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.normalize_advantage = normalize_advantage
        self.optimizer = optimizer
        self.freeze_graph = freeze_graph
        self.n_features = n_features
        self.gconv_type = gconv_type
        self.graph_has_relu = graph_has_relu
        self.graph_pooling = graph_pooling
        self.layer_pooling = layer_pooling
        self.mlp_act = mlp_act
        self.mlp_act_graph = mlp_act_graph
        self.n_workers = n_workers
        self.device = device
        self.n_mlp_layers_features_extractor = n_mlp_layers_features_extractor
        self.n_layers_features_extractor = n_layers_features_extractor
        self.hidden_dim_features_extractor = hidden_dim_features_extractor
        self.n_mlp_layers_shared = n_mlp_layers_shared
        self.hidden_dim_shared = hidden_dim_shared
        self.n_attention_heads = n_attention_heads
        self.reverse_adj = reverse_adj
        self.residual_gnn = residual_gnn
        self.normalize_gnn = normalize_gnn
        self.conflicts = conflicts
        self.n_mlp_layers_actor = n_mlp_layers_actor
        self.hidden_dim_actor = hidden_dim_actor
        self.n_mlp_layers_critic = n_mlp_layers_critic
        self.hidden_dim_critic = hidden_dim_critic
        self.fe_type = fe_type
        self.transformer_flavor = transformer_flavor
        self.dropout = dropout
        self.cache_lap_node_id = cache_lap_node_id
        self.lap_node_id_k = lap_node_id_k
        self.rpo = rpo
        self.rpo_smoothing_param = rpo_smoothing_param

        if mlp_act.lower() == "relu":
            self.activation_fn = torch.nn.LeakyReLU
        elif mlp_act.lower() == "tanh":
            self.activation_fn = torch.nn.Tanh
        elif mlp_act.lower() == "elu":
            self.activation_fn = torch.nn.ELU
        elif mlp_act.lower() == "gelu":
            self.activation_fn = torch.nn.GELU
        elif mlp_act.lower() == "selu":
            self.activation_fn = torch.nn.SELU
        else:
            raise Exception("Activation not recognized")

        self.mlp_act_graph = mlp_act_graph
        if mlp_act_graph.lower() == "relu":
            self.activation_fn_graph = torch.nn.LeakyReLU
        elif mlp_act_graph.lower() == "tanh":
            self.activation_fn_graph = torch.nn.Tanh
        elif mlp_act_graph.lower() == "elu":
            self.activation_fn_graph = torch.nn.ELU
        elif mlp_act_graph.lower() == "gelu":
            self.activation_fn_graph = torch.nn.GELU
        elif mlp_act_graph.lower() == "selu":
            self.activation_fn_graph = torch.nn.SELU
        else:
            raise Exception("Activation not recognized")

        if optimizer.lower() == "adam":
            self.optimizer_class = torch.optim.Adam
        elif optimizer.lower() == "sgd":
            self.optimizer_class = torch.optim.SGD
        elif optimizer.lower() == "adamw":
            self.optimizer_class = torch.optim.AdamW
        elif optimizer.lower() == "radam":
            self.optimizer_class = torch.optim.RAdam
        else:
            raise Exception("Optimizer not recognized")

        shared = [hidden_dim_shared] * n_mlp_layers_shared
        pi = [hidden_dim_actor] * n_mlp_layers_actor
        vf = [hidden_dim_critic] * n_mlp_layers_critic
        self.net_arch = shared + [dict(vf=vf, pi=pi)]

    def print_self(self):
        print(
            f"==========Agent Description   ==========\n"
            f"Learning rate:                    {self.lr}\n"
            f"Number steps per episode:         {self.n_steps_episode}\n"
            f"Batch size:                       {self.batch_size}\n"
            f"Number of epochs:                 {self.n_epochs}\n"
            f"Discount factor (gamma):          {self.gamma}\n"
            f"Entropy coefficient:              {self.ent_coef}\n"
            f"Value function coefficient:       {self.vf_coef}\n"
            f"Normalize advantage:              {self.normalize_advantage}\n"
            f"Optimizer:                        {self.optimizer}\n"
            f"Features extractor type:          {self.fe_type}\n"
            f"Layer Pooling:                    {self.layer_pooling}\n"
            f"Dropout:                          {self.dropout}\n"
            f"RPO:                              {self.rpo}\n"
            f"RPO smoothing:                    {self.rpo_smoothing_param}\n"
        )
        if self.fe_type == "tokengt":
            print(f"Net shapes:")
            shape = f"{self.n_features} -> ( {self.hidden_dim_features_extractor} / {self.n_attention_heads} ) x {self.n_layers_features_extractor}"

            shared_shape = f"" + "".join([f" -> {self.hidden_dim_shared}" for _ in range(self.n_mlp_layers_shared)])
            actor_shape = f"" + "".join([f" -> {self.hidden_dim_actor}" for _ in range(self.n_mlp_layers_actor)]) + " -> 1"
            critic_shape = (
                f"" + "".join([f" -> {self.hidden_dim_critic}" for _ in range(self.n_mlp_layers_critic)]) + " -> 1"
            )
            print(f" - Features extractor: TokenGT/{self.mlp_act_graph} {shape}")
            print(f" - Shared: {shared_shape}")
            print(f" - Actor: {actor_shape}")
            print(f" - Critic: {critic_shape}\n")
        else:
            print(
                f"Convolutional FE\n"
                f"Freezing graph during training:   {'Yes' if self.freeze_graph else 'No'}\n"
                f"Graph convolution type:           {self.gconv_type.upper()}\n"
                f"Add (R)eLU between graph layers:  {'Yes' if self.graph_has_relu else 'No'}\n"
                f"Graph pooling type:               {self.graph_pooling.title()}\n"
                f"Activation function of agent:     {self.mlp_act.title()}\n"
                f"Activation function of graph:     {self.mlp_act_graph.title()}\n"
                f"Net shapes:"
            )
            first_features_extractor_shape = f"{self.n_features} -> {self.hidden_dim_features_extractor}"

            other_features_extractor_shape = f"{self.hidden_dim_features_extractor} -> {self.hidden_dim_features_extractor}"

            shared_shape = f"" + "".join([f" -> {self.hidden_dim_shared}" for _ in range(self.n_mlp_layers_shared)])
            actor_shape = f"" + "".join([f" -> {self.hidden_dim_actor}" for _ in range(self.n_mlp_layers_actor)]) + " -> 1"
            critic_shape = (
                f"" + "".join([f" -> {self.hidden_dim_critic}" for _ in range(self.n_mlp_layers_critic)]) + " -> 1"
            )
            print(
                f" - Features extractor: {self.gconv_type.upper()}({first_features_extractor_shape}) => "
                + f"{self.gconv_type.upper()}({other_features_extractor_shape}) x {self.n_layers_features_extractor - 1}"
            )
            print(f" - Shared: {shared_shape}")
            print(f" - Actor: {actor_shape}")
            print(f" - Critic: {critic_shape}\n")
