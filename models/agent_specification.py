import torch


class AgentSpecification:
    def __init__(
        self,
        lr,
        n_steps_episode,
        batch_size,
        n_epochs,
        gamma,
        clip_range,
        target_kl,
        ent_coef,
        vf_coef,
        optimizer,
        freeze_graph,
        n_features,
        gconv_type,
        graph_has_relu,
        graph_pooling,
        mlp_act,
        n_workers,
        device,
        n_mlp_layers_features_extractor,
        n_layers_features_extractor,
        hidden_dim_features_extractor,
        n_attention_heads,
        n_mlp_layers_actor,
        hidden_dim_actor,
        n_mlp_layers_critic,
        hidden_dim_critic,
    ):
        self.lr = lr
        self.n_steps_episode = n_steps_episode
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.clip_range = clip_range
        self.target_kl = target_kl
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.optimizer = optimizer
        self.freeze_graph = (freeze_graph,)
        self.n_features = n_features
        self.gconv_type = gconv_type
        self.graph_has_relu = graph_has_relu
        self.graph_pooling = graph_pooling
        self.mlp_act = mlp_act
        self.n_workers = n_workers
        self.device = device
        self.n_mlp_layers_features_extractor = n_mlp_layers_features_extractor
        self.n_layers_features_extractor = n_layers_features_extractor
        self.hidden_dim_features_extractor = hidden_dim_features_extractor
        self.n_attention_heads = n_attention_heads
        self.n_mlp_layers_actor = n_mlp_layers_actor
        self.hidden_dim_actor = hidden_dim_actor
        self.n_mlp_layers_critic = n_mlp_layers_critic
        self.hidden_dim_critic = hidden_dim_critic

        if optimizer.lower() == "adam":
            self.optimizer_class = torch.optim.Adam
        elif optimizer.lower() == "sgd":
            self.optimizer_class = torch.optim.SGD
        else:
            raise Exception("Optimizer not recognized")

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
            f"Optimizer:                        {self.optimizer}\n"
            f"Freezing graph during training:   {'Yes' if self.freeze_graph else 'No'}\n"
            f"Graph convolution type:           {self.gconv_type.upper()}\n"
            f"Add (R)eLU between graph layers:  {'Yes' if self.graph_has_relu else 'No'}\n"
            f"Graph pooling type:               {self.graph_pooling.title()}\n"
            f"Activation function of actor:     {self.mlp_act.title()}\n"
            f"Net shapes:"
        )
        first_features_extractor_shape = f"{self.n_features}" + "".join(
            [f" -> {self.hidden_dim_features_extractor}" for _ in range(self.n_mlp_layers_features_extractor - 1)]
        )
        other_features_extractor_shape = f"{self.hidden_dim_features_extractor}" + "".join(
            [f" -> {self.hidden_dim_features_extractor}" for _ in range(self.n_mlp_layers_features_extractor - 1)]
        )
        actor_shape = (
            f"{self.hidden_dim_features_extractor * 2}"
            + "".join([f" -> {self.hidden_dim_actor}" for _ in range(self.n_mlp_layers_actor - 2)])
            + " -> 1"
        )
        critic_shape = (
            f"{self.hidden_dim_features_extractor}"
            + "".join([f" -> {self.hidden_dim_critic}" for _ in range(self.n_mlp_layers_critic - 2)])
            + " -> 1"
        )
        print(
            f" - Features extractor: {self.gconv_type.upper()}({first_features_extractor_shape}) => "
            + f"{self.gconv_type.upper()}({other_features_extractor_shape}) x {self.n_layers_features_extractor - 1}"
        )
        print(f" - Actor: {actor_shape}")
        print(f" - Critic: {critic_shape}\n")
