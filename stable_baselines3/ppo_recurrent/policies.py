from stable_baselines3.common.recurrent.policies import (
    GRUActorCriticCNNPolicy,
    GRUActorCriticFlattenPolicy,
    GRUActorCriticMultiInputPolicy,
)

MlpPolicy = GRUActorCriticFlattenPolicy
CnnPolicy = GRUActorCriticCNNPolicy
MultiInputPolicy = GRUActorCriticMultiInputPolicy
