from dataclasses import dataclass

@dataclass
class RealWorldConfig:
    """
    Configuration pour un déploiement fédéré en « real world ».

    Attributes:
        server_address: Adresse et port du serveur Flower (ex: "0.0.0.0:8080").
        num_rounds: Nombre total de tours (rounds) de fédération.
        fraction_fit: Fraction des clients participant à la phase de fit chaque round.
        fraction_eval: Fraction des clients participant à la phase d'évaluation chaque round.
        min_fit_clients: Nombre minimum de clients requis pour lancer la phase de fit.
        min_eval_clients: Nombre minimum de clients requis pour la phase d'évaluation.
    """
    server_address: str
    num_rounds: int
    fraction_fit: float
    fraction_eval: float
    min_fit_clients: int
    min_eval_clients: int
