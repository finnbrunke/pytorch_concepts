"""
Time series dataset for predator-prey dynamics.
Using Lotka-Volterra equations to generate synthetic data for predator-prey interactions.
Annotated with concepts and providing STL (Signal Temporal Logic) rules.
"""

import numpy as np
import torch
from ..base.temporal_dataset import TemporalConceptDatasetWithRules
from ...annotations import Annotations
from ...tensor import AnnotatedTensor
from scipy.integrate import solve_ivp


class PredatorPreyDataset(TemporalConceptDatasetWithRules):
    def __init__(self, num_samples=1000, time_steps=100, dt=0.1):
        """
        Args:
            num_samples (int): Number of samples to generate.
            time_steps (int): Number of time steps in each sample.
            dt (float): Time step size.
            alpha (float): Prey growth rate.
            beta (float): Predation rate.
            delta (float): Predator growth rate per prey eaten.
            gamma (float): Predator death rate.
        """
        self.num_samples = num_samples
        self.time_steps = time_steps
        self.dt = dt

        self.concept_labels = ["bounty", "depletion", "predator_bloom", "system_crash"]

        self.rulebook = [
            {
                "name": "depletion_implies_future_bounty",
                "task": "prey_recovery",
                "kind": "always_implies_eventually",
                "antecedent": {"concept": "depletion", "threshold": 0.5},
                "consequent": {"concept": "bounty", "threshold": 0.5},
                "interval": [0, 15],
            },
            {
                "name": "bloom_then_crash",
                "task": "predator_crash",
                "kind": "eventually_and_eventually",
                "first": {"concept": "predator_bloom", "threshold": 0.5},
                "second": {"concept": "system_crash", "threshold": 0.5},
                "interval": [0, 10],
            },
            {
                "name": "no_system_crash",
                "task": "system_stability",
                "kind": "always_not",
                "predicate": {"concept": "system_crash", "threshold": 0.5},
            },
        ]

        self.concept_annotations = Annotations(
            labels=self.concept_labels, 
            cardinalities=[1] * len(self.concept_labels), 
            types=["continuous"] * len(self.concept_labels)
        )

        self.trajectories, self.concepts, self.robustness = self._generate_dataset()

        super().__init__(
            input_data=self.trajectories,
            concepts=AnnotatedTensor(
                self.concepts,
                self.concept_annotations,
                axis=-1,
            ),
            rulebook=self.rulebook,
            precision=32,
            name="PredatorPrey"
        )

    def _generate_dataset(self):
        trajectories = []
        concepts_tensors = []
        robustness = []

        t_eval = np.linspace(0, self.time_steps * self.dt, self.time_steps)

        for _ in range(self.num_samples):
            alpha = np.random.uniform(0.1, 1.5)
            beta = np.random.uniform(0.1, 1.5)
            delta = np.random.uniform(0.1, 1.5)
            gamma = np.random.uniform(0.1, 1.5)

            prey_init = np.random.uniform(2.0, 8.0)
            predator_init = np.random.uniform(1.0, 5.0) 

            sol = solve_ivp(
                fun=self._lotka_volterra_ode,
                t_span=(0, self.time_steps * self.dt),
                y0=[prey_init, predator_init],
                t_eval=t_eval,
                args=(alpha, beta, delta, gamma)
            )

            traj = sol.y.T  
            if np.isnan(traj).any() or len(traj) != self.time_steps:
                continue 

            conc_dict = self._compute_concepts(traj)
            
            conc_tensor = torch.tensor(
                np.stack([conc_dict[name] for name in self.concept_labels], axis=-1), 
                dtype=torch.float32
            )

            robustness_scores = self._compute_robustness(conc_tensor)

            trajectories.append(traj)
            concepts_tensors.append(conc_tensor)
            robustness.append(robustness_scores)

        return np.array(trajectories, dtype=np.float32), \
               torch.stack(concepts_tensors), \
               np.array(robustness, dtype=np.float32)
        

    def _lotka_volterra_ode(self, t, y, alpha, beta, delta, gamma):
        prey, predator = y
        dprey_dt = alpha * prey - beta * prey * predator
        dpredator_dt = delta * prey * predator - gamma * predator
        return [dprey_dt, dpredator_dt]

    def _sigmoid(self, x, scale=2.0):
        return 1.0 / (1.0 + np.exp(-scale * x))
    
    def _compute_concepts(self, traj):
        x = traj[:, 0]
        y = traj[:, 1]
        
        return {
            "bounty": self._sigmoid(x - 5.0),
            "depletion": self._sigmoid(2.5 - x),
            "predator_bloom": self._sigmoid(y - 4.5),
            "system_crash": self._sigmoid(0.4 - np.minimum(x, y))
        }

    def _compute_robustness(self, conc_tensor):
        values = {name: conc_tensor[:, index] for index, name in enumerate(self.concept_labels)}
        def future_max(signal, end):
            return torch.stack(
                [
                    signal[index:min(index + end + 1, signal.shape[0])].max()
                    for index in range(min(end + 1, signal.shape[0]))
                ]
            )
        scores = []
        depletion = values["depletion"] - 0.5
        bounty = values["bounty"] - 0.5
        bloom = values["predator_bloom"] - 0.5
        crash = values["system_crash"] - 0.5
        scores.append(torch.min(torch.maximum(-depletion[:16], future_max(bounty, 15))).item())
        scores.append(torch.max(torch.minimum(bloom[:11], future_max(crash, 10))).item())
        scores.append(torch.min(-crash).item())
        return np.asarray(scores, dtype=np.float32)
    
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample["robustness"] = self.robustness[idx]
        return sample


if __name__ == "__main__":
    dataset = PredatorPreyDataset(num_samples=10, time_steps=100, dt=0.1)
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"Sample {i}:")
        print("Signal shape:", sample["inputs"]["x"].shape)
        print("Concepts shape:", sample["temporal_concepts"].shape)
        print("Robustness shape:", sample["robustness"].shape)
        