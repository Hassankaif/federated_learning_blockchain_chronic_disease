import flwr as fl
import sys
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

if __name__ == "__main__":
    # Start Flower server
    strategy = SaveModelStrategy()
    fl.server.start_server(
        server_address="localhost:" + str(sys.argv[1]),
        config=fl.server.ServerConfig(num_rounds=3),
        grpc_max_message_length=1024 * 1024 * 1024,
        strategy=strategy
    )

