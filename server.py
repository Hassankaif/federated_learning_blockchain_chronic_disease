import flwr as fl
import sys
import numpy as np
import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def evaluate_metrics_aggregation_fn(metrics):
    # Debug the structure of metrics
    print("DEBUG: metrics structure ->", metrics)
    
    # Extract accuracies from the second element of each tuple
    accuracies = [m[1]["accuracy"] for m in metrics if "accuracy" in m[1]]
    return {"accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0}



# Define the custom strategy
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

# Main block
if __name__ == "__main__":
    # Start Flower server with custom strategy
    strategy = SaveModelStrategy(evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
    fl.server.start_server(
        server_address="localhost:" + str(sys.argv[1]),
        config=fl.server.ServerConfig(num_rounds=3),
        grpc_max_message_length=1024 * 1024 * 1024,
        strategy=strategy
    )
