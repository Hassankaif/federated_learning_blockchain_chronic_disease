import flwr as fl
import sys
import numpy as np
import os
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Lists to store results for visualization and text output
accuracies = []
losses = []

def evaluate_metrics_aggregation_fn(metrics):
    # Debug the structure of metrics
    print("DEBUG: metrics structure ->", metrics)
    
    # Extract accuracies and losses from the second element of each tuple
    accuracies_data = [m[1]["accuracy"] for m in metrics if "accuracy" in m[1]]
    losses_data = [m[1]["loss"] for m in metrics if "loss" in m[1]]
    
    # Aggregate metrics (taking the average)
    avg_accuracy = sum(accuracies_data) / len(accuracies_data) if accuracies_data else 0.0
    avg_loss = sum(losses_data) / len(losses_data) if losses_data else 0.0

    # Store for visualization and file output
    accuracies.append(avg_accuracy)
    losses.append(avg_loss)
    
    return {"accuracy": avg_accuracy, "loss": avg_loss}

# Define the custom strategy
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

def aggregate_evaluate(self, rnd, results, failures):
    # Aggregate the evaluation results (accuracy and loss)
    if not results:
        return None, None
    
    # Initialize lists for accuracy and loss
    accuracies = []
    losses = []

    for result in results:
        # result is an EvaluateRes object
        accuracies.append(result.metrics["accuracy"])  # Access metrics as a dictionary
        losses.append(result.loss)  # Access loss

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_loss = sum(losses) / len(losses)
    return avg_loss, {"accuracy": avg_accuracy}


# Save the summary to a text file
def save_results_to_file():
    with open("final_results.txt", "w") as f:
        f.write("Round-wise Accuracy and Loss:\n")
        for i in range(len(accuracies)):
            f.write(f"Round {i+1}: Accuracy = {accuracies[i]:.4f}, Loss = {losses[i]:.4f}\n")

    print("Results saved to 'final_results.txt'.")

# Plot the accuracy and loss curves
def plot_results():
    plt.figure(figsize=(10, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(accuracies) + 1), accuracies, label="Accuracy", color="blue")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Rounds")
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(losses) + 1), losses, label="Loss", color="red")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Loss over Rounds")
    
    # Show plots
    plt.tight_layout()
    plt.show()

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

    # After the rounds finish, save the results and plot
    save_results_to_file()
    plot_results()
