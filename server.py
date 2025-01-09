import flwr as fl
import sys
import numpy as np
import matplotlib.pyplot as plt

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.metrics_dict = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            train_losses = []
            train_accuracies = []
            
            for _, fit_res in results:
                if fit_res.metrics:
                    metrics = fit_res.metrics
                    train_losses.append(metrics.get("loss", 0))
                    train_accuracies.append(metrics.get("accuracy", 0))
            
            if train_losses and train_accuracies:
                avg_loss = np.mean(train_losses)
                avg_accuracy = np.mean(train_accuracies)
                self.metrics_dict['train_loss'].append(avg_loss)
                self.metrics_dict['train_accuracy'].append(avg_accuracy)
                print(f"Round {rnd} average training loss: {avg_loss:.4f}")
                print(f"Round {rnd} average training accuracy: {avg_accuracy:.4f}")
            
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}
        
        val_losses = []
        val_accuracies = []
        
        for _, eval_res in results:
            if eval_res.metrics:
                val_losses.append(eval_res.loss)
                val_accuracies.append(eval_res.metrics.get("accuracy", 0))
        
        if val_losses and val_accuracies:
            avg_val_loss = np.mean(val_losses)
            avg_val_accuracy = np.mean(val_accuracies)
            self.metrics_dict['val_loss'].append(avg_val_loss)
            self.metrics_dict['val_accuracy'].append(avg_val_accuracy)
            print(f"Round {rnd} average validation loss: {avg_val_loss:.4f}")
            print(f"Round {rnd} average validation accuracy: {avg_val_accuracy:.4f}")
            return avg_val_loss, {"accuracy": avg_val_accuracy}
        
        return 0, {}

def save_results_to_file(strategy):
    with open("final_results.txt", "w") as f:
        f.write("Federated Learning Results\n")
        f.write("==========================\n\n")
        
        for round_idx in range(len(strategy.metrics_dict['train_loss'])):
            f.write(f"Round {round_idx + 1}:\n")
            f.write(f"  Training Loss: {strategy.metrics_dict['train_loss'][round_idx]:.4f}\n")
            f.write(f"  Training Accuracy: {strategy.metrics_dict['train_accuracy'][round_idx]:.4f}\n")
            f.write(f"  Validation Loss: {strategy.metrics_dict['val_loss'][round_idx]:.4f}\n")
            f.write(f"  Validation Accuracy: {strategy.metrics_dict['val_accuracy'][round_idx]:.4f}\n")
            f.write("\n")
    print("Results saved to 'final_results.txt'.")

def plot_results(strategy):
    rounds = range(1, len(strategy.metrics_dict['train_loss']) + 1)
    
    if not rounds:
        print("No metrics to plot. Training might not have completed.")
        return

    plt.figure(figsize=(12, 8))
    
    # Plot Loss
    plt.subplot(2, 1, 1)
    plt.plot(rounds, strategy.metrics_dict['train_loss'], 'ro-', label='Training Loss')
    plt.plot(rounds, strategy.metrics_dict['val_loss'], 'bo-', label='Validation Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Round')
    plt.grid(True)
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(rounds, strategy.metrics_dict['train_accuracy'], 'ro-', label='Training Accuracy')
    plt.plot(rounds, strategy.metrics_dict['val_accuracy'], 'bo-', label='Validation Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy per Round')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Plots saved as 'training_metrics.png'.")

if __name__ == "__main__":
    # Instantiate the SaveModelStrategy
    strategy = SaveModelStrategy()
    
    # Start the Flower server
    fl.server.start_server(
        server_address="localhost:" + str(sys.argv[1]),
        config=fl.server.ServerConfig(num_rounds=3),
        grpc_max_message_length=1024 * 1024 * 1024,
        strategy=strategy
    )
    
    # Save results to file and plot metrics
    save_results_to_file(strategy)
    plot_results(strategy)
