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
            
            # Collect metrics from results
            for _, fit_res in results:
                if fit_res.metrics:
                    metrics = fit_res.metrics
                    train_losses.append(metrics.get("loss", 0))
                    train_accuracies.append(metrics.get("accuracy", 0))
            
            # Calculate and store average metrics
            if train_losses and train_accuracies:
                avg_loss = np.mean(train_losses)
                avg_accuracy = np.mean(train_accuracies)
                self.metrics_dict['train_loss'].append(avg_loss)
                self.metrics_dict['train_accuracy'].append(avg_accuracy)
                print(f"Round {rnd} average training loss: {avg_loss:.4f}")
                print(f"Round {rnd} average training accuracy: {avg_accuracy:.4f}")
            
            # Save aggregated weights
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            return None, {}
        
        val_losses = []
        val_accuracies = []
        
        # Collect metrics from evaluation results
        for _, eval_res in results:
            if eval_res.metrics:
                val_losses.append(eval_res.loss)
                val_accuracies.append(eval_res.metrics.get("accuracy", 0))
        
        # Calculate and store average validation metrics
        if val_losses and val_accuracies:
            avg_val_loss = np.mean(val_losses)
            avg_val_accuracy = np.mean(val_accuracies)
            self.metrics_dict['val_loss'].append(avg_val_loss)
            self.metrics_dict['val_accuracy'].append(avg_val_accuracy)
            print(f"Round {rnd} average validation loss: {avg_val_loss:.4f}")
            print(f"Round {rnd} average validation accuracy: {avg_val_accuracy:.4f}")
            return avg_val_loss, {"accuracy": avg_val_accuracy}
        
        return 0, {}


if __name__ == "__main__":
    # Instantiate the SaveModelStrategy
    strategy = SaveModelStrategy()
    
    # Start the Flower server
    fl.server.start_server(
        server_address="localhost:" + str(sys.argv[1]),
        config=fl.server.ServerConfig(num_rounds=4),
        grpc_max_message_length=1024 * 1024 * 1024,
        strategy=strategy
    )
    

