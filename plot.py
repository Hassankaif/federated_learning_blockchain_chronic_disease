import matplotlib.pyplot as plt

# Initialize lists to store accuracy values
train_accuracy = []
val_accuracy = []
train_loss = []
val_loss = []

# Read the metrics from the text file
with open('metrics_log.txt', 'r') as file:
    for line in file:
        # Extract Train Accuracy, Val Accuracy, Train Loss, and Val Loss from each line
        parts = line.split('|')
        train_acc = float(parts[0].split(':')[1].strip())
        val_acc = float(parts[1].split(':')[1].strip())
        train_los = float(parts[2].split(':')[1].strip())
        val_los = float(parts[3].split(':')[1].strip())
        
        # Append the values to the lists
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)
        train_loss.append(train_los)
        val_loss.append(val_los)

# Plotting the Accuracy
plt.figure(figsize=(10, 6))
plt.plot(train_accuracy, label='Train Accuracy', color='blue')
plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the Loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Train Loss', color='red')
plt.plot(val_loss, label='Validation Loss', color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
