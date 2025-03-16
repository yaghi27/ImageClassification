import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import Logger
from torch.utils.tensorboard import SummaryWriter

class ModelTrainer:
    def __init__(self, data_loader, model, device, learning_rate=0.001, num_epochs=25):

        self.device = device
        self.num_epochs = num_epochs

        self.train_loader, self.test_loader = data_loader.load_data() # Get data from data loader class

        # Model, loss, and optimizer
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss() # Loss function used for multiclass classification
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Logger, TensorBoard
        self.logger = Logger.Logger('ModelTrainer')
        self.writer = SummaryWriter(log_dir='runs/ModelTrainer')


    def train(self):
        self.model.train()  # Set model to training mode

        for epoch in range(self.num_epochs):
            running_loss = 0.0  # Accumulate loss for the epoch
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device) # Send data to device

                self.optimizer.zero_grad()
                outputs = self.model(inputs) # get the model's prediction
                loss = self.criterion(outputs, labels) # calculate loss
                loss.backward() # Backward propagation
                self.optimizer.step() # Updata model's parameters
                running_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                self.writer.add_scalar('Training Loss (Batch)', loss.item(), epoch * len(self.train_loader) + i)

                # # Log progress every 250 batches
                # if (i + 1) % 250 == 0:
                #     avg_loss = running_loss / 250
                #     avg_accuracy = (correct / total) * 100
                #     self.logger.log_info(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")
                #     running_loss = 0.0  # Reset batch loss accumulator

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = (correct / total) * 100

            # Log epoch-level loss and accuracy to TensorBoard
            self.writer.add_scalar('Training Loss (Epoch)', epoch_loss, epoch)
            self.writer.add_scalar('Training Accuracy (Epoch)', epoch_accuracy, epoch)

            # Log epoch results
            self.logger.log_info(f"[Epoch {epoch + 1}] Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Save the trained model
        torch.save(self.model.state_dict(), "object_recognition_model.pth")
        self.logger.log_info("Model saved as 'object_recognition_model.pth'")

        # Close the writer to finalize TensorBoard logs
        self.writer.close()


    def evaluate(self):
        """Evaluate the model."""
        self.model.eval()  # Set the model to evaluation mode
        total = 0
        correct = 0
        total_loss = 0.0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Log batch-level loss to TensorBoard
                self.writer.add_scalar('Evaluation Loss (Batch)', loss.item(), i)

        # Calculate average loss and accuracy
        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total * 100

        # Log epoch-level evaluation results to TensorBoard
        self.writer.add_scalar('Evaluation Loss (Epoch)', avg_loss)
        self.writer.add_scalar('Evaluation Accuracy (Epoch)', accuracy)

        # Log evaluation results
        self.logger.log_info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

        return avg_loss, accuracy
