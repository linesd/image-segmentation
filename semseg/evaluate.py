import torch

class Evaluator:
    """
        Class to handle evaluation of model.
        Parameters
        ----------
        model: CNN.

        save_dir : str, optional
            Directory for saving logs.
    """
    def __init__(self, model, num_classes):
        self.model = model
        self.num_classes = num_classes

    def __call__(self, data_loader):
        """
        Compute test accuracy.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        is_accuracy: bool, optional
            Whether to compute and store the test accuracy.
        """

        correct = 0
        total = 0
        class_correct = list(0. for i in range(self.num_classes))
        class_total = list(0. for i in range(self.num_classes))
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()
                for i in range(self.num_classes):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        accuracy = 100 * correct / total
        class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(self.num_classes)]

        return accuracy, class_accuracy