public class Accuracy {
  private double loss;

  public Accuracy() {
    this.loss = 0;
  }

  public double update_loss(Dataset data, SVM model) {
    /*
    Updates the accuracy of the model based on the model class and a dataset.
    Instead of usi
    */
    double[][] X = data.getX();
    int[] y = data.getY();
    int [] preds = model.predict(X); //Using the model given in the parameter to test the model on the X dataset

    double correct = 0;
    double total = 0;

    for (int i = 0; i < preds.length; i++) { //Checking the accuracy of the classification
      if (preds[i] == y[i]) {
        correct += 1;
      }
      total += 1;
    }

    loss = correct / total;
    return loss; //Returns that the accuracy that it computed
  }
}
