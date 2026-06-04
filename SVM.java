public class SVM {
  private double[] weights; //the parameters of the model
  private Dataset data;
  public Accuracy loss; //loss class given by accuracy
  
  public SVM(Dataset data, double[] weights) {
    /*
    Instantiates the SVM class given data, and a custom weight
    This means that the weights won't be initialized as a default zero array
    */
    this.weights = weights;
    this.data = data;
    this.loss = new Accuracy();
  }

  public SVM(Dataset data) {
    /*
    Instantiates the class with data. Sets every weight to 0 with the length according to the X dataset
    */
    this.data = data;
    this.weights = new double[data.getX()[0].length + 1];
    this.loss = new Accuracy();
  }

  public int[] predict(double[][] X) {
    /*
    Makes a prediction (inference) based on X.
    This functionality is used throughout the code, from predicting from test to training
    */
    int[] prediction = new int[X.length];
    for (int dp = 0; dp < X.length; dp++) {
      double logit = 0;
      for (int i = 0; i < X[0].length; i++) {
        logit += X[dp][i] * weights[i]; //Logits is equal to the weights times every input. This is standard for linear models
      }
      logit += weights[weights.length - 1]; //Adding a bias (which is the final weight value) on top of the logits
      if (logit > 0) { //For classification, if the output is over 0, its classified as 1 and 0 otherwise
        prediction[dp] = 1;
      }
      else {
        prediction[dp] = 0;
      }
    }

    return prediction;
  }

  public void fit(double[] startParams, double range, double numRange, double rangeMult, int epochs) {
    /*
    Fits the model given the above parameters by utilizing the getBestParams recursive method
    This fit method pritns the accuracy every epoch, notifying the user of the models' accuracy at a given step.
    */
    double accuracy = loss.update_loss(data, this); //Computing and printing original loss
    System.out.println("Starting with accuracy: " + accuracy);
    for (int epoch = 0; epoch < epochs; epoch++) {
      double[] bestParams = getBestParams(startParams, range, numRange, 0); //Getting the parameters at a given epoch
      startParams = bestParams.clone(); //Cloning to make sure they don't have the same pointer (as a safety measure)
      range *= rangeMult;

      weights = bestParams;
      accuracy = loss.update_loss(data, this);
      //Computing and printing loss after the step, along with the optimal weights at the step
      System.out.print("Epoch " + (epoch+1) + " ending with Accuracy: " + accuracy + ", Weights: [");
      for (double weight : weights) {
        System.out.print(weight + ", ");
      }
      System.out.println("]");

      if (accuracy == 1) {
        System.out.println("STOPPING DUE TO ACCURACY==1");
        break;//Stopping training at accuracy==1 because there is no point in training even more
      }
    }
  }

  public double[] getBestParams(double[] startParams, double range, double numRange, int i) {
    /*
    Recursive method to get the best params given a range of values to try
    Because its a recursive method, it calls on itself and it has a base case, which uses i (the number of times that the recursive method ran)
    */
    if (i >= weights.length) {
      return startParams; //base case. Stops once it tested every weight for the range specified
    }

    Accuracy acc = new Accuracy();
    double start = startParams[i] - range;
    double end = startParams[i] + range;
    double interval = range * 2 / numRange;

    double[] bestParams = startParams;
    double bestLoss = 0.;

    for (double j = start; j <= end; j+= interval) {
      /*
      This loop tests the j possible values for a given weight, which is given by the ith value in the weight array
      The loop calls on the method again to test the next weight for a given weight value.
      This means that this entire method scales exponrentially with the number of dimensions.
      Here, it gets the best weight parameters and then returns it.
      This means that the firs method call will get the optimal weights given specific values of the first weight
      */
      //System.out.println(j + ", " + i);
      double[] newWeight = startParams.clone();
      newWeight[i] = j;
      double[] params = getBestParams(newWeight, range, numRange, i+1);
      SVM newModel = new SVM(data, params);
      double loss = acc.update_loss(data, newModel);
      if (loss > bestLoss) {
        bestParams = params;
        bestLoss = loss; //Updates the best parameters and loss to find the most optimal ones
      }
    }

    return bestParams; //Return the best possible parameters that can obtained given the timestep i and the weights that wer already set by recursive method calls before.
  }
}
