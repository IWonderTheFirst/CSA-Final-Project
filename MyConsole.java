public class MyConsole {
  public static void main(String[] args) {
    double[][] X = {
      // clear positives
      {0.50, 0.50, 0.50, 1.00},
      {1.00, 0.30, 0.70, 1.20},
      {0.20, 0.80, 0.40, 2.00},
      {1.50, 0.20, 0.60, 2.50},
  
      // clear negatives
      {2.00, 1.50, 1.20, 0.50},
      {1.50, 1.80, 0.90, 1.00},
      {2.20, 1.20, 1.50, 0.30},
      {1.80, 1.60, 1.10, 0.80},
  
      // very close pairs around boundary
      {1.00, 1.00, 1.00, 0.5350},
      {1.00, 1.00, 1.00, 0.5450},
  
      {1.20, 0.90, 1.10, 0.3750},
      {1.20, 0.90, 1.10, 0.3850},
  
      {0.80, 1.20, 0.70, 1.7550},
      {0.80, 1.20, 0.70, 1.7650},
  
      {1.50, 0.80, 1.00, 0.4550},
      {1.50, 0.80, 1.00, 0.4650},
  
      {0.70, 1.10, 1.30, 1.9750},
      {0.70, 1.10, 1.30, 1.9850},
  
      {1.30, 1.00, 0.80, 0.8350},
      {1.30, 1.00, 0.80, 0.8450},
  
      {0.90, 0.95, 1.20, 1.3750},
      {0.90, 0.95, 1.20, 1.3850},
  
      {1.60, 0.70, 0.90, 0.2450},
      {1.60, 0.70, 0.90, 0.2550},
  
      {0.60, 1.30, 0.60, 2.4150},
      {0.60, 1.30, 0.60, 2.4250},
  
      {1.10, 1.10, 0.90, 0.8750},
      {1.10, 1.10, 0.90, 0.8850}
    };
    
    int[] y = {
        1, 1, 1, 1,
        0, 0, 0, 0,
    
        0, 1,
        0, 1,
        0, 1,
        0, 1,
        0, 1,
        0, 1,
        0, 1,
        0, 1,
        0, 1,
        0, 1
    };

    Dataset data = new Dataset(X, y);
    data.addNoise(0.1);
    
    //Instantiating the SVM class with the X, y dataset
    SVM model = new SVM(data);

    double[][] test_X = {{1, 0.5}}; //Creating a test set with one example of 
    
    int[] predictionA = model.predict(test_X);
    System.out.println("Prediction Before: " + predictionA[0]); //Making a prediction with the weights before training

    model.fit(new double[X[0].length + 1], 10, 10, 0.5, 10); //Fitting the model for 5 epochs, rangeMult=0.5, numRange=20, and range=5

    int[] predictionB = model.predict(test_X); //Making a prediction with the weights after training
    System.out.println("Prediction After: " + predictionB[0]);
  }
}
