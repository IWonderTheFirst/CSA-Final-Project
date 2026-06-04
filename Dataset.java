import java.util.Random;

public class Dataset {
  private double[][] X;
  private int[] y;
  public Dataset(double[][] X, int[] y) {
    /*
    The dataset class stores X and y, so that it is more readable when I pass parameters.
    In the future, I can also implement various methods in the future (for example creating datasets)
    */
    this.X = X;
    this.y = y;
  }

  public double[][] getX() {
    return X;
  }
  public int[] getY() {
    return y;
  }

  public void addNoise(double variance) {
    /*
    Injects gaussian noise into the dataset, making the dataset more complex and not completely linearly seperable.
    It uses a double for loop (the first one can be enhanced bc an array stores a pointer) to add gaussian noise to every value in x
    */
    System.out.println("Adding gaussian noise of " + variance);
    Random rand = new Random();
    for (double[] data : X) {
      for (int i = 0; i < data.length; i++) {
        data[i] += rand.nextGaussian() * Math.sqrt(variance);
      }
    }
  }
}
