package MachineLearning.WeightedMajority;

import MachineLearning.*;

public class WeightedMajority implements Learner {
  private final boolean print_verbose;
  private double w[];
  
  // constructor for WeightedMajority
  public WeightedMajority(DataSet train, boolean print_verbose) { 
    this.print_verbose = print_verbose;
    int T = 1;
    this.w = new double[train.NBinaryAttributes()];
    for (int i = 0; i < w.length; ++i) {
      w[i] = 1.0;
    }
      
    // train on the DataSet
    for (int t = 0; t < train.NDataPoints(); ++t) {
      DataPoint data_point = train.KthBinaryDataPoint(t);
      int[] x = data_point.Attributes();
      int y = data_point.Label();
      
      double epsilon = 0.25;
      
      // calculate weight for both 0 and 1
      double w0 = 0.0;
      double w1 = 0.0;
      for (int i = 0; i < x.length; ++i) {
        if (x[i] == 0) w0 += w[i];
        else w1 += w[i];
      }
      
      // update weights
      for (int i = 0; i < x.length; ++i) {
        if (x[i] != y) w[i] = (1 - epsilon) * w[i];
      }
    }
  }
  
  // classify a particular data
  public int[] Classify(DataSet test) {
    int[] labels = new int[test.NDataPoints()];
    for (int i = 0; i < test.NDataPoints(); ++i) {
      int[] x = test.KthBinaryDataPoint(i).Attributes();
      double w0 = 0.0;
      double w1 = 0.0;
      for (int j = 0; j < x.length; ++j) {
        if (x[j] == 0) w0 += w[j];
        else w1 += w[j];
      }
      if (w0 > w1) labels[i] = 0;
      else labels[i] = 1;
    }
    return labels;
  }
}
