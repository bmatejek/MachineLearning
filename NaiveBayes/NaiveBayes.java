package MachineLearning.NaiveBayes;

import java.util.Vector;
import MachineLearning.*;

public class NaiveBayes implements Learner {
  private final boolean print_verbose;
  private DataSet train;
  private double[] C;
  private double[][][] x_given_Ck;
  
  // constructor for NaiveBayes
  public NaiveBayes(DataSet train, boolean print_verbose) { 
    this.print_verbose = print_verbose;
    this.train = train;
    
    // train on the DataSet
    C = new double[train.NLabels()];
    x_given_Ck = new double[train.NAttributes()][train.MaxClassesInAttribute()][train.NLabels()];
    for (int i = 0; i < train.NDataPoints(); ++i) {
      DataPoint data_point = train.KthDataPoint(i);
      C[data_point.Label()]++;
      for (int k = 0; k < train.NAttributes(); ++k) {
        x_given_Ck[k][data_point.KthAttribute(k)][data_point.Label()]++;
      }
    }
    
    // calculate P(x_i | C_k)
    for (int i = 0; i < train.NAttributes(); ++i) {
      for (int j = 0; j < train.NClassesKthAttribute(i); ++j) {
        for (int k = 0; k < train.NLabels(); ++k) {
          x_given_Ck[i][j][k] /= C[k];
        }
      }
    }
    
    // turn C into probabilities
    for (int k = 0; k < train.NLabels(); ++k) 
      C[k] /= train.NDataPoints();
  }
  
  // classify a particular data
  public int[] Classify(DataSet test) {
    int[] labels = new int[test.NDataPoints()];
    
    // go through each data point
    for (int j = 0; j < test.NDataPoints(); ++j) {
      DataPoint data_point = test.KthDataPoint(j);
      int label = 0;
      double max = Double.MIN_VALUE;
      // go through each of the possqble labels
      for (int k = 0; k < test.NLabels(); ++k) {
        double prod = C[k];
        // go through each attribute
        for (int n = 0; n < test.NAttributes(); ++n) {
          int i = data_point.KthAttribute(n);   
          prod *= x_given_Ck[n][i][k];
        }
        if (prod > max) {
          max = prod;
          label = k;
        }
      }
      labels[j] = label;
    }
    
    return labels;
  }
}