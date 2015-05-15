package MachineLearning.NeuralNetwork;

import java.util.Collections;
import java.util.Vector;
import MachineLearning.*;

public class NeuralNetwork implements Learner {
  private final boolean print_verbose;
  private final DataSet train;
  private double[][][] w;   // weights
  private double[][] a;
  private double[][] in;
  private double[][] delta;
  private double output_in;
  private double output_a;
  private double output_delta;
  private double alpha;
  private int n;
  private int L;            // number of layers (including input and output)
  private int epochs;       // number of epochs
  
  private double g(double z) {
    return 1.0 / (1.0 + Math.exp(-z));
  }
  
  private double gprime(double z) {
    return g(z) * (1 - g(z));
  }
  
  private void PropagateForwards(int[] x, int y) {
    // a_i <- x_i
    for (int i = 0; i < n; ++i) {
      a[0][i] = x[i];
    }
    
    // for each layer l = 1 to L do
    for (int l = 1; l < L; ++l) {
      // for each node in the layer
      for (int j = 0; j < n; ++j) {
        in[l][j] = 0.0;
        for (int i = 0; i < n; ++i) {
          in[l][j] += w[l - 1][i][j] * a[l - 1][i];
        }
        a[l][j] = g(in[l][j]);
      }
    }
    
    output_in = 0.0;
    for (int i = 0; i < n; ++i) {
      output_in += w[L - 1][i][0] * a[L - 1][i];
    }
    output_a = g(output_in);
  }
  
  private void PropagateBackwards(int y) {
    // set output, and do last layer of network
    output_delta = gprime(output_in) * (y - output_a);
    for (int i = 0; i < n; ++i) {
      delta[L - 1][i] = gprime(in[L - 1][i]) * w[L - 1][i][0] * output_delta;
    }
    // for each layer
    for (int l = L - 2; l >= 0; --l) {
      // for each node in the lyaer
      for (int j = 0; j < n; ++j) {
        delta[l][j] = 0.0;
        for (int i = 0; i < n; ++i) {
          delta[l][j] += w[l][j][i] * delta[l + 1][i];
        }
        delta[l][j] *= gprime(in[l][j]);
      }
    }
  }
  
  private void UpdateWeights() {
    for (int l = 0; l < L - 1; ++l) {
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          w[l][i][j] = w[l][i][j] + alpha * a[l][i] * delta[l + 1][j];
        }
      }
    }
    
    for (int i = 0; i < n; ++i) {
      w[L - 1][i][0] = w[L - 1][i][0] + alpha * a[L - 1][i] * output_delta;
    }
  }
  
  
  // create a neural network
  private void BackPropLearning(Vector<DataPoint> examples) {
    int nexamples = examples.size();
    
    for (int e = 0; e < epochs; ++e) {
      // randomize the examples
      Collections.shuffle(examples);
      
      // go through every example
      for (int idp = 0; idp < nexamples; ++idp) {
        // get the input attributes
        int[] x = examples.get(idp).Attributes();
        int y = examples.get(idp).Label();
        PropagateForwards(x, y);
        PropagateBackwards(y);
        UpdateWeights();
      }
    }
  }
  
  // construct for NeuralNetwork
  public NeuralNetwork(DataSet train, boolean print_verbose, int L, int epochs) {
    this.print_verbose = print_verbose;
    this.epochs = epochs;
    this.train = train;
    this.L = L;
    this.alpha = 0.005;
    
    // create arrays for internal use
    this.n = train.NBinaryAttributes();
    this.w = new double[L][n][n];
    this.a = new double[L][n];
    this.in = new double[L][n];
    this.delta = new double[L][n];
    
    // create variables for output node
    this.output_in = 0.0;
    this.output_a = 0.0;
    this.output_delta = 0.0;
    
    // create a vector of data points
    Vector<DataPoint> examples = new Vector<DataPoint>();
    for (int k = 0; k < train.NDataPoints(); ++k) {
      examples.addElement(new DataPoint(train.KthBinaryDataPoint(k)));
    }

    // train on the DataSet
    BackPropLearning(examples);
  }
  
  // constructor for NeuralNetwork
  public NeuralNetwork(DataSet train, boolean print_verbose) { 
    this(train, print_verbose, 1, 50000);
  }
  
  // classify a particular data
  public int[] Classify(DataSet test) {
    int[] labels = new int[test.NDataPoints()];
    for (int idp = 0; idp < test.NDataPoints(); ++idp) {
      int[] x = test.KthBinaryDataPoint(idp).Attributes();
      int y = test.KthBinaryDataPoint(idp).Label();
      /* do the same propagate forward process */
      PropagateForwards(x, y);
      if (output_a > 0.5) labels[idp] = 1;
      else labels[idp] = 0;
    }
    
    return labels;
  }
}