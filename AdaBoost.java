import java.util.ArrayList;

public class AdaBoost implements Learner {
  private final int print_verbose;
  private double eps = 1.0e-10; // Accuracy of error rate for convergence
  private int K; // Number of hypotheses
  private int[][] X; // Training examples
  private int[] labels; // Training example labels
  private boolean[] missed; // Misclassified examples on this round
  private double[] w; // Training weights
  private ArrayList<DecisionStump> l; // Array of Weak learners
  private ArrayList<Double> al; // Classifier weights
  private double err; // Error rate of full ensemble
  private DataSet training; // Pointer to the training dataset. 

  // Returns error rate of a particular candidate for the current weights
  // of the test samples. Error rate is simply the sum of misclassified 
  // weights. 
  private double getErr(DecisionStump ds) {
    int[] predictions = ds.Classify(this.training);
    double errorWeight = 0.0;
    for (int i = 0; i < predictions.length; i++) {
      if (predictions[i] != labels[i]) {
        errorWeight += w[i];
        missed[i] = true;
      }
      else {
        missed[i] = false;
      }
    }
    return errorWeight;
  }

  // Updates error rate of the ensemble on the training set
  private void updateErr() {
    int[] predictions = this.Classify(this.training);
    double errorWeight = 0.0;
    for (int i = 0; i < predictions.length; i++) {
      if (predictions[i] != labels[i]) {
        errorWeight += w[i];
        missed[i] = true;
      }
      else {
        missed[i] = false;
      }
    }
    if (this.print_verbose == 1 && errorWeight > this.err){
      System.out.printf("Increase in full hypothesis error rate from %f to %f\n", this.err, errorWeight);
    }
    this.err = errorWeight;
  }

  private double getAlpha(double error) {
    return 0.5 * (Math.log(1 - error) - Math.log(error)) / Math.log(Math.E);
  }

  // private double getNorm() {
  //   return 2.0 * Math.sqrt(this.err * (1.0 - this.err));
  // }

  private void updateWeights(double error) {
    for (int i = 0; i < this.w.length; i++) {
      if (this.missed[i]) {
        w[i] = 0.5 * w[i] / error;
        // w[i] *= Math.exp(al.get(al.size()-1)); UNNORMALIZED
      }
      else {
        w[i] = 0.5 * w[i] / (1.0 - error);
        //w[i] *= Math.exp(-1.0 * al.get(al.size()-1)); UNNORMALIZED
      }
    }
    if (print_verbose == 1) {
      double sum = 0.0;
      for (int i = 0; i < this.w.length; i++) {
        sum += w[i];
      }
      if (Math.abs(sum - 1.0) > 1.0e-10) {
        System.out.println("Weights sum to " + sum);
      }
    }
  }
  
  // constructor for AdaBoost, boosting solely on Decision Stumps
  public AdaBoost(DataSet train, int print_verbose) { 
    this.print_verbose = print_verbose;
    this.training = train;
    // Set up local copy of training data and labels
    this.X = new int[train.NDataPoints()][train.KthDataPoint(0).NAttributes()];
    this.labels = new int[X.length];
    for (int i = 0; i < X.length; i++) {
      DataPoint temp = train.KthDataPoint(i);
      labels[i] = temp.Label();
      // Set each attribute for example i
      for (int j = 0; j < X[0].length; j++) {
        X[i][j] = temp.KthAttribute(j);
      }
    }

    // Initialize weights
    this.w = new double[X.length];
    double val = 1.0 / this.w.length;
    for (int i = 0; i < this.w.length; i++) {
      this.w[i] = val;
    }

    // train on the DataSet
    l  = new ArrayList<DecisionStump>();
    al = new ArrayList<Double>();
    missed = new boolean[this.labels.length];
    this.err = Double.MAX_VALUE;
    this.K   = Integer.MAX_VALUE;
    //int asdf = 1;
    for (int i = 0; i < this.K && this.err > this.eps; i++) {
      DecisionStump DS = new DecisionStump(train, w, print_verbose);
      double DS_err = this.getErr(DS);
      if (DS_err > 0.5) break;
      l.add(DS);
      al.add(getAlpha(DS_err));
      this.updateWeights(DS_err);
      this.updateErr();
      //asdf = 0;

    }
  }

  // classify a particular dataset
  public int[] Classify(DataSet test) {
    double[] sum = new double[test.NDataPoints()];
    for (int i = 0; i < this.l.size(); i++) {
      int[] results = l.get(i).Classify(test);
      for (int j = 0; j < results.length; j++) {
        if (results[j] == 0) {
          sum[j] -= this.al.get(i);
        }
        else {
          sum[j] += this.al.get(i);
        }
      }
    }
    int[] ret = new int[sum.length];
    for (int i = 0; i < ret.length; i++) {
      if (sum[i] < 0.0) {
        ret[i] = 0;
      }
      else {
        ret[i] = 1;
      }
    }
    return ret;
  }
}








