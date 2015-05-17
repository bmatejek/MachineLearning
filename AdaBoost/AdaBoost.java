package MachineLearning.AdaBoost;

import java.util.ArrayList;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import MachineLearning.*;
import MachineLearning.DecisionStump.*;

public class AdaBoost implements Learner {
  private final boolean print_verbose;
  private double eps = 1.0e-4; // Accuracy of error rate for convergence
  private int K; // Number of hypotheses
  private static final int K_DEFAULT_VALUE = 1000;
  private int[][] X; // Training examples
  private int[] labels; // Training example labels
  private boolean[] missed; // Misclassified examples on this round
  private double[] w; // Training weights
  private ArrayList<DecisionStump> l; // Array of Weak learners
  private ArrayList<Double> al; // Classifier weights
  private double err; // Error rate of full ensemble
  private DataSet training; // Pointer to the training dataset. 
  private final int learning_time = 600000; // (time in milliseconds)


  // constructors for AdaBoost, boosting solely on Decision Stumps
  public AdaBoost(DataSet train, DataSet test, int k) {
    this(train, false, k, test);
  }

  public AdaBoost(DataSet train, boolean print_verbose) {
    this(train, print_verbose, K_DEFAULT_VALUE, null);
  }

  public AdaBoost(DataSet train, boolean print_verbose, int learners, DataSet test) { 
    this.K = learners;
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
    Timer timer = new Timer(learning_time);
    for (int i = 0; i < this.K && this.err > this.eps && timer.getTimeRemaining() >= 0; i++) {
      DecisionStump DS = new DecisionStump(train, this.w, print_verbose);
      double DS_err = this.getErr(DS);
      if (print_verbose && DS_err > 0.5) {
        System.out.println("stump error too high: " + DS_err);
        break;
      }
      l.add(DS);
      al.add(getAlpha(DS_err));
      this.updateWeights(DS_err);
      this.updateErr();
      if (test != null) {
        int[] output = this.Classify(test);
        int right = 0;
        int wrong = 0;
        for (int j = 0; j < output.length; j++) {
          if (output[j] == test.KthBinaryDataPoint(j).Label()) {
            right++;
          }
          else {
            wrong++;
          }
        }
        if (l.size() % 50 == 0) {
          System.out.printf("%15f   ", ((double) wrong / (wrong + right)));
        }
      }
      
    }
     System.out.printf("Final training error:  %15f\n", this.err);
    
  }

  public int NLearners() {
    return this.l.size();
  }
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
       // missed[i] = true;
      }
      // else {
      //   missed[i] = false;
      // }
    }
    if (this.print_verbose && errorWeight > this.err){
      System.out.printf("Increase in full hypothesis error rate from %f to %f\n", this.err, errorWeight);
    }
    this.err = errorWeight;
  }
  

  private double getAlpha(double error) {
    return 0.5 * Math.log((1 - error)/error);
  }

  // private double getNorm() {
  //   return 2.0 * Math.sqrt(this.err * (1.0 - this.err));
  // }

  private void updateWeights(double error) {
    //double sum = 0.0;
    for (int i = 0; i < this.w.length; i++) {
      if (this.missed[i]) {
        w[i] = 0.5 * w[i] / error;
        //w[i] *= Math.exp(al.get(al.size()-1)); 
        //sum += w[i];
      }
      else {
        w[i] = 0.5 * w[i] / (1.0 - error);
        //w[i] *= Math.exp(-1.0 * al.get(al.size()-1));
        //sum += w[i];
      }
    }
    // for (int i = 0; i < this.w.length; i++) {
    //   w[i] /= sum;
    // }

    if (print_verbose) {
      double total = 0.0;
      for (int i = 0; i < this.w.length; i++) {
        total += w[i];
      }
      if (print_verbose && Math.abs(total - 1.0) > 1.0e-10) {
        System.out.println("Weights sum to " + total);
      }
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

  // Borrowed from assignment 2, COS402, Fall 2014, timer class
  private class Timer {
    public Timer(long max_time) {
      tb = ManagementFactory.getThreadMXBean();
        cpu_time = tb.isThreadCpuTimeSupported();
        start_time = get_time();
        end_time = start_time + max_time;
    }
    public long getTimeElapsed() {
        return (get_time() - start_time);
    }

    public long getTimeRemaining() {
        return (end_time - get_time());
    }

    private long start_time, end_time;
    private static final long milli_to_nano = 1000000;
    private ThreadMXBean tb;
    private boolean cpu_time;

    private long get_time() {
        if (cpu_time)
            return tb.getCurrentThreadCpuTime() / milli_to_nano;
        else
            return System.currentTimeMillis();
    }
  }

}








