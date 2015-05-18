package MachineLearning.SVM;

/*****************************************************************************/
/*                                                                           */
/*  Theodore O. Brundage                                                     */
/*  Brian Matejek                                                            */
/*  COS 511, Theoretical Machine Learning, Professor Elad Hazan              */
/*  Final Project, 5/17/15                                                   */
/*                                                                           */
/*  Description: This code implements the SVM learner. SVMs require numerical*/
/*   data. Since the data used in our project are categorical, this          */
/*   implementation uses the Binary data points. Also, we expect only data   */
/*   two possible labels. Learning with more labels is expected to be handled*/
/*   by the client.                                                          */
/*                                                                           */
/*****************************************************************************/


import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;

import MachineLearning.*;


public class SVM implements Learner {

  private static boolean PRINT_VERBOSE_DEFAULT = false;
  private static double ALPHA_DEFAULT = 1.0e-4;
  private static double C_DEFAULT = 1.0;

  private final boolean print_verbose;
  private double alpha = 0.0001; // Learning rate
  private double C = 1.0; // weighting of training vs regularization
  private double eps = 1.0e-15; // Accuracy for determining convergence 
  private int convergence = 10; // number of updates less than eps wanted for convergence
  private final int learning_time = 60000; // (time in milliseconds)
  private int[][] X;  // Training set
  private int[] labels; // Training set labels
  private double[] theta; // Learned Coefficients
  private int totalSteps = 1000000; // For tests capping the number of steps


  

  // constructors for SVM
  public SVM(DataSet train, boolean print_verbose, double a, double c, DataSet test) {
    this.print_verbose = print_verbose;

    this.alpha = a;
    this.C = c;

    // Set up local copy of training data and labels
    this.X = new int[train.NDataPoints()][train.KthBinaryDataPoint(0).NAttributes()];
    this.labels = new int[X.length];
    for (int i = 0; i < X.length; i++) {
      DataPoint temp = train.KthBinaryDataPoint(i);
      labels[i] = temp.Label();
      for (int j = 0; j < X[0].length; j++) {
        X[i][j] = temp.KthAttribute(j);
      }
    }
    // Should this be initialized randomly with something nonzero?
    this.theta = new double[X[0].length]; 
    

    // train on the DataSet if we are testing during training
    if (test != null) {
      int steps = 0;
      while (steps++ < totalSteps) {
          this.gradient_step();
          if (steps % 10 == 0) {
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
          System.out.printf("%15d  %15f\n", steps, ((double) wrong / (wrong + right)));
        }
      }
    }
    else {
      Timer timer = new Timer(learning_time);
      int run = 0;
      int stp = 0;
      while (run < convergence && timer.getTimeRemaining() >= 0) {
        if (this.gradient_step() < this.eps) run++;
        else run = 0;
        stp++;
      } 
      System.out.println("Steps: " + stp);
    }
    
  }

  public SVM(DataSet train, boolean print_verbose, double a, double c) {
    this(train, print_verbose, a, c, null);
  }
  public SVM(DataSet train, boolean print_verbose) { 
    this(train, print_verbose, ALPHA_DEFAULT, C_DEFAULT, null);
  }
  public SVM(DataSet train, DataSet test) {
    this(train, PRINT_VERBOSE_DEFAULT, ALPHA_DEFAULT, C_DEFAULT, test);
  }

   
  


  // Gives the inner product of ith training example with theta
  private double theta_x_inner(int i) {
    double ret = 0.0;
    for (int j = 0; j < X[0].length; j++) {
      ret += X[i][j] * this.theta[j];
    }
    return ret;
  }
  
  private double hinge0(int i) {
    double arg = theta_x_inner(i);
    if (arg < -1.0) return 0;
    else return (1.0 + arg);
  }

  private double hinge1(int i) {
    double arg = theta_x_inner(i);
    if (arg > 1.0) return 0;
    else return (1.0 - arg);
  }

  // yields a + kb, where a and b are vectors 
  // Types specified here for these particular purposes
  private static void vector_incr(double[] a, int[] b, double k) {
    if (a.length != b.length) {
      System.err.printf("Error: cannot add vectors of different dimensions");
    }
    if (k != 0) {
      for (int i = 0; i < a.length; i++) {
       a[i] += k * b[i];
      }
    }
  }

  /*
  *  Return the cost of the current hypothesis. 
  */
  private double cost() {
    double trainingCost = 0.0;
    double regCost = 0.0;

    // Add cost due to training set
    for (int i = 0; i < X.length; i++) {
      if (labels[i] == 0) {
        trainingCost += hinge0(i);
      }
      else if (labels[i] == 1) {
        trainingCost += hinge1(i);
      }
      else {
        System.out.println("Not supposed to take in non-binary labeled data.");
      }
    }

    // Add cost due to regularization. 
    for (int i = 0; i < theta.length; i++) {
      regCost += theta[i] * theta[i];
    }

    return (this.C * trainingCost + 0.5 * regCost);
  }

  /*
  *  Return the gradient of the current hypothesis.
  */
  private double[] grad() {
    double[] ret = new double[this.theta.length];

    // Gradient due to the training errors from hinge functions
    for (int i = 0; i < X.length; i++) {
      if (labels[i] == 0) {
        if (theta_x_inner(i) < -1.0) continue;
        else vector_incr(ret, X[i], this.C);
      }
      else if (labels[i] == 1) {
        if (theta_x_inner(i) < 1.0) vector_incr(ret, X[i], -1.0*this.C);
        else continue;
      }
    }
    // Gradient due to regularization term
    for (int i = 0; i < this.theta.length; i++) {
      ret[i] += this.theta[i];
    }

    return ret;
  }

  // updates theta, and returns the absolute value of the max amount
  // by which an element changed
  public double gradient_step() {
    double[] g = this.grad();
    double del;
    double max = 0.0;
    for (int i = 0; i < this.theta.length; i++) {
      del = -1.0 * this.alpha * g[i];
      if (Math.abs(del) > max) max = Math.abs(del);
      this.theta[i] += del;
    }
    // for (int i = 0; i < this.theta.length/8; i++) {
    //   System.out.printf("%3.2f, ", theta[i]);
    // }
    // System.out.println();
    return max;
  }

  
  
  /*
  *  Hypothesis function (BINARY)
  */

  private int h(int[] x) {
    double arg = 0.0;
    for (int i = 0; i < x.length; i++) {
      arg += x[i] * this.theta[i];
    }
    if (arg >= 0.0) return 1;
    else return 0;
  }

  // classify a particular data
  public int[] Classify(DataSet test) {
    int[] ret = new int[test.NDataPoints()];
    for (int i = 0; i < ret.length; i++) {
      DataPoint dp = test.KthBinaryDataPoint(i);
      int[] x = new int[dp.NAttributes()];
      for (int j = 0; j < x.length; j++) {
        x[j] = dp.KthAttribute(j);
      }
      ret[i] = h(x);
    }
    return ret;
  }

  public static void main(String[] args) {


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






