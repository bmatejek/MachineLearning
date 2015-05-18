package MachineLearning.DecisionStump;

/*****************************************************************************/
/*                                                                           */
/*  Theodore O. Brundage                                                     */
/*  Brian Matejek                                                            */
/*  COS 511, Theoretical Machine Learning, Professor Elad Hazan              */
/*  Final Project, 5/17/15                                                   */
/*                                                                           */
/*  Description: This code implements the Decision Stumps. Generally, DSs can*/
/*   handle binary, categorical, or numeric data. However, in the interest of*/
/*   keeping this as simple and "weak" a learner as possible, we make it a   */
/*   binary DS, and only use Binary DataPoints. Also, we expect only data    */
/*   two possible labels. Learning with more labels is expected to be handled*/
/*   by the client.                                                          */
/*                                                                           */
/*****************************************************************************/

import MachineLearning.*;

public class DecisionStump implements Learner {
  private final boolean print_verbose;
  private int decision_attribute; // index of attribute used to make decision
  private int decision_direction; // value corresponding to 0 in decision_attribute
  private int[][] X; // Training Set
  private int[] labels; // Training set labels

  // constructor for DecisionStump
  // THIS IS MADE FOR BINARY DATA
  // w gives weights of each example
  public DecisionStump(DataSet train, double[] w, boolean print_verbose) { 
    this.print_verbose = print_verbose;
    assert w.length == train.NDataPoints();

    // This array holds the number of positive and negative examples that
    // correspond to having a 1 in their ith attribute. Note - positive and
    // negative corresponds to language from Russell and Norvig's textbook on
    // AI - here, they refer to labels of 1 and 0 respectively. 
    double[][] counts = new double[4][train.KthBinaryDataPoint(0).NAttributes()];
    int p0 = 0; // These are just index values
    int p1 = 1;
    int n0 = 2;
    int n1 = 3;


    // Set up local copy of training data and labels
    this.X = new int[train.NDataPoints()][train.KthBinaryDataPoint(0).NAttributes()];
    this.labels = new int[X.length];
    for (int i = 0; i < X.length; i++) {
      DataPoint temp = train.KthBinaryDataPoint(i);
      labels[i] = temp.Label();
      // Set each attribute for example i
      // While iterating, count number of 
      // positive and negative examples
      if (labels[i] == 0) {
        for (int j = 0; j < X[0].length; j++) {
          X[i][j] = temp.KthAttribute(j);
          if (X[i][j] == 0) counts[n0][j] += w[i];
          else counts[n1][j] += w[i];
        }
      }
      else {
        for (int j = 0; j < X[0].length; j++) {
          X[i][j] = temp.KthAttribute(j);
          if (X[i][j] == 0) counts[p0][j] += w[i];
          else counts[p1][j] += w[i];
        }
      }
    }

    // Brute force training 

    double max_difference = 0.0;
    // for each attribute
    for (int i = 0; i < X[0].length; i++) {
      if (counts[p0][i] + counts[n1][i] - counts[n0][i] - counts[p1][i] > max_difference) {
        max_difference = counts[p0][i] + counts[n1][i] - counts[n0][i] - counts[p1][i];
        decision_attribute = i;
        decision_direction = 0;
      }
      if (counts[p1][i] + counts[n0][i] - counts[n1][i] - counts[p0][i] > max_difference) {
        max_difference = counts[p1][i] + counts[n0][i] - counts[n1][i] - counts[p0][i];
        decision_attribute = i;
        decision_direction = 1;
      }
    }
    


    if (print_verbose) {
      System.out.println("Splitting on attribute " + this.decision_attribute + ".");
      System.out.println("Instances with value " + this.decision_direction + " are labeled 1.");
    }
  }

  // Hypothesis function for binary attribute vector x.
  private int h(int[] x) {
    if (x[decision_attribute] == decision_direction) return 1;
    else return 0;
  }

  // classify a particular data
  public int[] Classify(DataSet test) {
    int[] output = new int[test.NDataPoints()];
    for (int i = 0; i < output.length; i++) {
      DataPoint dp = test.KthBinaryDataPoint(i);
      if (dp.KthAttribute(decision_attribute) == decision_direction) {
        output[i] = 1;
      }
      else {
        output[i] = 0;
      }
    }
    return output;
  }
}