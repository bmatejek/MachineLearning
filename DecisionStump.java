public class DecisionStump implements Learner {
  private final int print_verbose;
  
  // constructor for DecisionStump
  public DecisionStump(DataSet train, int print_verbose) { 
    this.print_verbose = print_verbose;
    
    // Set up local copy of training data and labels
    this.X = new int[train.NDataPoints()][train.KthDataPoint(0).NAttributes()];
    this.labels = new int[X.length];
    for (int i = 0; i < X.length; i++) {
      DataPoint temp = train.KthDataPoint(i);
      labels[i] = temp.Label();
      for (int j = 0; j < X[0].length; j++) {
        X[i][j] = temp.KthAttribute(j);
      }
    }

    // train on the DataSet
  }
  // classify a particular data
  public int[] Classify(DataSet test) {
    return new int[0];
  }
}