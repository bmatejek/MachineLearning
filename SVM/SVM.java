public class SVM implements Learner {
  private final int print_verbose;
  
  // constructor for SVM
  public SVM(DataSet train, int print_verbose) { 
    this.print_verbose = print_verbose;
    
    // train on the DataSet
  }
  
  // classify a particular data
  public int[] Classify(DataSet test) {
    return new int[0];
  }
}