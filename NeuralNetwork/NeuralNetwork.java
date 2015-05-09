public class NeuralNetwork implements Learner {
  private final int print_verbose;
  
  // constructor for NeuralNetwork
  public NeuralNetwork(DataSet train, int print_verbose) { 
    this.print_verbose = print_verbose;
    
    // train on the DataSet
  }
  
  // classify a particular data
  public int[] Classify(DataSet test) {
    return new int[0];
  }
}