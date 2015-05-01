public interface Learner { 
  // train on a dataset
  public void Train(DataSet train);
  
  // classify a particular data
  public void Classify(DataPoint test);
}