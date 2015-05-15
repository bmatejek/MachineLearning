package MachineLearning;

public interface Learner { 
  // classify a particular data
  public int[] Classify(DataSet test);
}