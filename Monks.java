import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Vector;

public class Monks {
  // command line arguments
  private static boolean print_verbose = false;
  private static boolean ada_boost = false;
  private static boolean decision_stump = false;
  private static boolean decision_tree = false;
  private static boolean naive_bayes = false;
  private static boolean neural_network = false;
  private static boolean random_forest = false;
  private static boolean svm = false;
  private static boolean weighted_majority = false;
  private static double training_proportion = 0.0;
  private static double noise = 0.0;
  
  // array of training/testing file names
  private static String path_location = "./data/monks/";
  private static String[] training_filenames = {"monks-1.train", "monks-2.train", "monks-3.train"};
  private static String[] tessting_filenames = {"monks-1.test", "monks-2.test", "monks-3.test"};
  
  // array of attribute names and corresponding values
  private static String[] attributes = { "a1", "a2", "a3", "a4", "a5", "a6"};
  private static int[][] attribute_values = { {1, 2, 3}, {1, 2, 3}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, {1, 2} };
  private static String[] label_mapping = {"0", "1"};
  
  // create arrays of training and testing data sets
  private static DataSet[] training_datasets = new DataSet[training_filenames.length];
  private static DataSet[] testing_datasets = new DataSet[testing_filenames.length];
  
  // function that determines the class for this data point (depends on noise)
  private static int getLabel(int a1, int a2, int a3, int a4, int a5, int a6, int dataset, double noise) {
    int label;
    // for the first dataset (a1 = a2) or (a5 = 1)
    if (dataset == 0) {
      if ((a1 == a2) || (a5 == 1)) label = 1;
      else label = 0;
    }
    // for the second dataset, exactly 2 of {a1 = 1}, {a2 = 1}, {a3 = 1}, {a4 = 1}, {a5 = 1}, {a6 = 1}
    else if (dataset == 1) {
      int count = 0;
      if (a1 == 1) count++;
      if (a2 == 1) count++;
      if (a3 == 1) count++;
      if (a4 == 1) count++;
      if (a5 == 1) count++;
      if (a6 == 1) count++;
      if (count == 2) label = 1;
      else label = 0;
    }
    else {
      // for the third dataset (a5 == 3 && a4 == 1) or (a5 != 4 && a2 != 3)
      if ((a5 == 3 && a4 == 1) || (a5 != 4 && a2 != 3)) label = 1;
      else label = 0;
    }
    
    // switch the label with probability proportional to noise
    if (Math.random() < noise)
      label = 1 - label;
    return label;
  }
  
  // argument parsing function
  private static boolean parseArgs(String[] args) {
    for (int argv = 0; argv < args.length; ++argv) {
      if (args[argv].charAt(0) == '-') {
        if (args[argv].equals("-v")) print_verbose = true;
        else if (args[argv].equals("-AdaBoost")) ada_boost = true;
        else if (args[argv].equals("-DecisionStump")) decision_stump = true;
        else if (args[argv].equals("-DecisionTree")) decision_tree = true;
        else if (args[argv].equals("-NaiveBayes")) naive_bayes = true;
        else if (args[argv].equals("-NeuralNetwork")) neural_network = true;
        else if (args[argv].equals("-RandomForest")) random_forest = true;
        else if (args[argv].equals("-SVM")) svm = true;
        else if (args[argv].equals("-WeightedMajority")) weighted_majority = true;
        else if (args[argv].equals("-noise")) { 
          try { ++argv; noise = Double.parseDouble(args[argv]); }
          catch (Exception e) { System.err.printf("Error: need a noise parameter.\n"); return false; }
        }
        else if (args[argv].equals("-train")) {
          try { System.err.printf("Training proportion ignored for this data set.\n"); }
          catch(Exception e) { System.err.printf("Error: Unrecognized argument %s.\n", args[argv]); return false; }
        }
        else { System.err.printf("Error: Unrecognized argument %s.\n", args[argv]); return false; }
      }
      else {
        System.err.printf("Error: Unrecognized argument %s.\n", args[argv]); return false;
      }
    }

    // print out the level of noise
    if (print_verbose)
      System.out.printf("Applying a factor of %.2f noise.\n", noise);
    
    // return success
    return true;
  }
  
  public static void main(String[] args) {
    // parse the input arguments
    if (!parseArgs(args)) return;
    
    // create the mapping of attributes to possible values
    int nattributes = attributes.length;
    int maxvalues = 0; // find the second dimension of the array
    for (int i = 0; i < nattributes; ++i)
      if (attribute_values[i].length > maxvalues)
        maxvalues = attribute_values[i].length;
    // create physical 2d array and fill in values
    String[][] mapping = new String[nattributes][maxvalues + 1];
    for (int i = 0; i < nattributes; ++i) {
      mapping[i][0] = attributes[i];
      for (int j = 0; j < attribute_values[i].length; ++j) {
        mapping[i][j + 1] = Integer.toString(attribute_values[i][j]);
      }
    }
    
    // read in the training data sets
    for (int i = 0; i < training_filenames.length; ++i) {
      Vector<DataPoint> data_points  = new Vector<DataPoint>();
      try (BufferedReader training_br = new BufferedReader(new FileReader(path_location + training_filenames[i]))) {
        String line;
        while ((line = training_br.readLine()) != null) {
          String[] parse_attributes = line.split(" ");
          int a1 = Integer.parseInt(parse_attributes[2]);
          int a2 = Integer.parseInt(parse_attributes[3]);
          int a3 = Integer.parseInt(parse_attributes[4]);
          int a4 = Integer.parseInt(parse_attributes[5]);
          int a5 = Integer.parseInt(parse_attributes[6]);
          int a6 = Integer.parseInt(parse_attributes[7]);
          
          // create an array of attributes for this data point
          int[] data_attributes = {a1 - 1, a2 - 1, a3 - 1, a4 - 1, a5 - 1, a6 - 1};
          
          // get the label for this class (depends on file and noise)
          int label = getLabel(a1, a2, a3, a4, a5, a6, i, noise);
          
          // create a new data point and add it to the training dataset
          DataPoint data_point = new DataPoint(data_attributes, label, false);
          data_points.addElement(data_point);     
        }
      }
      catch (Exception e) {
        System.err.printf("Error: failed to read %s\n.", training_filenames[i]);
        return;
      }
      // create this training data set
      training_datasets[i] = new DataSet(mapping, label_mapping, data_points.toArray(new DataPoint[data_points.size()]));
    }

    // read in the testing data sets
    for (int i = 0; i < testing_filenames.length; ++i) {
      Vector<DataPoint> data_points  = new Vector<DataPoint>();
      try (BufferedReader testing_br = new BufferedReader(new FileReader(path_location + testing_filenames[i]))) {
        String line;
        while ((line = testing_br.readLine()) != null) {
          String[] parse_attributes = line.split(" ");
          int a1 = Integer.parseInt(parse_attributes[2]);
          int a2 = Integer.parseInt(parse_attributes[3]);
          int a3 = Integer.parseInt(parse_attributes[4]);
          int a4 = Integer.parseInt(parse_attributes[5]);
          int a5 = Integer.parseInt(parse_attributes[6]);
          int a6 = Integer.parseInt(parse_attributes[7]);
          
          // create an array of attributes for this data point
          int[] data_attributes = {a1 - 1, a2 - 1, a3 - 1, a4 - 1, a5 - 1, a6 - 1};
          
          // get the label for this class (depends on file and noise)
          int label = getLabel(a1, a2, a3, a4, a5, a6, i, 0.0);
          
          // create a new data point and add it to the training dataset
          DataPoint data_point = new DataPoint(data_attributes, label, false);
          data_points.addElement(data_point);     
        }
      }
      catch (Exception e) {
        System.err.printf("Error: failed to read %s\n.", testing_filenames[i]);
        return;
      }
      // create this training data set
      testing_datasets[i] = new DataSet(mapping, label_mapping, data_points.toArray(new DataPoint[data_points.size()]));
    }

    // run all of the instances specified by the user
    if (ada_boost) {}
    if (decision_stump) {}
    if (decision_tree) {}
    if (naive_bayes) {}
    if (neural_network) {}
    if (random_forest) {}
    if (svm) {}
    if (weighted_majority) {}
  }
}