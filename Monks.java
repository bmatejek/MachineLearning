package MachineLearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Vector;
import MachineLearning.*;
import MachineLearning.AdaBoost.*;
import MachineLearning.DecisionStump.*;
import MachineLearning.DecisionTree.*;
import MachineLearning.NaiveBayes.*;
import MachineLearning.NeuralNetwork.*;
import MachineLearning.RandomForest.*;
import MachineLearning.SVM.*;
import MachineLearning.TreeBagging.*;
import MachineLearning.WeightedMajority.*;

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
  private static boolean tree_bagging = false;
  private static boolean weighted_majority = false;
  private static double training_proportion = 0.0;
  private static double noise = 0.0;
  
  // array of training/testing file names
  private static String path_location = "MachineLearning/data/monks/";
  private static String[] training_filenames = {"monks-1.train", "monks-2.train", "monks-3.train"};
  private static String[] testing_filenames = {"monks-1.test", "monks-2.test", "monks-3.test"};
  
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
        else if (args[argv].equals("-TreeBagging")) tree_bagging = true;
        else if (args[argv].equals("-WeightedMajority")) weighted_majority = true;
        else if (args[argv].equals("-All")) {
          ada_boost = true;
          decision_stump = true;
          decision_tree = true;
          naive_bayes = true;
          neural_network = true;
          random_forest = true;
          svm = true;
          tree_bagging = true;
          weighted_majority = true;
        }
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
      try {
        BufferedReader training_br = new BufferedReader(new FileReader(path_location + training_filenames[i]));
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
        System.err.printf("Error: failed to read %s.\n", training_filenames[i]);
        return;
      }
      // create this training data set
      training_datasets[i] = new DataSet(mapping, label_mapping, data_points.toArray(new DataPoint[data_points.size()]));
    }
    
    // read in the testing data sets
    for (int i = 0; i < testing_filenames.length; ++i) {
      Vector<DataPoint> data_points  = new Vector<DataPoint>();
      try {
        BufferedReader testing_br = new BufferedReader(new FileReader(path_location + testing_filenames[i]));
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
        System.err.printf("Error: failed to read %s.\n", testing_filenames[i]);
        return;
      }
      // create this training data set
      testing_datasets[i] = new DataSet(mapping, label_mapping, data_points.toArray(new DataPoint[data_points.size()]));
    }
    
    for (int k = 0; k < testing_datasets.length; ++k) {
      // run all of the instances specified by the user
      if (ada_boost) {
        System.out.printf("Ada Boost:\n");
        AdaBoost ab = new AdaBoost(training_datasets[k], print_verbose); 
        int[] test = ab.Classify(testing_datasets[k]);
        int[] ncorrect = new int[testing_datasets[k].NLabels()];
        int[] noccurences  = new int[testing_datasets[k].NLabels()];
        for (int i = 0; i < testing_datasets[k].NDataPoints(); ++i) {
          int label = testing_datasets[k].KthDataPoint(i).Label();
          if (test[i] == label) {
            ncorrect[label]++;
          }
          noccurences[label]++;
        }
        for (int i = 0; i < ncorrect.length; ++i) {
          System.out.println(ncorrect[i] + "/" + noccurences[i]);
        }
      }
      if (decision_stump) {
        System.out.printf("Decision Stump:\n");
        double[] w = new double[training_datasets[k].NDataPoints()];
        for (int i = 0; i < w.length; ++i)
          w[i] = 1.0;
        DecisionStump ds = new DecisionStump(training_datasets[k], w, print_verbose); 
        int[] test = ds.Classify(testing_datasets[k]);
        int[] ncorrect = new int[testing_datasets[k].NLabels()];
        int[] noccurences  = new int[testing_datasets[k].NLabels()];
        for (int i = 0; i < testing_datasets[k].NDataPoints(); ++i) {
          int label = testing_datasets[k].KthDataPoint(i).Label();
          if (test[i] == label) {
            ncorrect[label]++;
          }
          noccurences[label]++;
        }
        for (int i = 0; i < ncorrect.length; ++i) {
          System.out.println(ncorrect[i] + "/" + noccurences[i]);
        }
      }
      if (decision_tree) { 
        System.out.printf("Decision Tree:\n");
        DecisionTree dt = new DecisionTree(training_datasets[k], print_verbose); 
        int[] test = dt.Classify(testing_datasets[k]);
        int[] ncorrect = new int[testing_datasets[k].NLabels()];
        int[] noccurences  = new int[testing_datasets[k].NLabels()];
        for (int i = 0; i < testing_datasets[k].NDataPoints(); ++i) {
          int label = testing_datasets[k].KthDataPoint(i).Label();
          if (test[i] == label) {
            ncorrect[label]++;
          }
          noccurences[label]++;
        }
        for (int i = 0; i < ncorrect.length; ++i) {
          System.out.println(ncorrect[i] + "/" + noccurences[i]);
        }
      }
      if (naive_bayes) {
        System.out.printf("Naive Bayes:\n");
        NaiveBayes nb = new NaiveBayes(training_datasets[k], print_verbose);
        int[] test = nb.Classify(testing_datasets[k]);
        int[] ncorrect = new int[testing_datasets[k].NLabels()];
        int[] noccurences = new int[testing_datasets[k].NLabels()];
        for (int i = 0; i < testing_datasets[k].NDataPoints(); ++i) {
          int label = testing_datasets[k].KthDataPoint(i).Label();
          if (test[i] == label) {
            ncorrect[label]++;
          }
          noccurences[label]++;
        }
        for (int i = 0; i < ncorrect.length; ++i) {
          System.out.println(ncorrect[i] + "/" + noccurences[i]);
        }
      }
      if (neural_network) {
        System.out.printf("Neural Network:\n");
        int k = 0;
        NeuralNetwork nn = new NeuralNetwork(training_datasets[k], print_verbose);
        int[] test = nn.Classify(testing_datasets[k]);
        int[] ncorrect = new int[testing_datasets[k].NLabels()];
        int[] noccurences = new int[testing_datasets[k].NLabels()];
        for (int i = 0; i < testing_datasets[k].NDataPoints(); ++i) {
          int label = testing_datasets[k].KthDataPoint(i).Label();
          if (test[i] == label) {
            ncorrect[label]++;
          }
          noccurences[label]++;
        }
        for (int i = 0; i < ncorrect.length; ++i) {
          System.out.println(ncorrect[i] + "/" + noccurences[i]);
        }
      }
      if (random_forest) {
        System.out.printf("Random Forest:\n");
        RandomForest rf = new RandomForest(training_datasets[0], print_verbose, 1000);
        int[] test = rf.Classify(testing_datasets[0]);
        int[] ncorrect = new int[testing_datasets[0].NLabels()];
        int[] noccurences = new int[testing_datasets[0].NLabels()];
        for (int i = 0; i < testing_datasets[0].NDataPoints(); ++i) {
          int label = testing_datasets[0].KthDataPoint(i).Label();
          if (test[i] == label) {
            ncorrect[label]++;
          }
          noccurences[label]++;
        }
        for (int i = 0; i < ncorrect.length; ++i) {
          System.out.println(ncorrect[i] + "/" + noccurences[i]);
        }
      }
      if (svm) {
        System.out.printf("SVM:\n");
        SVM svm = new SVM(training_datasets[k], print_verbose);
        int[] test = svm.Classify(testing_datasets[k]);
        int[] ncorrect = new int[testing_datasets[k].NLabels()];
        int[] noccurences = new int[testing_datasets[k].NLabels()];
        for (int i = 0; i < testing_datasets[k].NDataPoints(); ++i) {
          int label = testing_datasets[k].KthDataPoint(i).Label();
          if (test[i] == label) {
            ncorrect[label]++;
          }
          noccurences[label]++;
        }
        for (int i = 0; i < ncorrect.length; ++i) {
          System.out.println(ncorrect[i] + "/" + noccurences[i]);
        }
      }
      if (tree_bagging) {
        System.out.printf("Tree Bagging:\n");
        TreeBagging tb = new TreeBagging(training_datasets[k], print_verbose, 1000);
        int[] test = tb.Classify(testing_datasets[k]);
        int[] ncorrect = new int[testing_datasets[k].NLabels()];
        int[] noccurences = new int[testing_datasets[k].NLabels()];
        for (int i = 0; i < testing_datasets[k].NDataPoints(); ++i) {
          int label = testing_datasets[k].KthDataPoint(i).Label();
          if (test[i] == label) {
            ncorrect[label]++;
          }
          noccurences[label]++;
        }
        for (int i = 0; i < ncorrect.length; ++i) {
          System.out.println(ncorrect[i] + "/" + noccurences[i]);
        }
      }
      if (weighted_majority) {
        System.out.printf("Weighted Majority:\n");
        WeightedMajority wm = new WeightedMajority(training_datasets[k], print_verbose);
        int[] test = wm.Classify(testing_datasets[k]);
        int[] ncorrect = new int[testing_datasets[k].NLabels()];
        int[] noccurences = new int[testing_datasets[k].NLabels()];
        for (int i = 0; i < testing_datasets[k].NDataPoints(); ++i) {
          int label = testing_datasets[k].KthDataPoint(i).Label();
          if (test[i] == label) {
            ncorrect[label]++;
          }
          noccurences[label]++;
        }
        for (int i = 0; i < ncorrect.length; ++i) {
          System.out.println(ncorrect[i] + "/" + noccurences[i]);
        }
      }
    }
  }
}