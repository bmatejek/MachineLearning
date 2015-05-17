package MachineLearning;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.*;

import MachineLearning.RandomForest.*;
import MachineLearning.TreeBagging.*;

public class RFTBConnect4Test {
  
  // command line arguments
  private static boolean print_verbose = false;
  private static boolean random_forest = true;
  private static boolean tree_bagging = true;
  private static double training_proportion = 0.1;

  // array of training/testing file names
  private static String path_location = "./MachineLearning/data/connect4/";
  private static String fileName = "connect-4";

  public static void main(String args[]) {
    int forest_size = Integer.parseInt(args[0]);
    // create the mapping of attributes to possible values
    String line;
    
    // DataSet labels very deterministic - no need to read in labels
    String[] rows = {"a", "b", "c", "d", "e", "f", "g"};
    int columns = 6;
    String[] descriptions = {"player x has taken", "player o has taken", "blank"};
    String[] label_mapping = {"win", "loss", "draw"};
    ArrayList<String> labelSymbols = new ArrayList<String>();
    labelSymbols.add("win");
    labelSymbols.add("loss");
    labelSymbols.add("draw");
    ArrayList<String> symbols = new ArrayList<String>();
    symbols.add("x");
    symbols.add("o");
    symbols.add("b");
    
    String[][] mapping = new String[rows.length * columns][descriptions.length + 1];
    for (int i = 0; i < rows.length; i++) {
      for (int j = 1; j <= columns; j++) {
        int row = i*columns + j - 1;
        mapping[row][0] = rows[i] + j;
        mapping[row][1] = descriptions[0];
        mapping[row][2] = descriptions[1];
        mapping[row][3] = descriptions[2];
      }
    }
    
    // Read in data:
    DataSet training_dataset = null;
    DataSet testing_dataset  = null;
    try {
      BufferedReader brData = new BufferedReader(new FileReader(path_location + fileName + ".data"));
      
      ArrayList<DataPoint> dataSet = new ArrayList<DataPoint>();
      while ((line = brData.readLine()) != null) {
        String[] elmts = line.split(",");
        int label = labelSymbols.indexOf(elmts[elmts.length - 1]);
        int[] attribute = new int[mapping.length];
        for (int i = 0; i < elmts.length - 1; i++) {
          attribute[i] = symbols.indexOf(elmts[i]);
        }
        DataPoint dp = new DataPoint(attribute, label, false);
        dataSet.add(dp);
      }
      
      ArrayList<DataPoint> dsTrain = new ArrayList<DataPoint>();
      ArrayList<DataPoint> dsTests = new ArrayList<DataPoint>();
      for (int i = 0; i < dataSet.size(); i++) {
        if (Math.random() < training_proportion)
          dsTrain.add(dataSet.get(i));
        else {
          dsTests.add(dataSet.get(i));
        }
      }
      DataPoint[] data_points_train = dsTrain.toArray(new DataPoint[dsTrain.size()]);
      DataPoint[] data_points_tests = dsTests.toArray(new DataPoint[dsTests.size()]);
      training_dataset = new DataSet(mapping, label_mapping, data_points_train);
      testing_dataset  = new DataSet(mapping, label_mapping, data_points_tests);
      if (print_verbose){
        System.out.println("training_dataset size: " + training_dataset.NDataPoints());
        System.out.println("testing_dataset size:  " + testing_dataset.NDataPoints());
      }
    }
    
    catch (IOException e) {
      System.err.printf("Error: Failed to read from %s.data\n", fileName);
    }
    
    // run all of the instances specified by the user
    int prnt = 0;
    if (print_verbose) prnt = 1;
    System.out.println("training_dataset size: " + training_dataset.NDataPoints());
    System.out.println("testing_dataset size:  " + testing_dataset.NDataPoints());
    
    if (random_forest) {
      System.out.printf("Random Forest:\n");
      RandomForest rf = new RandomForest(training_dataset, print_verbose, forest_size);
      int[] test = rf.Classify(testing_dataset);
      int[] ncorrect = new int[testing_dataset.NLabels()];
      int[] noccurences = new int[testing_dataset.NLabels()];
      for (int i = 0; i < testing_dataset.NDataPoints(); ++i) {
        int label = testing_dataset.KthDataPoint(i).Label();
        if (test[i] == label) {
          ncorrect[label]++;
        }
        noccurences[label]++;
      }
      for (int i = 0; i < ncorrect.length; ++i) {
        System.out.println(ncorrect[i] + "/" + noccurences[i]);
      }
      int total_correct = 0;
      int total_instances = 0;
      for (int i = 0; i < ncorrect.length; ++i) {
        total_correct += ncorrect[i];
        total_instances += noccurences[i];
      }
      System.out.println(((double) total_correct) / (total_instances));
    }
    if (tree_bagging) {
      System.out.printf("Tree Bagging:\n");
      TreeBagging tb = new TreeBagging(training_dataset, print_verbose, forest_size);
      int[] test = tb.Classify(testing_dataset);
      int[] ncorrect = new int[testing_dataset.NLabels()];
      int[] noccurences = new int[testing_dataset.NLabels()];
      for (int i = 0; i < testing_dataset.NDataPoints(); ++i) {
        int label = testing_dataset.KthDataPoint(i).Label();
        if (test[i] == label) {
          ncorrect[label]++;
        }
        noccurences[label]++;
      }
      for (int i = 0; i < ncorrect.length; ++i) {
        System.out.println(ncorrect[i] + "/" + noccurences[i]);
      }
      int total_correct = 0;
      int total_instances = 0;
      for (int i = 0; i < ncorrect.length; ++i) {
        total_correct += ncorrect[i];
        total_instances += noccurences[i];
      }
      System.out.println(((double) total_correct) / (total_instances));
    }
  }
}

