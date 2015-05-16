package MachineLearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Vector;
import java.util.List;
import MachineLearning.*;
import MachineLearning.RandomForest.*;
import MachineLearning.TreeBagging.*;

public class TestRandomForest {
  // command line arguments
  private static boolean print_verbose = false;
  
  // array of training/testing file names
  private static String path_location = "MachineLearning/data/monks/";
  private static String output_path_location = "output/forest_";
  private static String[] filenames = {
    "new-monks-1", 
    "new-monks-2", 
    "new-monks-3",
    "new-monks-4",
    "new-monks-5",
    "new-monks-6",
    "new-monks-7",
    "new-monks-8"
  };
  
  // array of attribute names and corresponding values
  private static String[] attributes = { "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12" };
  private static int[][] attribute_values = { {1, 2, 3}, {1, 2, 3}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, {1, 2}, {1, 2, 3, 4, 5}, {1, 2, 3}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3}, {1, 2} };
  private static String[] label_mapping = {"0", "1"};
  
  // create arrays of training and testing data sets
  private static DataSet[] training_datasets = new DataSet[filenames.length];
  private static DataSet[] testing_datasets = new DataSet[filenames.length];
  
  public static void main(String[] args) {
    int forest_size = Integer.parseInt(args[0]);
    
    BufferedWriter bw = null;
    FileWriter fstream = null;
    try {
      String file = output_path_location + Integer.toString(forest_size) + ".txt";
      fstream = new FileWriter(file, false);
      bw = new BufferedWriter(fstream);
    } catch(Exception e) {
      System.out.println("Shoot!");
      System.exit(0);
    }
    
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
    for (int i = 0; i < filenames.length; ++i) {
      Vector<DataPoint> data_points  = new Vector<DataPoint>();
      try {
        BufferedReader br = new BufferedReader(new FileReader(path_location + filenames[i]));
        String line;
        while ((line = br.readLine()) != null) {
          String[] parse_attributes = line.split(" ");
          int a1 = Integer.parseInt(parse_attributes[0]);
          int a2 = Integer.parseInt(parse_attributes[1]);
          int a3 = Integer.parseInt(parse_attributes[2]);
          int a4 = Integer.parseInt(parse_attributes[3]);
          int a5 = Integer.parseInt(parse_attributes[4]);
          int a6 = Integer.parseInt(parse_attributes[5]);
          int a7 = Integer.parseInt(parse_attributes[6]);
          int a8 = Integer.parseInt(parse_attributes[7]);
          int a9 = Integer.parseInt(parse_attributes[8]);
          int a10 = Integer.parseInt(parse_attributes[9]);
          int a11 = Integer.parseInt(parse_attributes[10]);
          int a12 = Integer.parseInt(parse_attributes[11]);
          int label = Integer.parseInt(parse_attributes[12]);
          
          // create an array of attributes for this data point
          int[] data_attributes = {a1 - 1, a2 - 1, a3 - 1, a4 - 1, a5 - 1, a6 - 1, a7 - 1, a8 - 1, a9 - 1, a10 - 1, a11 - 1, a12 - 1};

          // create a new data point and add it to the training dataset
          DataPoint data_point = new DataPoint(data_attributes, label, false);
          data_points.addElement(data_point);     
        }
      }
      catch (Exception e) {
        System.err.printf("Error: failed to read %s.\n", filenames[i]);
        return;
      }
      
      int training_size = (int) (data_points.size() * 0.01);
      int testing_size = data_points.size() - training_size;
      
      List<DataPoint> training_data_points = data_points.subList(0, training_size);
      List<DataPoint> testing_data_points = data_points.subList(training_size, testing_size + training_size);

      // create this training data set
      training_datasets[i] = new DataSet(mapping, label_mapping, training_data_points.toArray(new DataPoint[training_data_points.size()]));
      testing_datasets[i] = new DataSet(mapping, label_mapping, testing_data_points.toArray(new DataPoint[testing_data_points.size()]));
    }
    
    for (int k = 0; k < testing_datasets.length; ++k) {
      try {
        bw.write("Random Forest:\n");
      }
      catch (Exception e) {
        System.out.println("Shoot!");
        System.exit(0);
      }
      RandomForest rf = new RandomForest(training_datasets[k], print_verbose, forest_size);
      int[] test = rf.Classify(testing_datasets[k]);
      int[] ncorrect = new int[testing_datasets[k].NLabels()];
      int[] noccurrences = new int[testing_datasets[k].NLabels()];
      for (int i = 0; i < testing_datasets[k].NDataPoints(); ++i) {
        int label = testing_datasets[k].KthDataPoint(i).Label();
        if (test[i] == label) {
          ncorrect[label]++;
        }
        noccurrences[label]++;
      }
      try {
        System.out.println(Integer.toString(ncorrect[0]));
        for (int i = 0; i < ncorrect.length; ++i) {
          bw.write(Integer.toString(ncorrect[i]));
          bw.write("/");
          bw.write(Integer.toString(noccurrences[i]));
          bw.write("\n");
        }
      }
      catch (Exception e) {
        System.out.println("Shoot!");
        System.exit(0);
      }
      try {
        bw.write("Tree Bagging:\n");
      }
      catch (Exception e) {
        System.out.println("Shoot!");
        System.exit(0);
      }
      TreeBagging tb = new TreeBagging(training_datasets[k], print_verbose, forest_size);
      test = tb.Classify(testing_datasets[k]);
      ncorrect = new int[testing_datasets[k].NLabels()];
      noccurrences = new int[testing_datasets[k].NLabels()];
      for (int i = 0; i < testing_datasets[k].NDataPoints(); ++i) {
        int label = testing_datasets[k].KthDataPoint(i).Label();
        if (test[i] == label) {
          ncorrect[label]++;
        }
        noccurrences[label]++;
      }
      try {
        for (int i = 0; i < ncorrect.length; ++i) {
          bw.write(Integer.toString(ncorrect[i]));
          bw.write("/");
          bw.write(Integer.toString(noccurrences[i]));
          bw.write("\n");
        }
      }
      catch (Exception e) {
        System.out.println("Shoot!");
        System.exit(0);
      }
    }
    try {
      bw.close();
    } catch (Exception e) {
      System.out.println("Shoot!");
      System.exit(0);
    }
  }
}