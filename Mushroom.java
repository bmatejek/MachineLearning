/*****************************************************************************/
/*                                                                           */
/*  Theodore O. Brundage                                                     */
/*  Brian Matejek                                                            */
/*  COS 511, Theoretical Machine Learning, Professor Elad Hazan              */
/*  Final Project, 5/17/15                                                   */
/*                                                                           */
/*  Description: This code is the control for testing the mushroom dataset.  */
/*     We read in data, and run requested learning algorithms on it.         */
/*                                                                           */
/*****************************************************************************/


import MachineLearning.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.*;

public class Mushroom {
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
 private static double training_proportion = 0.1;
 private static double noise = 0.0;


 // array of training/testing file names
 private static String path_location = "./data/mushroom/";
 private static String fileName = "agaricus-lepiota";
 



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


 public static void main(String args[]) {
  // parse the input arguments
  if (!parseArgs(args)) return;

  // create the mapping of attributes to possible values


  String line;
  String[][] mapping = null;
  String[] label_mapping = null;
  ArrayList<String> labelAL      = new ArrayList<String>();
  ArrayList<String> labelSymbols = new ArrayList<String>();
  ArrayList<ArrayList<String>> mappingAL    = new ArrayList<ArrayList<String>>();
  ArrayList<ArrayList<String>> symbols      = new ArrayList<ArrayList<String>>();
  
  // Read in strings for dataset
  try {
   BufferedReader brNames = new BufferedReader(
               new FileReader(path_location + fileName + ".names"));

   line = brNames.readLine();
   while (!line.contains("Attribute Information")) {

    line = brNames.readLine();

   }

   Pattern labelPattern = Pattern.compile("\\p{Blank}(\\p{Alnum}*?)=(.)");
   Matcher labelMatcher = labelPattern.matcher(line);
   while (labelMatcher.find()) {
    labelAL.add(labelMatcher.group(1));
    labelSymbols.add(labelMatcher.group(2));
   }
   label_mapping = labelAL.toArray(new String[labelAL.size()]);
   line = brNames.readLine();

   Pattern categoryPattern  = Pattern.compile("\\p{Digit}.\\p{Blank}((\\p{Alnum}|-)*?\\p{Punct}?):");
   Pattern attributePattern = Pattern.compile("(\\p{Blank}|,)(\\p{Alnum}*?)=");
   Pattern symbolPattern    = Pattern.compile("=(.)");
   int max = 0;
   while (!line.contains("Missing Attribute Values:")) {
    Matcher categoryMatcher  = categoryPattern.matcher(line);
    Matcher attributeMatcher = attributePattern.matcher(line);
    Matcher symbolMatcher    = symbolPattern.matcher(line);

    // Add next category to category list
    ArrayList<String> temp = new ArrayList<String>();
    if (categoryMatcher.find())
      temp.add(categoryMatcher.group(1));
    

    // Add attributes to attribute list
    while(attributeMatcher.find()) {
     temp.add(attributeMatcher.group(2));
    }
    if (temp.size() != 0)
     mappingAL.add(temp);

    // Add symbols to symbol array
    temp = new ArrayList<String>();
    while(symbolMatcher.find()) {
     temp.add(symbolMatcher.group(1));
    }
    if (temp.size() > max) max = temp.size();
    if (temp.size() != 0)
      symbols.add(temp);

    line = brNames.readLine();
   }
   // toArray method does not work for 2D arrays here.
   mapping = new String[mappingAL.size()][max];
   for (int i = 0; i < mapping.length; i++) {
    mapping[i] = mappingAL.get(i).toArray(new String[max]);
   }
   // for (int i = 0; i < mapping.length; i++) {
   //  for (int j = 0; j < mapping[0].length; j++) {
   //   System.out.print(mapping[i][j] + " ");
   //  }
   //  System.out.println();
   // }
   // for (int i = 0; i < symbols.size(); i++) {
   //  ArrayList<String> temp = symbols.get(i);
   //  for (int j = 0; j < temp.size(); j++) {
   //   System.out.print(temp.get(j) + " ");
   //  } 
   //  System.out.println();
   // }
   
   

  }

  catch (IOException e) {
   System.err.printf("Error: failed to read from %s.names\n", fileName);
  }
  

  // Read in actual data and put in base10 and bin forms. 
  DataSet training_dataset = null;
  DataSet testing_dataset  = null;
  try {
   BufferedReader brData = new BufferedReader(
          new FileReader(path_location + fileName + ".data"));

   ArrayList<DataPoint> dataSet = new ArrayList<DataPoint>();
   while ((line = brData.readLine()) != null) {
    String[] elmts = line.split(",");
    int label = labelSymbols.indexOf(elmts[0]);
    if (Math.random() < noise)
     label = 1 - label;
    int[] attribute = new int[symbols.size()];
    for (int i = 1; i < elmts.length; i++) {
     attribute[i-1] = symbols.get(i-1).indexOf(elmts[i]);
    }
    DataPoint dp = new DataPoint(attribute, label, false);
    dataSet.add(dp);
    // System.out.println("\nDatapoint " + label_mapping[label]);
    // for (int i = 0; i < attribute.length; i++) {
    //  System.out.println(mapping[i][attribute[i] + 1] + "  " + attribute[i]);
    // }
   }

   // print out the level of noise
        if (print_verbose)
            System.out.printf("Applying a factor of %.2f noise.\n", noise);

        ArrayList<DataPoint> dsTrain = new ArrayList<DataPoint>();
        ArrayList<DataPoint> dsTests = new ArrayList<DataPoint>();
        for (int i = 0; i < dataSet.size(); i++) {
         if (Math.random() < training_proportion)
          dsTrain.add(dataSet.get(i));
         else {
          dsTests.add(dataSet.get(i));
         }
        }
   // List<DataPoint> dsTrain = dataSet.subList(0, (int) (training_proportion*dataSet.size())); 
   // List<DataPoint> dsTests = dataSet.subList((int) (training_proportion*dataSet.size()), dataSet.size());
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
     if (ada_boost) {
      AdaBoost boost = new AdaBoost(training_dataset, 1);
      int[] output = boost.Classify(testing_dataset);

      int right = 0;
      int wrong = 0;
      for (int i = 0; i < output.length; i++) {
       if (output[i] == testing_dataset.KthBinaryDataPoint(i).Label()) {
        right++;
       }
       else {
        wrong++;
       }
      }
      System.out.println("Error Rate: " + ((double) wrong / (wrong + right)));
     }
     if (decision_stump) {
      double[] w = new double[training_dataset.NDataPoints()];
      double val = 1.0 / w.length;
      for (int i = 0; i < w.length; i++) {
       w[i] = val;
      }
      DecisionStump stump = new DecisionStump(training_dataset, w, 1);
      int[] output = stump.Classify(testing_dataset);

      int right = 0;
      int wrong = 0;
      for (int i = 0; i < output.length; i++) {
       if (output[i] == testing_dataset.KthBinaryDataPoint(i).Label()) {
        right++;
       }
       else {
        wrong++;
       }
      }
      System.out.println("Error Rate: " + ((double) wrong / (wrong + right)));
     }
     if (decision_tree) {}
     if (naive_bayes) {}
     if (neural_network) {}
     if (random_forest) {}
     if (svm) {
      SVM svm = new SVM(training_dataset, 1);
      int[] output = svm.Classify(testing_dataset);

      int right = 0;
      int wrong = 0;
      for (int i = 0; i < output.length; i++) {
       if (output[i] == testing_dataset.KthBinaryDataPoint(i).Label()) {
        right++;
       }
       else {
        wrong++;
       }
      }
      System.out.println("right: " + right + "  wrong: " + wrong);
      System.out.println("Error Rate: " + ((double) wrong / (wrong + right)));
     }
     if (weighted_majority) {}

 }

}

