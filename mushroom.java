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
  private static double training_proportion = 0.0;
  private static double noise = 0.0;
  
  // filename where all of the data is
  private static String filename = "data/mushroom/agaricus-lepiota";
  
  // array lists used for the data set object
  private static ArrayList<String> labels       = new ArrayList<String>();
  private static ArrayList<String> labelSymbols = new ArrayList<String>();
  private static ArrayList<String> categories   = new ArrayList<String>();
  private static ArrayList<ArrayList<String>> attributes = new ArrayList<ArrayList<String>>();
  private static ArrayList<ArrayList<String>> symbols    = new ArrayList<ArrayList<String>>();  
  
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
          try { ++argv; training_proportion = Double.parseDouble(args[argv]); }
          catch(Exception e) { System.err.printf("Error: Unrecognized argument %s.\n", args[argv]); return false; }
        }
        else { System.err.printf("Error: Unrecognized argument %s.\n", args[argv]); return false; }
      }
      else {
        System.err.printf("Error: Unrecognized argument %s.\n", args[argv]); return false;
      }
    }

    // print out the level of noise and training proportion
    if (print_verbose) {
      System.out.printf("Applying a factor of %.2f noise.\n", noise);
      System.out.printf("Using %.2f of data for training.\n", training_proportion);
    }
    
    // return success
    return true;
  }
  
  
  public static void main(String args[]) {
    // parse the input arguments
    if (!parseArgs(args)) return;
    
    

    // Read in strings for dataset
    try {
      BufferedReader brNames = new BufferedReader(new FileReader(filename + ".names"));
      
      String line;
      line = brNames.readLine();
      while (!line.contains("Attribute Information"))
        line = brNames.readLine();
      
      Pattern labelPattern = Pattern.compile("\\p{Blank}(\\p{Alnum}*?)=(.)");
      Matcher labelMatcher = labelPattern.matcher(line);
      while (labelMatcher.find()) {
        labels.add(labelMatcher.group(1));
        labelSymbols.add(labelMatcher.group(2));
      }
      line = brNames.readLine();
      
      Pattern categoryPattern  = Pattern.compile("\\p{Digit}.\\p{Blank}((\\p{Alnum}|-)*?\\p{Punct}?):");
      Pattern attributePattern = Pattern.compile("(\\p{Blank}|,)(\\p{Alnum}*?)=");
      Pattern symbolPattern    = Pattern.compile("=(.)");
      while (!line.contains("Missing Attribute Values:")) {
        Matcher categoryMatcher  = categoryPattern.matcher(line);
        Matcher attributeMatcher = attributePattern.matcher(line);
        Matcher symbolMatcher    = symbolPattern.matcher(line);
        
        // Add next category to category list
        while(categoryMatcher.find())
          categories.add(categoryMatcher.group(1));
        
        // Add attributes to attribute list
        ArrayList<String> temp = new ArrayList<String>();
        while(attributeMatcher.find())
          temp.add(attributeMatcher.group(2));
        
        attributes.add(temp);
        
        // Add symbols to symbol array
        temp = new ArrayList<String>();
        while(symbolMatcher.find()) {
          temp.add(symbolMatcher.group(1));
        }
        symbols.add(temp);
        
        line = brNames.readLine();
      }
      
      /* PRINT OUT STRINGS
       System.out.println("LABELS:");
       for (int i = 0; i < labels.size(); i++) {
       System.out.println("  " + labels.get(i));
       }
       System.out.println("\nCategories and Attributes:");
       for (int i = 0; i < categories.size(); i++) {
       System.out.println(categories.get(i));
       String[] temp = attributes.get(i);
       String[] symb = symbols.get(i);
       for (int j = 0; j < temp.length; j++) {
       System.out.println("  " + temp[j] + "  " + symb[j]);
       }
       }
       */
      
      
    }
    
    catch (IOException e) {
      System.out.println("LALALALA FILL THIS IN");
    }
    
    
    // Read in actual data and put in base10 and bin forms. 
    try {
      BufferedReader brData = new BufferedReader(new FileReader(filename + ".data"));
      
      ArrayList<DataPoint> dataset = new ArrayList<DataPoint>();
      String line;
      while ((line = brData.readLine()) != null) {
        String[] elmts = line.split(",");
        int label = labelSymbols.indexOf(elmts[0]);
        int[] data_attributes = new int[elmts.length - 1];
        for (int i = 1; i < elmts.length; i++)
          data_attributes[i-1] = symbols.get(i-1).indexOf(elmts[i]);
        
        // create a new DataPoint
        DataPoint data_point = new DataPoint(data_attributes, label, false);
        dataset.add(data_point);
        
        
        // HERE'S WHAT I EXPECTED TO DO, BUT TYPES DON'T TOTALLY AGREE YET FOR
        // WHATEVER DATAPOINT IS EXPECTING TO RECEIVE. 
        //DataPoint dp = new DataPoint(attribute, label);
        //dataSet.add(ds);
        /*
         Hi Brian,
         What I have now, is the following:
         
         Name: labels
         An ArrayList of the strings of each possible label. 
         
         Name: categories
         An ArrayList of strings giving the names of each category of each attribute,
         i.e., for mushrooms, that's "cap-shape", "cap-surface", etc. 
         
         Name: attributes
         An ArrayList of ArrayLists with possibilities for each category. 
         Element ij gives the string name of the jth option 
         for the ith characteristic. (I.e., if it's element {2,3} for
         mushrooms, that'd be gray, per item 7 in file agaricus-lepiota.names
         that gives the labels/attributes/etc for the mushrooms. Note - used zero
         indexing.)
         
         Name: dataSet
         An ArrayList of DataPoints. This hasn't actually been made, it's commented
         out above. I was expecting to pass it a single integer, indicating the 
         index of its label in the label ArrayList, and then an integer array, the
         length of the number of categories, where each element gives the particular
         attribute. I.e, we can get the label for the attribute by accessing the attribute
         arraylist of arraylists as follows: attributes.get(i).get(attribute[i]). 
         Note, "attributes" is the arraylist of arraylists holding strings to describe
         each possible attribute, and "attribute" is the int array that represents the 
         attributes of a particular data point. 
         
         I have some other info, but I expect this is what's relevant. 
         */
      }
    }
    
    catch (IOException e) {
      System.out.println("LALALALA FILL ME IN.");
    }
    
  }
  
}