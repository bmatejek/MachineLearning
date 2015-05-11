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


// TEST THIS


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
		ArrayList<String> mappingAL    = new ArrayList<String>();
		ArrayList<String> symbols      = new ArrayList<String>();
		
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
			int cat = 0; //Number each category
			while (!line.contains("Missing Attribute Values:")) {
				Matcher categoryMatcher  = categoryPattern.matcher(line);
				Matcher attributeMatcher = attributePattern.matcher(line);
				Matcher symbolMatcher    = symbolPattern.matcher(line);

				// Add next category to category list
				String category = " ";
				if (categoryMatcher.find())
				  category = categoryMatcher.group(1);
				

				// Add attributes to attribute list
				while(attributeMatcher.find()) {
					String attribute = category + "-" + attributeMatcher.group(2);
					mappingAL.add(attribute);
				}

				// Add symbols to symbol array
				while(symbolMatcher.find()) 
					symbols.add(cat + symbolMatcher.group(1));
				cat++;

				line = brNames.readLine();
			}
			// toArray method does not work for 2D arrays here.
			mapping = new String[mappingAL.size()][1];
			for (int i = 0; i < mapping.length; i++) {
				mapping[i][0] = mappingAL.get(i);
			}
			

		}

		catch (IOException e) {
			System.err.printf("Error: failed to read from %s.names\n", fileName);
		}
		

		// Read in actual data and put in base10 and bin forms. 
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
					attribute[symbols.indexOf((i-1) + elmts[i])] = 1;
				}
				DataPoint dp = new DataPoint(attribute, label, false);
				dataSet.add(dp);
				// System.out.println("\nDatapoint " + label_mapping[label]);
				// for (int i = 0; i < attribute.length; i++) {
				// 	System.out.println(mapping[i][0] + "  " + attribute[i]);
				// }
			}

			// print out the level of noise
      		if (print_verbose)
      		    System.out.printf("Applying a factor of %.2f noise.\n", noise);

			List<DataPoint> dsTrain = dataSet.subList(0, (int) training_proportion*dataSet.size()); 
			List<DataPoint> dsTests = dataSet.subList((int) training_proportion*dataSet.size(), dataSet.size());
			DataPoint[] data_points_train = dsTrain.toArray(new DataPoint[dsTrain.size()]);
			DataPoint[] data_points_tests = dsTests.toArray(new DataPoint[dsTests.size()]);
			DataSet training_dataset = new DataSet(mapping, label_mapping, data_points_train);
			DataSet testing_datasets = new DataSet(mapping, label_mapping, data_points_tests);

		}

		catch (IOException e) {
			System.err.printf("Error: Failed to read from %s.data\n", fileName);
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