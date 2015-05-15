package MachineLearning;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.*;
public class Connect4 {

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
	private static double training_proportion = 0.01;
	private static double noise = 0.0;


	// array of training/testing file names
	private static String path_location = "./MachineLearning/data/connect4/";
	private static String fileName = "connect-4";
	



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
			BufferedReader brData = new BufferedReader(
										new FileReader(path_location + fileName + ".data"));

			ArrayList<DataPoint> dataSet = new ArrayList<DataPoint>();
			while ((line = brData.readLine()) != null) {
				String[] elmts = line.split(",");
				int label = labelSymbols.indexOf(elmts[elmts.length - 1]);
				if (Math.random() < noise)
					label = 1 - label;
				int[] attribute = new int[mapping.length];
				for (int i = 0; i < elmts.length - 1; i++) {
					attribute[i] = symbols.indexOf(elmts[i]);
				}
				DataPoint dp = new DataPoint(attribute, label, false);
				dataSet.add(dp);
			}
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

    	if (ada_boost) {
    		AdaBoost boost = new AdaBoost(training_dataset, prnt);
    		int[] output = boost.Classify(training_dataset);

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
    		System.out.println("AdaBoost Error Rate: " + ((double) wrong / (wrong + right)));
    	}
    	if (decision_stump) {
    		double[] w = new double[training_dataset.NDataPoints()];
	    	double val = 1.0 / w.length;
	    	for (int i = 0; i < w.length; i++) {
	    		w[i] = val;
	    	}
	    	DecisionStump stump = new DecisionStump(training_dataset, w, prnt);
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
	    	System.out.println("Decision Stump Error Rate: " + ((double) wrong / (wrong + right)));
	    }
	    if (decision_tree) {}
	    if (naive_bayes) {}
	    if (neural_network) {}
	    if (random_forest) {}
	    if (svm) {
	    	SVM svm = new SVM(training_dataset, prnt);
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
	    	System.out.println("SVM Error Rate: " + ((double) wrong / (wrong + right)));
	    }
	    if (weighted_majority) {}

	}

}

