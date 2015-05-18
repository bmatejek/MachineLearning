package MachineLearning;


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

public class Mushroom {
	// command line arguments
	private static boolean print_verbose = false;
	private static boolean ada_boost = false;
	private static boolean ada_boost_p_k = false;
	private static boolean decision_stump = false;
	private static boolean decision_tree = false;
	private static boolean naive_bayes = false;
	private static boolean neural_network = false;
	private static boolean random_forest = false;
	private static boolean svm = false;
	private static boolean svmAlphaC = false;
	private static boolean svmCNoise = false;
	private static boolean weighted_majority = false;
	private static double training_proportion = 0.1;
	private static double noise = 0.0;


	// array of training/testing file names
	private static String path_location = "./MachineLearning/data/mushroom/";
	private static String fileName = "agaricus-lepiota";
	



	private static boolean parseArgs(String[] args) {
    for (int argv = 0; argv < args.length; ++argv) {
      if (args[argv].charAt(0) == '-') {
        if (args[argv].equals("-v")) print_verbose = true;
        else if (args[argv].equals("-AdaBoost")) ada_boost = true;
        else if (args[argv].equals("-AdaBoost-p-k")) ada_boost_p_k = true;
        else if (args[argv].equals("-DecisionStump")) decision_stump = true;
        else if (args[argv].equals("-DecisionTree")) decision_tree = true;
        else if (args[argv].equals("-NaiveBayes")) naive_bayes = true;
        else if (args[argv].equals("-NeuralNetwork")) neural_network = true;
        else if (args[argv].equals("-RandomForest")) random_forest = true;
        else if (args[argv].equals("-SVM")) svm = true;
        else if (args[argv].equals("-SVM-alpha-c")) svmAlphaC = true;
        else if (args[argv].equals("-SVM-c-noise")) svmCNoise = true;
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
		ArrayList<DataPoint> dsTrain = new ArrayList<DataPoint>();
      	ArrayList<DataPoint> dsTests = new ArrayList<DataPoint>();
		
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
			// 	for (int j = 0; j < mapping[0].length; j++) {
			// 		System.out.print(mapping[i][j] + " ");
			// 	}
			// 	System.out.println();
			// }
			// for (int i = 0; i < symbols.size(); i++) {
			// 	ArrayList<String> temp = symbols.get(i);
			// 	for (int j = 0; j < temp.size(); j++) {
			// 		System.out.print(temp.get(j) + " ");
			// 	} 
			// 	System.out.println();
			// }
			
			

		}

		catch (IOException e) {
			System.err.printf("Error: failed to read from %s.names\n", fileName);
		}
		

		// Read in actual data and put in base10 and bin forms. 
		DataSet training_dataset = null;
		DataSet testing_dataset  = null;
		ArrayList<DataPoint> dataSet = null;
		try {
			BufferedReader brData = new BufferedReader(
										new FileReader(path_location + fileName + ".data"));

			dataSet = new ArrayList<DataPoint>();
			while ((line = brData.readLine()) != null) {
				String[] elmts = line.split(",");
				int label = labelSymbols.indexOf(elmts[0]);
				// if (Math.random() < noise)
				// 	label = 1 - label;
				int[] attribute = new int[symbols.size()];
				for (int i = 1; i < elmts.length; i++) {
					attribute[i-1] = symbols.get(i-1).indexOf(elmts[i]);
				}
				DataPoint dp = new DataPoint(attribute, label, false);
				dataSet.add(dp);
				// System.out.println("\nDatapoint " + label_mapping[label]);
				// for (int i = 0; i < attribute.length; i++) {
				// 	System.out.println(mapping[i][attribute[i] + 1] + "  " + attribute[i]);
				// }
			}

			// print out the level of noise
      		if (print_verbose)
      		    System.out.printf("Applying a factor of %.2f noise.\n", noise);

      		
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
			System.out.println("binary attributes" + training_dataset.NBinaryAttributes());
		}

		catch (IOException e) {
			System.err.printf("Error: Failed to read from %s.data\n", fileName);
		}

		// run all of the instances specified by the user
    	if (ada_boost) {
    		AdaBoost boost = new AdaBoost(training_dataset, print_verbose);
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
    		System.out.println("AdaBoost Error Rate: " + ((double) wrong / (wrong + right)));
    	}
    	if (ada_boost_p_k) {
    		int runs = 10;
    		double tp_min = 0.0;
    		double tp_max = .95;
    		double tp_step = 0.05;
    		double[] tp = new double[(int) ((tp_max - tp_min)/tp_step) + 1];
    		for (int i = 0; i < tp.length; i++) {
    			if (i == 0) tp[i] = 0.01;
    			else tp[i] = tp_min + i * tp_step;
    		}

    		int K = 1000;
    		System.out.println("For Performance on Testing Sets, function of K, p = 0.1");
    		double[] results = new double[runs];
    		for (int i = 1; i <= K; i++) {
    			System.out.printf("%15d   ", i);
    		}
    		System.out.println();
    		for (int n = 0; n < runs; n++) {
				ArrayList<DataPoint> trainingPoints = new ArrayList<DataPoint>();
				ArrayList<DataPoint> testingPoints = new ArrayList<DataPoint>();
				for (int i = 0; i < dataSet.size(); i++) {
					if (Math.random() < training_proportion) {
						trainingPoints.add(dataSet.get(i));
					}
					else {
						testingPoints.add(dataSet.get(i));
					}
				}

    			AdaBoost boost = new AdaBoost(new DataSet(mapping,label_mapping,trainingPoints.toArray(new DataPoint[trainingPoints.size()])),
    				                          new DataSet(mapping,label_mapping,testingPoints.toArray(new DataPoint[testingPoints.size()])), K);
    			System.out.println();
    		}


    		// System.out.println("For Performance on Training Set, function of TP");
    		// double[] results = new double[tp.length];
    		// for (int n = 0; n < runs; n++) {
	     //  		DataSet[] TrainSets = new DataSet[tp.length];
	     //  		DataSet[] TestSets = new DataSet[tp.length];
	    	// 	for (int i = 0; i < tp.length; i++) {
	    	// 		dsTrain = new ArrayList<DataPoint>();
	     //  			dsTests = new ArrayList<DataPoint>();
			   //  	for (int j = 0; j < dataSet.size(); j++) {
		    //   			if (Math.random() < tp[i])
		    //   				dsTrain.add(dataSet.get(j));
		    //   			else {
		    //   				dsTests.add(dataSet.get(j));
		    //   			}
		    //   		}
		    //   		TrainSets[i] = new DataSet(mapping,label_mapping, dsTrain.toArray(new DataPoint[dsTrain.size()]));
		    //   		TestSets[i]  = new DataSet(mapping,label_mapping, dsTests.toArray(new DataPoint[dsTests.size()]));
		    //   	}

		    //   	for (int i = 0; i < TrainSets.length; i++){
		    //   		System.out.println(i);
		    //   		AdaBoost boost = new AdaBoost(TrainSets[i], false);
		    //   		results[i] += (double) boost.NLearners() / (double) runs;
		    //   	}
	    	// }
	    	// for (int i = 0; i < tp.length; i++) {
	    	// 	System.out.printf("%15f   ", tp[i]);
	    	// }
	    	// System.out.println();
	    	// for (int i = 0; i < tp.length; i++) {
	    	// 	System.out.printf("%15f   ", results[i]);
	    	// }
	    	// System.out.println();


	    	// AdaBoost boost = new AdaBoost(training_dataset, print_verbose);
    		// int[] output = boost.Classify(testing_dataset);

    		// int right = 0;
    		// int wrong = 0;
    		// for (int i = 0; i < output.length; i++) {
    		// 	if (output[i] == testing_dataset.KthBinaryDataPoint(i).Label()) {
    		// 		right++;
    		// 	}
    		// 	else {
    		// 		wrong++;
    		// 	}
    		// }
    		// System.out.println("Final AdaBoost Error Rate: " + ((double) wrong / (wrong + right)));
	    }
    	if (decision_stump) {
    	    for (int z = 0; z < 1000; z++) {
	    		double[] w = new double[training_dataset.NDataPoints()];
		    	double val = 1.0 / w.length;
		    	for (int i = 0; i < w.length; i++) {
		    		w[i] = val;
		    	}
		    	DecisionStump stump = new DecisionStump(training_dataset, w, print_verbose);
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
		    	System.out.println(/*"Decision Stump Error Rate: " + */((double) wrong / (wrong + right)));
		    }
	    }
	    if (decision_tree) {}
	    if (naive_bayes) {}
	    if (neural_network) {}
	    if (random_forest) {}
	    if (svm) {
	    	SVM svm = new SVM(training_dataset, print_verbose, 1.0e-4, 1.0);
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

	    if (svmAlphaC) {
	    	// int a_mag_min = -4;
	    	// int a_mag_max = -3;
	    	// double[] alphas = new double[(a_mag_max - a_mag_min + 1)];
	    	// for (int i = 0; i < alphas.length; i++) {
	    	// 	alphas[i] = Math.pow(10, a_mag_min + i);
	    	// 	//alphas[2*i + 1] = 5 * Math.pow(10, a_mag_min + i);
	    	// }
	    	double[] alphas = {1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2};

	    	// int c_mag_min = -3;
	    	// int c_mag_max = 3;
	    	// double[] Cs = new double[(c_mag_max - c_mag_min + 1)];
	    	// for (int i = 0; i < Cs.length; i++) {
	    	// 	Cs[i] = Math.pow(10, c_mag_min + i);
	    	// }
	    	double[] Cs = {1.0e-9, 1.0e-5, 1.0e-2, 1.0, 1.0e2, 1.0e3, 1.0e5, 1.0e9};

	    	for (int i = 0; i < alphas.length; i++) {
	    		for (int j = 0; j < Cs.length; j++) {
		    		SVM svm = new SVM(training_dataset, print_verbose, alphas[i], Cs[j]);
			    	int[] output = svm.Classify(testing_dataset);

			    	int right = 0;
			    	int wrong = 0;
			    	for (int k = 0; k < output.length; k++) {
			    		if (output[k] == testing_dataset.KthBinaryDataPoint(k).Label()) {
			    			right++;
			    		}
			    		else {
			    			wrong++;
			    		}
			    	}
			    	System.out.println("SVM Error Rate (alpha = " + alphas[i] + ", C = " + Cs[j] + "): " + ((double) wrong / (wrong + right)));
			    }
		    }
	    }

	    if (svmCNoise) {
	    	int reps = 1;

	    	double noise_min = 0.0;
	    	double noise_max = 1.0;
	    	double noise_stp = 0.05;

	    	double[] noise = new double[(int) ((noise_max - noise_min)/noise_stp) + 1];
	    	for (int i = 0; i < noise.length; i++) {
	    		noise[i] = noise_min + i*noise_stp;
	    	}

	    	int C_mag_min = -5;
	    	int C_mag_max = 10;
	    	double[] Cs = new double[C_mag_max - C_mag_min + 1];
	    	System.out.printf("%15e    ", 9.999999999e99);
	    	for (int i = 0; i < Cs.length; i++) {
	    		Cs[i] = Math.pow(10, C_mag_min + i);
	    		System.out.printf("%15e    ",Cs[i]);
	    	}
	    	System.out.println();
	    	double alpha = 1.0e-4;


	    	DataPoint[] data_points_train = dsTrain.toArray(new DataPoint[dsTrain.size()]);
			DataPoint[] data_points_tests = dsTests.toArray(new DataPoint[dsTests.size()]);

			double[][] results = new double[noise.length][Cs.length];

			for (int n = 0; n < reps; n++) {
				DataSet[] trainingSets = new DataSet[noise.length];
				for (int i = 0; i < trainingSets.length; i++) {
					DataPoint[] pts = new DataPoint[data_points_train.length];
					for (int j = 0; j < data_points_train.length; j++){
						if (Math.random() < noise[i]) {
							pts[j] = new DataPoint(data_points_train[j].Attributes(), 
														 1 - data_points_train[j].Label(), 
														 data_points_train[j].isBinary());
						}
						else {
							pts[j] = new DataPoint(data_points_train[j]);
						}	
					}
					trainingSets[i] = new DataSet(mapping, label_mapping, pts);
				}


		    	
		    	//double[] Cs = {1.0e-6, 1.0, 1.0e6};

		    	for (int i = 0; i < trainingSets.length; i++) {
		    		System.out.printf("%15e    ", noise[i]);
		    		for (int j = 0; j < Cs.length; j++) {
			    		SVM svm = new SVM(trainingSets[i], print_verbose, alpha, Cs[j]);
				    	int[] output = svm.Classify(testing_dataset);

				    	int right = 0;
				    	int wrong = 0;
				    	for (int k = 0; k < output.length; k++) {
				    		if (output[k] == testing_dataset.KthBinaryDataPoint(k).Label()) {
				    			right++;
				    		}
				    		else {
				    			wrong++;
				    		}
				    	}
				    	results[i][j] += ((double) wrong / (wrong + right)) / reps;
				    	System.out.printf("%15e    ", results[i][j]);
				    	//System.out.println("SVM Error Rate (C = " + Cs[i] + ", noise = " + noise[j] + "): " + ((double) wrong / (wrong + right)));
			    	}
			    	System.out.println();
			    }
			}
			// for (int i = 0; i < results.length; i++) {
			// 	System.out.printf("%15e    ", noise[i]);
			// 	for (int j = 0; j < results[0].length; j++) {
			// 		System.out.printf("%15e    ", results[i][j]);
			// 	}
			// 	System.out.println();
			// }

	    }

	    if (weighted_majority) {}

	}

}

