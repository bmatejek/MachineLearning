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

public class mushroom {


	
	public static void main(String args[]) {

		String line;
		String fileName = "data/mushroom/agaricus-lepiota";
		ArrayList<String> labels       = new ArrayList<String>();
		ArrayList<String> labelSymbols = new ArrayList<String>();
		ArrayList<String> categories   = new ArrayList<String>();
		ArrayList<ArrayList<String>> attributes = new ArrayList<ArrayList<String>>();
		ArrayList<ArrayList<String>> symbols    = new ArrayList<ArrayList<String>>();
		
		// Read in strings for dataset
		try {
			BufferedReader brNames = new BufferedReader(
										new FileReader(fileName + ".names"));

			line = brNames.readLine();
			while (!line.contains("Attribute Information")) {

				line = brNames.readLine();

			}

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
				while(categoryMatcher.find()) {
					categories.add(categoryMatcher.group(1));
					//System.out.println(categoryMatcher.group(1));
				}

				// Add attributes to attribute list
				ArrayList<String> temp = new ArrayList<String>();
				while(attributeMatcher.find()) {
					temp.add(attributeMatcher.group(2));
					//System.out.println(attributeMatcher.group(2));
				}
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
			BufferedReader brData = new BufferedReader(
										new FileReader(fileName + ".data"));

			ArrayList<DataPoint> dataSet = new ArrayList<DataPoint>();
			while ((line = brData.readLine()) != null) {
				String[] elmts = line.split(",");
				int label = labelSymbols.indexOf(elmts[0]);
				int[] attribute = new int[elmts.length - 1];
				for (int i = 1; i < elmts.length; i++) {
					attribute[i-1] = symbols.get(i-1).indexOf(elmts[i]);
				}
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