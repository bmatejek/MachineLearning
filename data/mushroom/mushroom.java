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
import java.util.regex.*;

public class Mushroom {


	
	public static void main(String args[]) {

		String fileName = "agaricus-lepiota";
		ArrayList<String> labels       = new ArrayList<String>();
		ArrayList<String> categories   = new ArrayList<String>();
		ArrayList<String[]> attributes = new ArrayList<String[]>();
		ArrayList<String[]> symbols    = new ArrayList<String[]>();
		
		// Read in strings for dataset
		try {
			BufferedReader brNames = new BufferedReader(
										new FileReader(fileName + ".names"));

			String line = brNames.readLine();
			while (!line.contains("Attribute Information")) {

				line = brNames.readLine();

			}

			Pattern labelPattern = Pattern.compile("\\p{Blank}(\\p{Alnum}*?)=");
			Matcher labelMatcher = labelPattern.matcher(line);
			while (labelMatcher.find()) {
				labels.add(labelMatcher.group(1));
			}
			line = brNames.readLine();

			Pattern categoryPattern  = Pattern.compile("\\p{Digit}.\\p{Blank}((\\p{Alnum}|-)*?\\p{Punct}?):");
			Pattern attributePattern = Pattern.compile("(\\p{Blank}|,)(\\p{Alnum}*?)=");
			Pattern symbolPattern    = Pattern.compile("=(\\p{Alnum})");
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
				attributes.add(temp.toArray(new String[temp.size()]));

				// Add symbols to symbol array
				temp = new ArrayList<String>();
				while(symbolMatcher.find()) {
					temp.add(symbolMatcher.group(1));
				}
				symbols.add(temp.toArray(new String[temp.size()]));

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
			int temp = 0;
			while ((line = brData.readLine()) != null) {
				String[] elmts = line.split(",");
				
			}
		}

		catch (IOException e) {
			System.out.println("LALALALA FILL ME IN.");
		}

	}

}