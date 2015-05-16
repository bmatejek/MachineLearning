package MachineLearning;

import MachineLearning.*;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Vector;
import java.util.Collections;

public class GenerateMonks {
  private static final int num_attributes = 12;
  private static final int[] attributes_size = {3, 3, 2, 3, 4, 2, 5, 3, 4, 4, 3, 2};
  private static final String[][] mapping = {
    {"a1", "1", "2", "3"},
    {"a2", "1", "2", "3"},
    {"a3", "1", "2"},
    {"a4", "1", "2", "3"},
    {"a5", "1", "2", "3", "4"},
    {"a6", "1", "2"},
    {"a7", "1", "2", "3", "4", "5"},
    {"a8", "1", "2", "3"},
    {"a9", "1", "2", "3", "4"},
    {"a10", "1", "2", "3", "4"},
    {"a11", "1", "2", "3"},
    {"a12", "1", "2"}
  };
  private static final String[] label_mapping = {"0", "1"};
  private static final String[] output_files = { 
    "MachineLearning/data/monks/new-monks-1",
    "MachineLearning/data/monks/new-monks-2",
    "MachineLearning/data/monks/new-monks-3",
    "MachineLearning/data/monks/new-monks-4",
    "MachineLearning/data/monks/new-monks-5",
    "MachineLearning/data/monks/new-monks-6",
    "MachineLearning/data/monks/new-monks-7",
    "MachineLearning/data/monks/new-monks-8"
  };
  
  private static int FunctionOne(int[] a) {
    if (a[2] * a[2] - a[4] == 0) return 1;
    else if (a[2] * a[2] - a[8] == 0) return 1;
    else return 0;
  }
  
  private static int FunctionTwo(int[] a) {
    if (a[11] == 1) return 1;
    else return 0;
  }
  
  private static int FunctionThree(int[] a) {
    int total_ones = 0;
    for (int i = 0; i < a.length; ++i) {
      if (a[i] == 1)
        total_ones++;
    }
    if (total_ones >= a.length / 2) return 1;
    else return 0;
  }
  
  private static int FunctionFour(int[] a) {
    if ((a[6] * a[6] * a[6] * a[6] - a[5] * a[5] * a[5] <= 128) && (a[6] * a[6] * a[6] * a[6] - a[5] * a[5] * a[5] >= 8)) return 1;
    else return 0;
  }
  
  private static int FunctionFive(int[] a) {
    if (Math.abs(a[6] - a[10]) == 1) return 1;
    else return 0;
  }
  
  private static int FunctionSix(int[] a) {
    boolean no_fours = true;
    boolean no_threes = true;
    for (int i = 0; i < a.length; ++i) {
      if (a[i] == 4) no_fours = false;
      if (a[i] == 3) no_threes = false;
    }
    if (no_fours || no_threes) return 1;
    else return 0;
  }
  
  private static int FunctionSeven(int[] a) {
    if (Math.sin(a[0] - a[5] + a[3] - a[4]) <= 0.0) return 1;
    else return 0;
  }
  
  private static int FunctionEight(int[] a) {
    if (a[9] > a[8]) return 1;
    else return 0;
  }
  
  public static void main(String[] args) {
    int train_test_instance = 10000;
    BufferedWriter[] out = new BufferedWriter[8];
    try {
      for (int i = 0; i < 8; ++i) {
        FileWriter fstream = new FileWriter(output_files[i], false);
        out[i] = new BufferedWriter(fstream);
      }
    }
    catch (Exception e) {
      System.out.println("Shoot!");
    }
    // generate ten thousand training and testing variables
    for (int i = 0; i < train_test_instance; ++i) {
      // create a random set of attributes
      int[] attributes = new int[num_attributes];
      int[] labels = new int[8];
      
      for (int j = 0; j < attributes.length; ++j) {
        attributes[j] = 1 + (int) (attributes_size[j] * Math.random());
      }
      
      labels[0] = FunctionOne(attributes);
      labels[1] = FunctionTwo(attributes);
      labels[2] = FunctionThree(attributes);
      labels[3] = FunctionFour(attributes);
      labels[4] = FunctionFive(attributes);
      labels[5] = FunctionSix(attributes);
      labels[6] = FunctionSeven(attributes);
      labels[7] = FunctionEight(attributes);
      
      for (int j = 0; j < 8; ++j) {
        try {
          for (int k = 0; k < attributes.length; ++k)
            out[j].write(Integer.toString(attributes[k]) + " ");
          out[j].write(Integer.toString(labels[j]));
          out[j].write("\n");
        }
        catch (Exception e) {
          System.out.println("Shoot!");
        }
      }
    }
    
    try {
      for (int i = 0; i < 8; ++i) {
        out[i].close();
      }
    } catch (Exception e) {
      System.out.println("Shoot!");
    }
  }
}