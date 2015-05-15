package MachineLearning.TreeBagging;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Vector;
import java.util.HashMap;
import MachineLearning.*;

public class TreeBagging implements Learner {
  private final boolean print_verbose;
  private final DataSet train;
  private final int forest_size;
  private TBTree[] trees;
  
  // create the actual tree structure
  private class TBTree { 
    int label;
    int[] splitting_attributes;
    HashMap<List<Integer>, TBTree> subtrees;
    
    // tree that is a leaf has no root test
    private TBTree(int label) {
      this.label = label;
      this.splitting_attributes = null;
      this.subtrees = null;
    }
    
    public TBTree(int[] splitting_attributes) {
      this.splitting_attributes = new int[splitting_attributes.length];
      for (int i = 0; i < splitting_attributes.length; ++i)
        this.splitting_attributes[i] = splitting_attributes[i];
      this.label = -1;
      subtrees = new HashMap<List<Integer>, TBTree>();
    }
    
    // add the subtree to this tree
    public void AddSubTree(List<Integer> key, TBTree tree) {
      subtrees.put(key, tree);
    }
    
    // return the subtree 
    public TBTree SubTree(List<Integer> key) {
      return subtrees.get(key);
    }
    
    // print the tree
    public String toString() {
      return "";
    } 
  }
  
  // see if all of the examples have the same label
  private boolean SameClassification(Vector<DataPoint> examples) {
    int label = examples.get(0).Label();
    for (int i = 1; i < examples.size(); ++i)
      if (examples.get(i).Label() != label) return false;
    return true;
  }
  
  // return the largest example in the parent class
  private int PluralityValue(Vector<DataPoint> parent_examples) {
    int[] noccurences = new int[train.NLabels()];
    for (int i = 0; i < parent_examples.size(); ++i)
      noccurences[parent_examples.get(i).Label()]++;
    int maxoccurences = 0;
    int label = 0;
    for (int i = 0; i < noccurences.length; ++i) {
      if (maxoccurences < noccurences[i]) {
        maxoccurences = noccurences[i];
        label = i;
      }
    }
    return label;
  }
  
  private double log2(double q) {
    return Math.log(q) / Math.log(2);
  }
  
  private double B(double q) {
    if (q == 0.0 || q == 1.0) return 0.0;
    else return -1.0 * (q * log2(q) + (1 - q) * log2(1 - q));
  }
  
  private double Remainder(int A, Vector<DataPoint> examples) {
    // get the total number of positive and negative examples
    int p = 0; 
    int n = 0;
    for (int i = 0; i < examples.size(); ++i) {
      if (examples.get(i).Label() == 1) p++;
      else n++;
    }
    
    double remainder = 0.0;
    // go through every possible attribute value
    for (int k = 0; k < train.NClassesKthAttribute(A); ++k) {
      // keep track of the number of positive and negative examples for this attribute
      int pk = 0;
      int nk = 0;
      // go through each example
      for (int i = 0; i < examples.size(); ++i) {
        if (examples.get(i).KthAttribute(A) == k) {
          if (examples.get(i).Label() == 1) pk++;
          else nk++;
        }
      }
      if (pk != 0.0 || nk != 0.0) remainder += ((double) (pk + nk)) / ((double) (p + n)) * B(((double) pk) / ((double) (pk + nk)));
    }
    return remainder;
  }
  
  private int[] Importance(Vector<Integer> attributes, Vector<DataPoint> examples) {
    // return an array of size one for decision trees
    int A = -1;
    double least_remainder = Double.MAX_VALUE;
    for (int i = 0; i < attributes.size(); ++i) {
      int a = attributes.get(i);
      double remainder = Remainder(a, examples);
      if (remainder < least_remainder) {
        least_remainder = remainder;
        A = a;
      }
    }
    int[] ret = {A};
    return ret;
  }
  
  private TBTree DecisionTreeLearning(Vector<DataPoint> examples,  Vector<Integer> attributes, Vector<DataPoint> parent_examples) {
    // if examples is empty then return Plurality-Value(parent_examples)
    if (examples.size() == 0) return new TBTree(PluralityValue(parent_examples));
    // else if all examples have the same classification then return the classification
    else if (SameClassification(examples)) return new TBTree(examples.get(0).Label());
    // else if attributes is empty then return Plurality-Value(examples)
    else if (attributes.size() == 0) return new TBTree(PluralityValue(examples));
    // or else
    else {
      // A <- argmax a \in attributes Importance(a, examples)
      int[] A = Importance(attributes, examples);
      
      // create a list of attributes without A
      Vector<Integer> attributes_sans_A = new Vector<Integer>();
      for (int ia = 0; ia < attributes.size(); ++ia) {
        boolean include = true;
        for (int ja = 0; ja < A.length; ++ja) {
          if (A[ja] == attributes.get(ia)) include = false;
        }
        if (include) attributes_sans_A.addElement(attributes.get(ia));
      }
      TBTree tree = new TBTree(A);
      RecursiveMethod(A, new Integer[0], A, examples, attributes_sans_A, tree);
      return tree;
    }
  }
  
  public int[] RemoveFirst(int[] a) {
    int[] b = new int[a.length - 1];
    for (int i = 0; i < b.length; ++i) {
      b[i] = a[i + 1];
    }
    return b;
  }
  
  public Integer[] Concat(Integer[] a, Integer[] b) {
    Integer[] c = new Integer[a.length + b.length];
    for (int i = 0; i < a.length; ++i)
      c[i] = a[i];
    for (int i = 0; i < b.length; ++i)
      c[i + a.length] = b[i];
    return c;
  }
  
  public void RecursiveMethod(int[] attributes_remaining, Integer[] attributes_determined, int[] A, Vector<DataPoint> examples, Vector<Integer> attributes, TBTree tree) {
    if (attributes_remaining.length == 0) {
      // get all of the examples that belong to this pattern
      Vector<DataPoint> exs = new Vector<DataPoint>();
      for (int e = 0; e < examples.size(); ++e) {
        DataPoint data_point = examples.get(e);
        boolean include = true;
        for (int k = 0; k < attributes_determined.length; ++k) {
          if (data_point.KthAttribute(A[k]) != attributes_determined[k]) include = false;
        }
        if (include) exs.addElement(new DataPoint(data_point));
      }
      tree.AddSubTree(Arrays.asList(attributes_determined), DecisionTreeLearning(exs, attributes, examples));
    }
    else {
      int attribute = attributes_remaining[0];
      for (int i = 0; i < train.NClassesKthAttribute(attribute); ++i) {
        Integer[] this_attribute = {i};
        RecursiveMethod(RemoveFirst(attributes_remaining), Concat(attributes_determined, this_attribute), A, examples, attributes, tree);
      }
    }
  }
  
  // constructor for TreeBagging
  public TreeBagging(DataSet train, boolean print_verbose, int forest_size) { 
    this.print_verbose = print_verbose;
    this.train = train;
    this.forest_size = forest_size;
    this.trees = new TBTree[forest_size];
    
    Vector<DataPoint> parent_examples = new Vector<DataPoint>();
    Vector<Integer> attributes = new Vector<Integer>();
    for (int i = 0; i < train.NAttributes(); ++i) {
      attributes.addElement(i);
    }
    for (int i = 0; i < trees.length; ++i) {
      Vector<DataPoint> examples = new Vector<DataPoint>();
      for (int j = 0; j < train.NDataPoints(); ++j) {
        int element = (int) (Math.random() * train.NDataPoints());
        examples.addElement(new DataPoint(train.KthDataPoint(element)));
      }
      trees[i] = DecisionTreeLearning(examples, attributes, parent_examples);
    }
  }
  
  // constructor for RandomForest
  public TreeBagging(DataSet train, boolean print_verbose) {
    this(train, print_verbose, 1000);
  }
  
  // classify example
  private int ClassifyExample(DataPoint data_point, TBTree tree) {
    if (tree.label != -1) return tree.label;
    else {
      int[] tree_splitting_value = tree.splitting_attributes;
      Integer[] splitting_value = new Integer[tree_splitting_value.length];
      for (int i = 0; i < tree_splitting_value.length; ++i) {
        splitting_value[i] = data_point.KthAttribute(tree_splitting_value[i]);
      }
      return ClassifyExample(data_point, tree.SubTree(Arrays.asList(splitting_value)));
    }
  }
  
  // classify a particular data
  public int[] Classify(DataSet test) {
    int[] labels = new int[test.NDataPoints()];
    
    for (int i = 0; i < test.NDataPoints(); ++i) {
      int[] noccurences = new int[test.NLabels()];
      for (int j = 0; j < trees.length; ++j) {
        noccurences[ClassifyExample(test.KthDataPoint(i), trees[j])]++;
      }
      // see which occurence is dominant
      int current_max = Integer.MIN_VALUE;
      for (int j = 0; j < noccurences.length; ++j) {
        if (noccurences[j] > current_max) {
          current_max = noccurences[j];
          labels[i] = j;
        }
      }
    }
    
    return labels;
  }
}